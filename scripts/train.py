from __future__ import annotations

import argparse
import contextlib
from functools import partial
import math
import re
import sys
import time
from pathlib import Path

import torch
from torch import amp
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distilled_clamp.data import (
    PreprocessedDistillDataset,
    collate_preprocessed_batch,
    list_preprocessed_files,
)
from distilled_clamp.losses import joint_loss, reconstruction_smooth_loss
from distilled_clamp.models import build_distilled_student_from_cfg
from distilled_clamp.utils import ensure_dir, get_device, load_yaml, set_seed


def _optimizer_unique_param_numel(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    seen: set[int] = set()
    total = 0
    for g in optimizer.param_groups:
        for p in g["params"]:
            i = id(p)
            if i not in seen:
                seen.add(i)
                total += p.numel()
    return len(seen), total


def _optimizer_state_numel(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    n_tensors = 0
    n_elems = 0
    for st in optimizer.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                n_tensors += 1
                n_elems += v.numel()
    return n_tensors, n_elems


def _log_adam_footprint(tag: str, optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> None:
    n_par, par_nel = _optimizer_unique_param_numel(optimizer)
    n_st, st_nel = _optimizer_state_numel(optimizer)
    m_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if st_nel == 0:
        note = (
            f" | opt_state not allocated yet (PyTorch creates Adam buffers on first step; "
            f"expect ~2×param elems ≈ {2 * par_nel:,})"
        )
    else:
        note = f" | opt_state={n_st} tensors, {st_nel:,} elems"
    print(
        f"[train_from_preprocessed] {tag} AdamW: opt_params={n_par} tensors, {par_nel:,} elems | "
        f"model_trainable={m_tr:,} elems{note}"
    )


def _build_train_loader(
    cfg: dict,
    train_files: list[Path],
    *,
    pad_token_id: int,
) -> tuple[PreprocessedDistillDataset, DataLoader, int]:
    if not train_files:
        raise RuntimeError("No train files (empty list)")
    train_ds = PreprocessedDistillDataset(
        train_files,
        max_tokens=0,
        min_train_tokens=0,
        random_crop=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=bool(cfg["data"]["drop_last"]),
        collate_fn=partial(collate_preprocessed_batch, pad_token_id=pad_token_id),
    )
    return train_ds, train_loader, len(train_files)


def _build_val_loader(
    cfg: dict,
    val_root: Path,
    *,
    pad_token_id: int,
) -> tuple[PreprocessedDistillDataset, DataLoader, int]:
    val_files = list_preprocessed_files(val_root)
    if not val_files:
        raise RuntimeError(f"No val files found in {val_root}")
    val_ds = PreprocessedDistillDataset(
        val_files,
        max_tokens=0,
        min_train_tokens=0,
        random_crop=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=False,
        collate_fn=partial(collate_preprocessed_batch, pad_token_id=pad_token_id),
    )
    return val_ds, val_loader, len(val_files)


def _validate_token_id_range(files: list[Path], vocab_size: int, split_name: str) -> None:
    bad_examples = []
    observed_min = None
    observed_max = None
    for src_path in tqdm(files, desc=f"scan-{split_name}", leave=False):
        blob = torch.load(src_path, map_location="cpu")
        if not isinstance(blob, dict) or "token_ids" not in blob:
            raise RuntimeError(
                f"Invalid preprocessed sample format at {src_path}; expected dict with 'token_ids'"
            )
        source = torch.as_tensor(blob["token_ids"], dtype=torch.long)
        if source.numel() == 0:
            continue
        cur_min = int(source.min().item())
        cur_max = int(source.max().item())
        observed_min = cur_min if observed_min is None else min(observed_min, cur_min)
        observed_max = cur_max if observed_max is None else max(observed_max, cur_max)
        if cur_min < 0 or cur_max >= vocab_size:
            bad_examples.append((str(src_path), cur_min, cur_max))
            if len(bad_examples) >= 5:
                break

    if bad_examples:
        details = "\n".join(
            f"- {p} (min_id={mn}, max_id={mx}, expected_range=[0,{vocab_size - 1}])"
            for p, mn, mx in bad_examples
        )
        raise RuntimeError(
            "Out-of-range token IDs detected in preprocessed source tensors.\n"
            "This usually means caches were built with an older vocab definition.\n"
            "Please re-run preprocessing with current config before training.\n"
            f"Examples:\n{details}"
        )
    if observed_min is not None and observed_max is not None:
        print(
            f"[data-check] split={split_name} token_id_range=[{observed_min},{observed_max}] "
            f"expected=[0,{vocab_size - 1}]"
        )


def _amp_autocast(device: torch.device, use_amp: bool):
    if device.type == "cuda" and use_amp:
        return amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _resolve_lambda_recon(
    loss_cfg: dict,
    *,
    global_step: int,
    max_steps: int,
) -> float:
    base = float(loss_cfg.get("lambda_recon", 1.0))
    sched = loss_cfg.get("lambda_recon_schedule")
    if not isinstance(sched, dict):
        return base
    kind = str(sched.get("type", "linear")).strip().lower()
    final = float(sched.get("final", base))
    steps = int(sched.get("steps", max_steps))
    if steps <= 0:
        return final
    progress = min(1.0, max(0.0, float(global_step) / float(steps)))
    if kind == "linear":
        return base + (final - base) * progress
    if kind == "cosine":
        return final + (base - final) * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unsupported loss.lambda_recon_schedule.type={kind!r}; use linear|cosine")


def run_validation(
    cfg: dict,
    model,
    loader,
    device,
    *,
    use_amp: bool,
    pad_token_id: int,
    lambda_recon_override: float | None = None,
) -> dict:
    val_start = time.perf_counter()
    model.eval()
    total = 0.0
    agg = {
        "L_recon": 0.0,
        "L_distill": 0.0,
        "L_soft": 0.0,
        "L_rkd": 0.0,
        "L_distill_unbanked": 0.0,
        "L_soft_unbanked": 0.0,
        "L_rkd_unbanked": 0.0,
        "L_distill_banked": 0.0,
        "L_soft_banked": 0.0,
        "L_rkd_banked": 0.0,
        "L_smooth": 0.0,
        "alignment_cos": 0.0,
        "alignment_frac_above_threshold": 0.0,
    }
    steps = 0
    val_accum_steps = max(1, int(cfg.get("optim", {}).get("accumulation_steps", 1)))
    val_bank_student: list[torch.Tensor] = []
    val_bank_teacher: list[torch.Tensor] = []
    val_accum_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            token_ids = batch["token_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            dec_tgt = batch["decoder_target_sequence"].to(device)
            teacher_tgt = batch["teacher_global_target"].to(device)
            if int(token_ids.max().item()) > int(pad_token_id):
                raise ValueError(
                    f"validation token_ids exceed pad token id: max={int(token_ids.max().item())} "
                    f"pad_token_id={int(pad_token_id)}"
                )
            with _amp_autocast(device, use_amp):
                out = model(
                    token_ids,
                    mask,
                    dec_tgt,
                    inputs_are_one_hot=False,
                    smooth_noise_std=float(cfg["loss"].get("smooth_noise_std", 0.0)),
                )
                loss, d = joint_loss(
                    logits=out["logits"],
                    decoder_targets=dec_tgt,
                    decoder_valid_mask=mask,
                    student_global=out["student_global"],
                    teacher_global=teacher_tgt,
                    logits_noisy=out.get("logits_noisy"),
                    lambda_recon=(
                        float(lambda_recon_override)
                        if lambda_recon_override is not None
                        else float(cfg["loss"]["lambda_recon"])
                    ),
                    lambda_soft=float(cfg["loss"].get("lambda_soft", 1.0)),
                    lambda_rkd=float(cfg["loss"].get("lambda_rkd", 1.0)),
                    lambda_smooth=float(cfg["loss"].get("lambda_smooth", 0.0)),
                    smooth_loss_type=str(cfg["loss"].get("smooth_loss_type", "kl")),
                    alignment_threshold=float(cfg["loss"].get("alignment_threshold", 0.8)),
                    temperature=float(cfg["loss"]["temperature"]),
                    teacher_temperature=float(cfg["loss"].get("teacher_temperature", 0.0)),
                )
                if val_bank_student:
                    distill_student_banked = torch.cat(val_bank_student + [out["student_global"]], dim=0)
                    distill_teacher_banked = torch.cat(val_bank_teacher + [teacher_tgt], dim=0)
                else:
                    distill_student_banked = out["student_global"]
                    distill_teacher_banked = teacher_tgt
                _, d_banked = joint_loss(
                    logits=out["logits"],
                    decoder_targets=dec_tgt,
                    decoder_valid_mask=mask,
                    student_global=distill_student_banked,
                    teacher_global=distill_teacher_banked,
                    logits_noisy=out.get("logits_noisy"),
                    lambda_recon=0.0,
                    lambda_soft=float(cfg["loss"].get("lambda_soft", 1.0)),
                    lambda_rkd=float(cfg["loss"].get("lambda_rkd", 1.0)),
                    lambda_smooth=0.0,
                    smooth_loss_type=str(cfg["loss"].get("smooth_loss_type", "kl")),
                    alignment_threshold=float(cfg["loss"].get("alignment_threshold", 0.8)),
                    temperature=float(cfg["loss"]["temperature"]),
                    teacher_temperature=float(cfg["loss"].get("teacher_temperature", 0.0)),
                )
            total += float(loss.item())
            steps += 1
            agg["L_recon"] += d["L_recon"]
            agg["L_distill"] += d["L_distill"]
            agg["L_soft"] += d.get("L_soft", 0.0)
            agg["L_rkd"] += d.get("L_rkd", 0.0)
            agg["L_distill_unbanked"] += d["L_distill"]
            agg["L_soft_unbanked"] += d.get("L_soft", 0.0)
            agg["L_rkd_unbanked"] += d.get("L_rkd", 0.0)
            agg["L_distill_banked"] += d_banked["L_distill"]
            agg["L_soft_banked"] += d_banked.get("L_soft", 0.0)
            agg["L_rkd_banked"] += d_banked.get("L_rkd", 0.0)
            agg["L_smooth"] += d["L_smooth"]
            agg["alignment_cos"] += d["alignment_cos"]
            agg["alignment_frac_above_threshold"] += d["alignment_frac_above_threshold"]
            is_final_val_micro = (val_accum_count + 1) >= val_accum_steps
            if is_final_val_micro:
                val_bank_student.clear()
                val_bank_teacher.clear()
                val_accum_count = 0
            else:
                val_bank_student.append(out["student_global"].detach())
                val_bank_teacher.append(teacher_tgt.detach())
                val_accum_count += 1
    if steps == 0:
        return {
            "loss": 0.0,
            "L_recon": 0.0,
            "L_distill": 0.0,
            "L_soft": 0.0,
            "L_rkd": 0.0,
            "L_distill_unbanked": 0.0,
            "L_soft_unbanked": 0.0,
            "L_rkd_unbanked": 0.0,
            "L_distill_banked": 0.0,
            "L_soft_banked": 0.0,
            "L_rkd_banked": 0.0,
            "L_smooth": 0.0,
            "alignment_cos": 0.0,
            "alignment_frac_above_threshold": 0.0,
        }
    out = {
        "loss": total / steps,
        "L_recon": agg["L_recon"] / steps,
        "L_distill": agg["L_distill"] / steps,
        "L_soft": agg["L_soft"] / steps,
        "L_rkd": agg["L_rkd"] / steps,
        "L_distill_unbanked": agg["L_distill_unbanked"] / steps,
        "L_soft_unbanked": agg["L_soft_unbanked"] / steps,
        "L_rkd_unbanked": agg["L_rkd_unbanked"] / steps,
        "L_distill_banked": agg["L_distill_banked"] / steps,
        "L_soft_banked": agg["L_soft_banked"] / steps,
        "L_rkd_banked": agg["L_rkd_banked"] / steps,
        "L_smooth": agg["L_smooth"] / steps,
        "alignment_cos": agg["alignment_cos"] / steps,
        "alignment_frac_above_threshold": agg["alignment_frac_above_threshold"] / steps,
    }
    out["duration_sec"] = time.perf_counter() - val_start
    return out


def _latest_step_checkpoint(ckpt_dir: Path) -> Path | None:
    all_ckpt = sorted(ckpt_dir.glob("step_*.pt"))
    if not all_ckpt:
        return None
    return all_ckpt[-1]


def _build_lr_lambda(max_steps: int, warmup_steps: int, lr: float, lr_min: float, use_cosine: bool):
    """Piecewise: linear warmup (LambdaLR ``step`` matches PyTorch ``last_epoch`` after ``step()``)."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step <= warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if not use_cosine:
            return 1.0
        denom = float(max(1, max_steps - warmup_steps))
        progress = float(step - warmup_steps) / denom
        progress = min(1.0, max(0.0, progress))
        min_ratio = float(lr_min) / float(max(lr, 1e-12))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


_LORA_MISSING_RE = re.compile(
    r"^m3_core\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.lora_(a|b)\.weight$"
)


def _remap_linear_lora_state_keys(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    """Map attention Linear<->LoRA key names to allow cross-mode checkpoint loading."""
    model_keys = set(model.state_dict().keys())
    out = dict(state_dict)
    remapped = 0
    for key in list(out.keys()):
        value = out.get(key)
        if value is None:
            continue
        if key.endswith(".base.weight"):
            plain = key[: -len(".base.weight")] + ".weight"
            if plain in model_keys and plain not in out:
                out[plain] = out.pop(key)
                remapped += 1
                continue
        if key.endswith(".base.bias"):
            plain = key[: -len(".base.bias")] + ".bias"
            if plain in model_keys and plain not in out:
                out[plain] = out.pop(key)
                remapped += 1
                continue
        if key.endswith(".weight"):
            base = key[: -len(".weight")] + ".base.weight"
            if base in model_keys and base not in out:
                out[base] = out.pop(key)
                remapped += 1
                continue
        if key.endswith(".bias"):
            base = key[: -len(".bias")] + ".base.bias"
            if base in model_keys and base not in out:
                out[base] = out.pop(key)
                remapped += 1
                continue
    return out, remapped


def _load_model_state_compat(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load model state with LoRA/non-LoRA compatibility for attention projections."""
    remapped_state, remapped_count = _remap_linear_lora_state_keys(model, state_dict)
    missing, unexpected = model.load_state_dict(remapped_state, strict=False)
    tolerated_missing = [k for k in missing if _LORA_MISSING_RE.match(k)]
    hard_missing = [k for k in missing if k not in tolerated_missing]
    if hard_missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch after compatibility remap.\n"
            f"  hard_missing={hard_missing[:20]}\n"
            f"  unexpected={unexpected[:20]}"
        )
    if remapped_count > 0:
        print(f"[resume] remapped {remapped_count} attention keys between base Linear and LoRA modules")
    if tolerated_missing:
        print(
            "[resume] initialized missing LoRA adapter weights from module defaults: "
            f"{len(tolerated_missing)} tensors"
        )


def _module_design_tag(input_module_type: str, *, patch_len: int, perceiver_num_latents: int) -> str:
    kind = str(input_module_type).strip().lower()
    if kind in {"patch", "student", "student_patch"}:
        return f"input_patch_len{int(patch_len)}"
    if kind in {"perceiver", "perceiver_resampling", "resampler"}:
        return f"input_perceiver_lat{int(perceiver_num_latents)}"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in kind)
    return f"input_{safe or 'unknown'}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train distilled_clamp model from preprocessed tensors.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_root", type=str, default="")
    parser.add_argument("--val_root", type=str, default="")
    parser.add_argument("--tb_dir", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument(
        "--weights_only_resume",
        action="store_true",
        help="Load model weights from --resume but start optimizer/scheduler/scaler fresh.",
    )
    parser.add_argument(
        "--rescan-train-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rescan train directory each epoch (pick up new shards). "
        "Default: config data.rescan_train_each_epoch, else false. "
        "Use --no-rescan-train-each-epoch to force off.",
    )
    parser.add_argument(
        "--rescan-val-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rescan val directory each epoch (pick up new shards). "
        "Default: config data.rescan_val_each_epoch, else false. "
        "Use --no-rescan-val-each-epoch to force off.",
    )
    parser.add_argument(
        "--max_training_samples",
        type=int,
        default=-1,
        help="Use at most this many preprocessed train files (0 = no cap, -1 = use config).",
    )
    parser.add_argument(
        "--max_train_pairs",
        type=int,
        default=-1,
        help="Deprecated alias for data.max_training_samples. If set (>=0), overrides config.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    try:
        from anticipation.vocab import CONTROL_OFFSET, VOCAB_SIZE  # type: ignore

        ar_event_vocab_size = int(CONTROL_OFFSET)
        full_vocab_size = int(VOCAB_SIZE)
    except Exception:
        ar_event_vocab_size = int(cfg["source"]["vocab_size"])
        full_vocab_size = ar_event_vocab_size

    train_root = Path(args.train_root) if args.train_root else Path(cfg["paths"]["preprocessed_train_root"])
    val_root = Path(args.val_root) if args.val_root else Path(cfg["paths"]["preprocessed_val_root"])
    val_files = list_preprocessed_files(val_root)
    if not val_files:
        raise RuntimeError(f"No val files found in {val_root}")

    max_training_samples = int(cfg.get("data", {}).get("max_training_samples", 0) or 0)
    if int(args.max_training_samples) >= 0:
        max_training_samples = int(args.max_training_samples)
    elif int(args.max_train_pairs) >= 0:
        max_training_samples = int(args.max_train_pairs)
    train_cap = max_training_samples if max_training_samples > 0 else None
    train_files = list_preprocessed_files(train_root, max_files=train_cap)
    if not train_files:
        raise RuntimeError(f"No train files found in {train_root}")

    _validate_token_id_range(train_files, ar_event_vocab_size, split_name="train")
    _validate_token_id_range(val_files, ar_event_vocab_size, split_name="val")

    cfg_pad_token_id = int(cfg["source"].get("pad_token_id", ar_event_vocab_size))
    if cfg_pad_token_id != int(ar_event_vocab_size):
        raise ValueError(
            "source.pad_token_id must equal source.vocab_size / CONTROL_OFFSET for dedicated pad token. "
            f"got pad_token_id={cfg_pad_token_id} expected={int(ar_event_vocab_size)}"
        )
    pad_token_id = int(cfg_pad_token_id)
    rescan_train_each_epoch = bool(cfg["data"].get("rescan_train_each_epoch", False))
    rescan_val_each_epoch = bool(cfg["data"].get("rescan_val_each_epoch", False))
    if bool(cfg["data"].get("rescan_each_epoch", False)):
        rescan_train_each_epoch = True
        rescan_val_each_epoch = True
    if args.rescan_train_each_epoch is not None:
        rescan_train_each_epoch = bool(args.rescan_train_each_epoch)
    if args.rescan_val_each_epoch is not None:
        rescan_val_each_epoch = bool(args.rescan_val_each_epoch)

    train_ds, train_loader, train_file_count = _build_train_loader(cfg, train_files, pad_token_id=pad_token_id)
    val_ds, val_loader, val_file_count = _build_val_loader(
        cfg, val_root, pad_token_id=pad_token_id
    )

    patch_cfg = cfg.get("patching", {})
    input_cfg = cfg.get("input_module", {})
    input_module_type = str(input_cfg.get("type", patch_cfg.get("input_module_type", "patch")))
    perceiver_num_latents = int(input_cfg.get("perceiver_num_latents", patch_cfg.get("perceiver_num_latents", 64)))
    patch_len = int(patch_cfg["patch_len"])

    model = build_distilled_student_from_cfg(cfg, vocab_size=ar_event_vocab_size + 1).to(device)

    optim_cfg = cfg["optim"]
    base_lr = float(optim_cfg["lr"])
    optim_use_amp = bool(optim_cfg.get("use_amp", True)) and device.type == "cuda"
    accum_steps = int(optim_cfg.get("accumulation_steps", 1))
    warmup_steps = int(optim_cfg.get("warmup_steps", 0))
    lr_min = float(optim_cfg.get("lr_min", 0.0))
    use_cosine = bool(optim_cfg.get("cosine", True))
    max_steps = int(optim_cfg["max_steps"])

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,
        weight_decay=float(optim_cfg["weight_decay"]),
    )
    _log_adam_footprint("init", optimizer, model)
    lr_lambda_fn = _build_lr_lambda(max_steps, warmup_steps, base_lr, lr_min, use_cosine)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fn)
    scaler = GradScaler(enabled=optim_use_amp)

    out_root = ensure_dir(cfg["paths"]["output_root"])
    module_tag = _module_design_tag(
        input_module_type,
        patch_len=patch_len,
        perceiver_num_latents=perceiver_num_latents,
    )
    ckpt_dir = ensure_dir(out_root / "checkpoints" / module_tag)
    tb_dir = ensure_dir(Path(args.tb_dir) if args.tb_dir else out_root / "tensorboard")
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard logs: {tb_dir}")
    print(f"[setup] input_module={input_module_type} module_tag={module_tag} checkpoints={ckpt_dir}")

    global_step = 0
    best_val = float("inf")
    start_epoch = 1
    resume_path: Path | None = Path(args.resume) if args.resume else None
    if args.auto_resume and resume_path is None:
        resume_path = _latest_step_checkpoint(ckpt_dir)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        _load_model_state_compat(model, state["model"])
        resumed_optimizer = False
        if args.weights_only_resume:
            print("[resume] --weights_only_resume enabled; skipping optimizer/scheduler/scaler state restore")
        elif "optimizer" in state:
            try:
                optimizer.load_state_dict(state["optimizer"])
                resumed_optimizer = True
                _log_adam_footprint("after_resume", optimizer, model)
            except Exception as exc:
                print(
                    "[resume] optimizer state incompatible with current trainable parameters; "
                    f"continuing with fresh optimizer/scheduler/scaler ({type(exc).__name__}: {exc})"
                )
        if resumed_optimizer:
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            if optim_use_amp and "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            global_step = int(state.get("global_step", 0))
            best_val = float(state.get("best_val_loss", best_val))
            start_epoch = int(state.get("epoch", 1))
            print(f"Resumed full state from: {resume_path} at global_step={global_step}")
        else:
            print(f"Loaded model weights only from: {resume_path}; starting optimizer state fresh")

    log_every_steps = int(cfg["log"]["log_every_steps"])
    save_every_steps = int(cfg["log"]["save_every_steps"])
    grad_clip = float(optim_cfg["grad_clip"])
    train_batches_per_epoch = len(train_loader)
    val_batches = len(val_loader)
    remaining_steps = max(0, max_steps - global_step)
    print(
        "[setup] "
        f"device={device} train_root={train_root} val_root={val_root} "
        f"train_files={train_file_count}"
        + (
            f" (capped from dir, max_training_samples={max_training_samples})"
            if max_training_samples > 0
            else ""
        )
        + f" val_files={val_file_count} "
        f"train_batches/epoch={train_batches_per_epoch} val_batches={val_batches}"
    )
    print(
        "[setup] "
        f"full_vocab={full_vocab_size} ar_event_vocab={ar_event_vocab_size} "
        "(bridge uses AR-valid event sub-vocab only)"
    )
    print(
        "[setup] "
        f"log_every_steps={log_every_steps} save_every_steps={save_every_steps} "
        f"max_steps={max_steps} remaining_steps={remaining_steps} grad_clip={grad_clip} "
        f"use_amp={optim_use_amp} accum_steps={accum_steps} warmup_steps={warmup_steps} "
        f"lr_min={lr_min} cosine={use_cosine}"
    )
    print("[setup] train uses full preprocessed chunks (length defined by source.preproc_chunk_min/max)")
    print(
        f"[setup] data.rescan_train_each_epoch={rescan_train_each_epoch} "
        f"data.rescan_val_each_epoch={rescan_val_each_epoch}"
    )

    epoch = start_epoch
    done = False
    logged_adam_after_first_step = False
    train_start = time.perf_counter()
    last_step_time = train_start
    while not done:
        if rescan_train_each_epoch:
            rescan_files = list_preprocessed_files(train_root, max_files=train_cap)
            train_ds, train_loader, train_file_count = _build_train_loader(
                cfg, rescan_files, pad_token_id=pad_token_id
            )
            train_batches_per_epoch = len(train_loader)
            print(
                f"[epoch] refresh epoch={epoch} train_files={train_file_count} "
                f"train_batches/epoch={train_batches_per_epoch}"
            )
        if rescan_val_each_epoch:
            val_ds, val_loader, val_file_count = _build_val_loader(
                cfg, val_root, pad_token_id=pad_token_id
            )
            val_batches = len(val_loader)
            print(
                f"[epoch] refresh val epoch={epoch} val_files={val_file_count} "
                f"val_batches={val_batches}"
            )
        if not rescan_train_each_epoch:
            print(
                f"[epoch] start epoch={epoch} train_files={train_file_count} "
                f"train_batches/epoch={train_batches_per_epoch} global_step={global_step}"
            )
        model.train()
        running = {
            "loss": 0.0,
            "L_recon": 0.0,
            "L_soft": 0.0,
            "L_rkd": 0.0,
            "L_distill": 0.0,
            "L_soft_unbanked": 0.0,
            "L_rkd_unbanked": 0.0,
            "L_distill_unbanked": 0.0,
            "L_soft_banked": 0.0,
            "L_rkd_banked": 0.0,
            "L_distill_banked": 0.0,
            "L_smooth": 0.0,
            "alignment_cos": 0.0,
            "alignment_frac_above_threshold": 0.0,
        }
        window = 0
        accum_count = 0
        micro_losses: list[float] = []
        micro_recon: list[float] = []
        micro_soft: list[float] = []
        micro_rkd: list[float] = []
        micro_distill: list[float] = []
        micro_soft_unbanked: list[float] = []
        micro_rkd_unbanked: list[float] = []
        micro_distill_unbanked: list[float] = []
        micro_soft_banked: list[float] = []
        micro_rkd_banked: list[float] = []
        micro_distill_banked: list[float] = []
        micro_smooth: list[float] = []
        micro_align_cos: list[float] = []
        micro_align_frac: list[float] = []
        micro_lambda_recon: list[float] = []
        bank_student: list[torch.Tensor] = []
        bank_teacher: list[torch.Tensor] = []
        optimizer.zero_grad(set_to_none=True)
        lambda_soft = float(cfg["loss"].get("lambda_soft", 1.0))
        lambda_rkd = float(cfg["loss"].get("lambda_rkd", 1.0))
        lambda_smooth = float(cfg["loss"].get("lambda_smooth", 0.0))
        smooth_loss_type = str(cfg["loss"].get("smooth_loss_type", "kl"))
        alignment_threshold = float(cfg["loss"].get("alignment_threshold", 0.8))
        temperature = float(cfg["loss"]["temperature"])
        teacher_temperature = float(cfg["loss"].get("teacher_temperature", 0.0))
        if rescan_train_each_epoch:
            print(f"[epoch] start epoch={epoch} global_step={global_step}")
        for batch in tqdm(train_loader, desc=f"train@epoch{epoch}", leave=False):
            token_ids = batch["token_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            dec_tgt = batch["decoder_target_sequence"].to(device)
            teacher_tgt = batch["teacher_global_target"].to(device)
            lambda_recon_now = _resolve_lambda_recon(
                cfg["loss"],
                global_step=global_step,
                max_steps=max_steps,
            )
            with _amp_autocast(device, optim_use_amp):
                out = model(
                    token_ids,
                    mask,
                    dec_tgt,
                    inputs_are_one_hot=False,
                    smooth_noise_std=float(cfg["loss"].get("smooth_noise_std", 0.0)),
                )
                is_final_micro = (accum_count + 1) >= accum_steps
                core_loss_total, core_d = reconstruction_smooth_loss(
                    logits=out["logits"],
                    decoder_targets=dec_tgt,
                    decoder_valid_mask=mask,
                    logits_noisy=out.get("logits_noisy"),
                    lambda_recon=lambda_recon_now,
                    lambda_smooth=lambda_smooth,
                    smooth_loss_type=smooth_loss_type,
                )
                distill_loss_total = torch.zeros((), device=core_loss_total.device, dtype=core_loss_total.dtype)
                distill_d = {
                    "L_soft": 0.0,
                    "L_rkd": 0.0,
                    "L_distill": 0.0,
                    "alignment_cos": 0.0,
                    "alignment_frac_above_threshold": 0.0,
                }
                if is_final_micro:
                    if bank_student:
                        distill_student = torch.cat(bank_student + [out["student_global"]], dim=0)
                        distill_teacher = torch.cat(bank_teacher + [teacher_tgt], dim=0)
                    else:
                        distill_student = out["student_global"]
                        distill_teacher = teacher_tgt
                    distill_loss_total, distill_d = joint_loss(
                        logits=out["logits"],
                        decoder_targets=dec_tgt,
                        decoder_valid_mask=mask,
                        student_global=distill_student,
                        teacher_global=distill_teacher,
                        logits_noisy=out.get("logits_noisy"),
                        lambda_recon=0.0,
                        lambda_soft=lambda_soft,
                        lambda_rkd=lambda_rkd,
                        lambda_smooth=0.0,
                        smooth_loss_type=smooth_loss_type,
                        alignment_threshold=alignment_threshold,
                        temperature=temperature,
                        teacher_temperature=teacher_temperature,
                    )
            loss = (core_loss_total / float(accum_steps)) + distill_loss_total
            if optim_use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1
            micro_losses.append(float((core_loss_total + distill_loss_total).item()))
            micro_recon.append(core_d["L_recon"])
            micro_smooth.append(core_d["L_smooth"])
            micro_lambda_recon.append(float(lambda_recon_now))
            if is_final_micro:
                _, distill_d_unbanked = joint_loss(
                    logits=out["logits"],
                    decoder_targets=dec_tgt,
                    decoder_valid_mask=mask,
                    student_global=out["student_global"],
                    teacher_global=teacher_tgt,
                    logits_noisy=out.get("logits_noisy"),
                    lambda_recon=0.0,
                    lambda_soft=lambda_soft,
                    lambda_rkd=lambda_rkd,
                    lambda_smooth=0.0,
                    smooth_loss_type=smooth_loss_type,
                    alignment_threshold=alignment_threshold,
                    temperature=temperature,
                    teacher_temperature=teacher_temperature,
                )
                micro_soft_unbanked.append(float(distill_d_unbanked["L_soft"]))
                micro_rkd_unbanked.append(float(distill_d_unbanked["L_rkd"]))
                micro_distill_unbanked.append(float(distill_d_unbanked["L_distill"]))
                micro_soft_banked.append(float(distill_d["L_soft"]))
                micro_rkd_banked.append(float(distill_d["L_rkd"]))
                micro_distill_banked.append(float(distill_d["L_distill"]))
                micro_soft.append(float(distill_d["L_soft"]))
                micro_rkd.append(float(distill_d["L_rkd"]))
                micro_distill.append(float(distill_d["L_distill"]))
                micro_align_cos.append(float(distill_d["alignment_cos"]))
                micro_align_frac.append(float(distill_d["alignment_frac_above_threshold"]))
            else:
                bank_student.append(out["student_global"].detach())
                bank_teacher.append(teacher_tgt.detach())

            if accum_count < accum_steps:
                continue

            if optim_use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if optim_use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if not logged_adam_after_first_step:
                _, st_nel = _optimizer_state_numel(optimizer)
                if st_nel > 0:
                    _log_adam_footprint("after_first_step", optimizer, model)
                    logged_adam_after_first_step = True
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            bank_student.clear()
            bank_teacher.clear()

            global_step += 1
            soft_avg = (sum(micro_soft) / len(micro_soft)) if micro_soft else 0.0
            rkd_avg = (sum(micro_rkd) / len(micro_rkd)) if micro_rkd else 0.0
            distill_avg = (sum(micro_distill) / len(micro_distill)) if micro_distill else 0.0
            soft_unbanked_avg = (sum(micro_soft_unbanked) / len(micro_soft_unbanked)) if micro_soft_unbanked else 0.0
            rkd_unbanked_avg = (sum(micro_rkd_unbanked) / len(micro_rkd_unbanked)) if micro_rkd_unbanked else 0.0
            distill_unbanked_avg = (
                (sum(micro_distill_unbanked) / len(micro_distill_unbanked)) if micro_distill_unbanked else 0.0
            )
            soft_banked_avg = (sum(micro_soft_banked) / len(micro_soft_banked)) if micro_soft_banked else 0.0
            rkd_banked_avg = (sum(micro_rkd_banked) / len(micro_rkd_banked)) if micro_rkd_banked else 0.0
            distill_banked_avg = (sum(micro_distill_banked) / len(micro_distill_banked)) if micro_distill_banked else 0.0
            align_cos_avg = (sum(micro_align_cos) / len(micro_align_cos)) if micro_align_cos else 0.0
            align_frac_avg = (sum(micro_align_frac) / len(micro_align_frac)) if micro_align_frac else 0.0
            lambda_recon_avg = (sum(micro_lambda_recon) / len(micro_lambda_recon)) if micro_lambda_recon else 0.0
            running["loss"] += sum(micro_losses) / len(micro_losses)
            running["L_recon"] += sum(micro_recon) / len(micro_recon)
            running["L_soft"] += soft_avg
            running["L_rkd"] += rkd_avg
            running["L_distill"] += distill_avg
            running["L_soft_unbanked"] += soft_unbanked_avg
            running["L_rkd_unbanked"] += rkd_unbanked_avg
            running["L_distill_unbanked"] += distill_unbanked_avg
            running["L_soft_banked"] += soft_banked_avg
            running["L_rkd_banked"] += rkd_banked_avg
            running["L_distill_banked"] += distill_banked_avg
            running["L_smooth"] += sum(micro_smooth) / len(micro_smooth)
            running["alignment_cos"] += align_cos_avg
            running["alignment_frac_above_threshold"] += align_frac_avg
            micro_losses.clear()
            micro_recon.clear()
            micro_soft.clear()
            micro_rkd.clear()
            micro_distill.clear()
            micro_soft_unbanked.clear()
            micro_rkd_unbanked.clear()
            micro_distill_unbanked.clear()
            micro_soft_banked.clear()
            micro_rkd_banked.clear()
            micro_distill_banked.clear()
            micro_smooth.clear()
            micro_align_cos.clear()
            micro_align_frac.clear()
            micro_lambda_recon.clear()
            window += 1

            lr_now = scheduler.get_last_lr()[0]

            if global_step % log_every_steps == 0:
                avg = {k: v / window for k, v in running.items()}
                now = time.perf_counter()
                elapsed = now - train_start
                step_time = (now - last_step_time) / max(log_every_steps, 1)
                steps_done = max(1, global_step)
                avg_step = elapsed / steps_done
                remaining = max(0, max_steps - global_step)
                eta_sec = remaining * avg_step
                eta_min = eta_sec / 60.0
                print(
                    f"[train] step={global_step} loss={avg['loss']:.6f} "
                    f"recon={avg['L_recon']:.6f} "
                    f"soft(u/b)={avg['L_soft_unbanked']:.6f}/{avg['L_soft_banked']:.6f} "
                    f"rkd(u/b)={avg['L_rkd_unbanked']:.6f}/{avg['L_rkd_banked']:.6f} "
                    f"distill(u/b)={avg['L_distill_unbanked']:.6f}/{avg['L_distill_banked']:.6f} "
                    f"smooth={avg['L_smooth']:.6f} align_cos={avg['alignment_cos']:.4f} "
                    f"align@thr={avg['alignment_frac_above_threshold']:.4f} "
                    f"lambda_recon={lambda_recon_avg:.4f} "
                    f"lr={lr_now:.2e} step_time={step_time:.3f}s eta={eta_min:.1f}m"
                )
                writer.add_scalar("train/loss", avg["loss"], global_step)
                writer.add_scalar("train/L_recon", avg["L_recon"], global_step)
                writer.add_scalar("train/L_soft", avg["L_soft"], global_step)
                writer.add_scalar("train/L_rkd", avg["L_rkd"], global_step)
                writer.add_scalar("train/L_distill", avg["L_distill"], global_step)
                writer.add_scalar("train/L_soft_unbanked", avg["L_soft_unbanked"], global_step)
                writer.add_scalar("train/L_rkd_unbanked", avg["L_rkd_unbanked"], global_step)
                writer.add_scalar("train/L_distill_unbanked", avg["L_distill_unbanked"], global_step)
                writer.add_scalar("train/L_soft_banked", avg["L_soft_banked"], global_step)
                writer.add_scalar("train/L_rkd_banked", avg["L_rkd_banked"], global_step)
                writer.add_scalar("train/L_distill_banked", avg["L_distill_banked"], global_step)
                writer.add_scalar("train/L_smooth", avg["L_smooth"], global_step)
                writer.add_scalar("train/alignment_cos", avg["alignment_cos"], global_step)
                writer.add_scalar(
                    "train/alignment_frac_above_threshold",
                    avg["alignment_frac_above_threshold"],
                    global_step,
                )
                writer.add_scalar("train/lambda_recon", lambda_recon_avg, global_step)
                writer.add_scalar("train/lr", lr_now, global_step)
                running = {
                    "loss": 0.0,
                    "L_recon": 0.0,
                    "L_soft": 0.0,
                    "L_rkd": 0.0,
                    "L_distill": 0.0,
                    "L_soft_unbanked": 0.0,
                    "L_rkd_unbanked": 0.0,
                    "L_distill_unbanked": 0.0,
                    "L_soft_banked": 0.0,
                    "L_rkd_banked": 0.0,
                    "L_distill_banked": 0.0,
                    "L_smooth": 0.0,
                    "alignment_cos": 0.0,
                    "alignment_frac_above_threshold": 0.0,
                }
                window = 0
                last_step_time = now

            if global_step % save_every_steps == 0:
                print(f"[checkpoint] step={global_step} running validation...")
                val_ds, val_loader, val_file_count = _build_val_loader(
                    cfg, val_root, pad_token_id=pad_token_id
                )
                print(
                    f"[val-refresh] step={global_step} val_files={val_file_count} "
                    f"val_batches={len(val_loader)}"
                )
                lambda_recon_val = _resolve_lambda_recon(
                    cfg["loss"],
                    global_step=global_step,
                    max_steps=max_steps,
                )
                val = run_validation(
                    cfg,
                    model,
                    val_loader,
                    device,
                    use_amp=optim_use_amp,
                    pad_token_id=pad_token_id,
                    lambda_recon_override=lambda_recon_val,
                )
                # Validation switches model to eval(); switch back for training loop.
                model.train()
                print(
                    f"[val] step={global_step} loss={val['loss']:.6f} "
                    f"recon={val['L_recon']:.6f} "
                    f"soft(u/b)={val['L_soft_unbanked']:.6f}/{val['L_soft_banked']:.6f} "
                    f"rkd(u/b)={val['L_rkd_unbanked']:.6f}/{val['L_rkd_banked']:.6f} "
                    f"distill(u/b)={val['L_distill_unbanked']:.6f}/{val['L_distill_banked']:.6f} "
                    f"smooth={val['L_smooth']:.6f} align_cos={val['alignment_cos']:.4f} "
                    f"align@thr={val['alignment_frac_above_threshold']:.4f} "
                    f"lambda_recon={lambda_recon_val:.4f} val_time={val.get('duration_sec', 0.0):.1f}s"
                )
                writer.add_scalar("val/loss", val["loss"], global_step)
                writer.add_scalar("val/L_recon", val["L_recon"], global_step)
                writer.add_scalar("val/L_soft", val["L_soft"], global_step)
                writer.add_scalar("val/L_rkd", val["L_rkd"], global_step)
                writer.add_scalar("val/L_distill", val["L_distill"], global_step)
                writer.add_scalar("val/L_soft_unbanked", val["L_soft_unbanked"], global_step)
                writer.add_scalar("val/L_rkd_unbanked", val["L_rkd_unbanked"], global_step)
                writer.add_scalar("val/L_distill_unbanked", val["L_distill_unbanked"], global_step)
                writer.add_scalar("val/L_soft_banked", val["L_soft_banked"], global_step)
                writer.add_scalar("val/L_rkd_banked", val["L_rkd_banked"], global_step)
                writer.add_scalar("val/L_distill_banked", val["L_distill_banked"], global_step)
                writer.add_scalar("val/L_smooth", val["L_smooth"], global_step)
                writer.add_scalar("val/alignment_cos", val["alignment_cos"], global_step)
                writer.add_scalar(
                    "val/alignment_frac_above_threshold",
                    val["alignment_frac_above_threshold"],
                    global_step,
                )
                writer.add_scalar("val/lambda_recon", lambda_recon_val, global_step)
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "cfg": cfg,
                }
                if optim_use_amp:
                    ckpt["scaler"] = scaler.state_dict()
                torch.save(ckpt, ckpt_dir / f"step_{global_step:08d}.pt")
                torch.save(ckpt, ckpt_dir / "latest.pt")
                print(f"[checkpoint] saved step_{global_step:08d}.pt and latest.pt")
                if val["loss"] < best_val:
                    best_val = val["loss"]
                    ckpt["best_val_loss"] = best_val
                    torch.save(ckpt, ckpt_dir / "best.pt")
                    writer.add_scalar("checkpoint/best_val_loss", best_val, global_step)
                    print(f"[checkpoint] new best checkpoint at step={global_step} best_val={best_val:.6f}")

            if global_step >= max_steps:
                done = True
                break

        if accum_count > 0:
            print(f"[warn] discarding partial gradient accumulation at epoch end (accum_count={accum_count})")
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            bank_student.clear()
            bank_teacher.clear()
        epoch += 1

    writer.close()
    total_min = (time.perf_counter() - train_start) / 60.0
    print(f"Training done. max_steps={max_steps} best_val={best_val:.6f} total_time={total_min:.1f}m")


if __name__ == "__main__":
    main()

