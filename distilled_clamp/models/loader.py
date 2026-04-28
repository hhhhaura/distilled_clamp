"""Build :class:`DistilledAntiClamp2Model` from a distilled training YAML dict."""

from __future__ import annotations

from typing import Any, Mapping

from .student import DistilledAntiClamp2Model


def build_distilled_student_from_cfg(
    distilled_cfg: Mapping[str, Any],
    *,
    vocab_size: int,
) -> DistilledAntiClamp2Model:
    """Construct the student bridge from ``yaml.safe_load`` output (``configs/*.yaml``)."""
    mcfg = distilled_cfg["model"]
    patch_cfg = distilled_cfg.get("patching", {})
    input_cfg = distilled_cfg.get("input_module", {})
    rope_cfg = distilled_cfg.get("rope", {})
    patch_len = int(patch_cfg.get("patch_len", 63))
    max_patches = int(patch_cfg.get("max_patches", 0))
    shared_num_layers = int(mcfg.get("num_layers", mcfg.get("encoder_num_layers", mcfg.get("decoder_num_layers", 4))))
    encoder_num_layers = int(mcfg.get("encoder_num_layers", shared_num_layers))
    decoder_num_layers = int(mcfg.get("decoder_num_layers", shared_num_layers))
    input_module_type = str(input_cfg.get("type", patch_cfg.get("input_module_type", "patch")))
    perceiver_num_latents = int(input_cfg.get("perceiver_num_latents", patch_cfg.get("perceiver_num_latents", 64)))
    rope_theta = float(rope_cfg.get("theta", 10000.0))
    rope_max_positions = int(rope_cfg.get("max_positions", 4096))
    alignment_mode = str(mcfg.get("align", mcfg.get("m3_alignment", "distilled_head")))
    return DistilledAntiClamp2Model(
        vocab_size=vocab_size,
        d_model=int(mcfg["d_model"]),
        nhead=int(mcfg["nhead"]),
        num_layers=shared_num_layers,
        encoder_num_layers=encoder_num_layers,
        decoder_num_layers=decoder_num_layers,
        dim_feedforward=int(mcfg["dim_feedforward"]),
        dropout=float(mcfg["dropout"]),
        patch_len=patch_len,
        freeze_enable=mcfg["freeze"]["enable"],
        freeze_lora=mcfg["freeze"]["lora"],
        freeze_rank=mcfg["freeze"]["rank"],
        freeze_alpha=mcfg["freeze"]["alpha"],
        freeze_dropout=mcfg["freeze"]["dropout"],
        out_dim=int(distilled_cfg["target"]["dim"]),
        max_patches=max_patches,
        rope_theta=rope_theta,
        rope_max_positions=rope_max_positions,
        input_module_type=input_module_type,
        perceiver_num_latents=perceiver_num_latents,
        m3_alignment=alignment_mode,
        align_mlp_hidden_dim=int(mcfg.get("align_mlp_hidden_dim", 0)),
        align_mlp_dropout=float(mcfg.get("align_mlp_dropout", mcfg.get("dropout", 0.1))),
    )
