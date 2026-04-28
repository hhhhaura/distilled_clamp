from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import mido
import torch
from transformers import BertConfig
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distilled_clamp.utils import ensure_dir, load_yaml, save_jsonl, set_seed

try:
    from symusic import Score  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Score = None


def _text_cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _collect_midis(root: Path, glob_a: str, glob_b: str, max_files: int) -> list[Path]:
    items = sorted(list(root.glob(glob_a)) + list(root.glob(glob_b)))
    if max_files > 0:
        return items[:max_files]
    return items


def _symusic_load(path: Path) -> None:
    # Fast parse/validation path; events are still produced by anticipation.
    if Score is None:
        return
    _ = Score(str(path))


class ClampTeacher:
    def __init__(self, cfg: dict):
        clamp3_code = Path(cfg["paths"]["clamp3_root"]) / "code"
        if str(clamp3_code) not in sys.path:
            sys.path.insert(0, str(clamp3_code))
        import config as clamp_config  # type: ignore
        import utils as clamp_utils  # type: ignore

        self.cfg = clamp_config
        self.utils = clamp_utils
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audio_config = BertConfig(
            vocab_size=1,
            hidden_size=clamp_config.AUDIO_HIDDEN_SIZE,
            num_hidden_layers=clamp_config.AUDIO_NUM_LAYERS,
            num_attention_heads=clamp_config.AUDIO_HIDDEN_SIZE // 64,
            intermediate_size=clamp_config.AUDIO_HIDDEN_SIZE * 4,
            max_position_embeddings=clamp_config.MAX_AUDIO_LENGTH,
        )
        symbolic_config = BertConfig(
            vocab_size=1,
            hidden_size=clamp_config.M3_HIDDEN_SIZE,
            num_hidden_layers=clamp_config.PATCH_NUM_LAYERS,
            num_attention_heads=clamp_config.M3_HIDDEN_SIZE // 64,
            intermediate_size=clamp_config.M3_HIDDEN_SIZE * 4,
            max_position_embeddings=clamp_config.PATCH_LENGTH,
        )
        self.model = clamp_utils.CLaMP3Model(
            audio_config=audio_config,
            symbolic_config=symbolic_config,
            text_model_name=clamp_config.TEXT_MODEL_NAME,
            hidden_size=clamp_config.CLAMP3_HIDDEN_SIZE,
            load_m3=clamp_config.CLAMP3_LOAD_M3,
        ).to(self.device)
        self.model.eval()
        self.patchilizer = clamp_utils.M3Patchilizer()

        ckpt_path = cfg["teacher"]["clamp_weights_path"]
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(checkpoint["model"], strict=False)

    @torch.no_grad()
    def encode_global_from_events(self, events: list[int]) -> torch.Tensor:
        from anticipation.convert import events_to_midi  # type: ignore

        midi_obj = events_to_midi(events)
        mtf_text = _midi_to_mtf_text(midi_obj)
        patches = self.patchilizer.encode(mtf_text, add_special_patches=True)
        input_data = torch.tensor(patches, dtype=torch.long)

        patch_length = int(self.cfg.PATCH_LENGTH)
        segments = []
        for i in range(0, len(input_data), patch_length):
            segments.append(input_data[i : i + patch_length])
        segments[-1] = input_data[-patch_length:]

        feats = []
        weights = []
        for seg in segments:
            mask = torch.ones(seg.size(0), dtype=torch.float32)
            pad_count = patch_length - seg.size(0)
            if pad_count > 0:
                pad = torch.ones((pad_count, self.cfg.PATCH_SIZE), dtype=torch.long) * self.patchilizer.pad_token_id
                seg = torch.cat([seg, pad], dim=0)
                mask = torch.cat([mask, torch.zeros(pad_count)], dim=0)
            out = self.model.get_symbolic_features(
                symbolic_inputs=seg.unsqueeze(0).to(self.device),
                symbolic_masks=mask.unsqueeze(0).to(self.device),
                get_global=True,
            )
            feats.append(out.squeeze(0).detach().cpu())
            weights.append(float(mask.sum().item()))
        feat = torch.stack(feats, dim=0)
        w = torch.tensor(weights, dtype=feat.dtype).unsqueeze(-1)
        pooled = (feat * w).sum(dim=0) / w.sum()
        return pooled


def _sample_random_triplet_chunk(
    events: list[int],
    rng: random.Random,
    preproc_chunk_min: int,
    preproc_chunk_max: int,
) -> tuple[list[int], int] | None:
    """Random contiguous window: start on a triplet boundary, length a multiple of ``EVENT_SIZE`` (3).

    Anticipation events are (time, dur, note) triplets in a flat list; we never split mid-event.
    """
    from anticipation.config import EVENT_SIZE  # type: ignore

    n = len(events)
    if n < EVENT_SIZE:
        return None
    n = (n // EVENT_SIZE) * EVENT_SIZE
    events = events[:n]

    if preproc_chunk_max <= 0:
        target_len = n
        target_len = (target_len // EVENT_SIZE) * EVENT_SIZE
        if target_len < EVENT_SIZE:
            return None
        max_start = n - target_len
        start = rng.randrange(0, max_start // EVENT_SIZE + 1) * EVENT_SIZE
        return events[start : start + target_len], start

    lo = min(preproc_chunk_min, preproc_chunk_max)
    hi = max(preproc_chunk_min, preproc_chunk_max)
    hi_adj = min(hi, n)
    lo_adj = min(lo, hi_adj)
    if lo_adj < EVENT_SIZE:
        lo_adj = EVENT_SIZE
    lo_t = ((lo_adj + EVENT_SIZE - 1) // EVENT_SIZE) * EVENT_SIZE
    hi_t = (hi_adj // EVENT_SIZE) * EVENT_SIZE
    if lo_t > hi_t:
        return None
    lo_units = lo_t // EVENT_SIZE
    hi_units = hi_t // EVENT_SIZE
    target_len = rng.randint(lo_units, hi_units) * EVENT_SIZE
    if target_len > n:
        return None
    max_start = n - target_len
    if max_start < 0:
        return None
    start = rng.randrange(0, max_start // EVENT_SIZE + 1) * EVENT_SIZE
    return events[start : start + target_len], start


def _try_sample_random_triplet_chunk(
    events: list[int],
    rng: random.Random,
    preproc_chunk_min: int,
    preproc_chunk_max: int,
    max_attempts: int = 64,
) -> tuple[list[int], int] | None:
    for _ in range(max_attempts):
        out = _sample_random_triplet_chunk(events, rng, preproc_chunk_min, preproc_chunk_max)
        if out is not None:
            return out
    return None


def _reoffset_and_clamp_ar_chunk(events: list[int]) -> list[int]:
    """Make onset times relative to the first AR event in the chunk and clamp to valid bands.

    ``midi_to_compound`` accumulates absolute wall-clock ticks; ``compound_to_events`` maps
    them with ``TIME_OFFSET + tok`` without capping ``tok`` to ``MAX_TIME``. Long MIDIs then
    produce time tokens past the intended time range (and into duration / note ids), which
    breaks AR-only distillation (``vocab_size == CONTROL_OFFSET``).

    This mirrors the spirit of ``dab/ttm.py`` ``_filter_event_tokens_time_within_max`` (time
    must lie in ``[TIME_OFFSET, TIME_OFFSET + MAX_TIME)``), plus a rolling-window style
    re-offset so the first retained AR event has relative time 0.

    Triplets with ``note == SEPARATOR`` or ``note >= CONTROL_OFFSET`` are dropped so the
    student sees only ``[0, CONTROL_OFFSET)``.
    """
    from anticipation.config import EVENT_SIZE, MAX_DUR, MAX_TIME  # type: ignore
    from anticipation.vocab import CONTROL_OFFSET, DUR_OFFSET, SEPARATOR, TIME_OFFSET  # type: ignore

    n = (len(events) // EVENT_SIZE) * EVENT_SIZE
    if n < EVENT_SIZE:
        return []
    events = events[:n]

    base_rel: int | None = None
    for i in range(0, n, EVENT_SIZE):
        t, _, note = events[i], events[i + 1], events[i + 2]
        if note == SEPARATOR or int(note) >= int(CONTROL_OFFSET):
            continue
        base_rel = int(t) - int(TIME_OFFSET)
        break

    if base_rel is None:
        return []

    out: list[int] = []
    for i in range(0, n, EVENT_SIZE):
        t, d, note = events[i], events[i + 1], events[i + 2]
        if note == SEPARATOR or int(note) >= int(CONTROL_OFFSET):
            continue
        rel_t = int(t) - int(TIME_OFFSET)
        new_rel_t = rel_t - base_rel
        new_rel_t = max(0, min(new_rel_t, int(MAX_TIME) - 1))
        new_t = int(TIME_OFFSET) + new_rel_t

        rel_d = int(d) - int(DUR_OFFSET)
        rel_d = max(0, min(rel_d, int(MAX_DUR) - 1))
        new_d = int(DUR_OFFSET) + rel_d

        out.extend([new_t, new_d, int(note)])

    return out


def _events_to_token_ids(events: list[int], vocab_size: int) -> torch.Tensor:
    if len(events) == 0:
        return torch.tensor([], dtype=torch.long)
    ids = torch.tensor(events, dtype=torch.long)
    if int(ids.max().item()) >= vocab_size:
        raise ValueError(f"Token id out of range for vocab_size={vocab_size}. Max token={int(ids.max().item())}")
    return ids


def _midi_to_mtf_text(midifile: mido.MidiFile) -> str:
    lines = [f"ticks_per_beat {midifile.ticks_per_beat}"]
    # For MIDI objects reconstructed from event tokens, `merged_track` can miss
    # note messages. Explicitly merge tracks to preserve musical content.
    for msg in mido.merge_tracks(midifile.tracks):
        if msg.is_meta and msg.type in {
            "text",
            "copyright",
            "track_name",
            "instrument_name",
            "lyrics",
            "marker",
            "cue_marker",
            "device_name",
        }:
            continue
        lines.append(" ".join(str(v) for v in msg.dict().values()))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess for distilled_clamp.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_root", type=str, default="")
    parser.add_argument("--val_root", type=str, default="")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    from anticipation.convert import midi_to_events  # type: ignore
    from anticipation.vocab import CONTROL_OFFSET, VOCAB_SIZE  # type: ignore

    ar_event_vocab_size = int(CONTROL_OFFSET)
    full_vocab_size = int(VOCAB_SIZE)
    cfg_vocab_size = int(cfg["source"]["vocab_size"])
    if cfg_vocab_size != ar_event_vocab_size:
        print(
            "[warn] source.vocab_size does not match AR event vocab; "
            f"cfg={cfg_vocab_size} ar_event_vocab={ar_event_vocab_size}. "
            "Using AR event vocab for preprocessing."
        )
    print(
        f"[vocab] full_vocab={full_vocab_size} ar_event_vocab={ar_event_vocab_size} "
        "(tokens valid after AUTOREGRESS)"
    )

    src_cfg = cfg["source"]
    preproc_chunk_min = int(src_cfg.get("preproc_chunk_min", 0))
    preproc_chunk_max = int(src_cfg.get("preproc_chunk_max", 0))
    if preproc_chunk_max > 0 and preproc_chunk_min > preproc_chunk_max:
        raise ValueError("source.preproc_chunk_min must be <= source.preproc_chunk_max")

    midi_root = Path(cfg["paths"]["lakh_midi_root"])
    midi_files = _collect_midis(
        midi_root,
        cfg["data"]["midi_glob"],
        cfg["data"]["midi_glob_alt"],
        int(cfg["data"]["max_files"]),
    )
    if not midi_files:
        raise RuntimeError(f"No MIDI files found in {midi_root}")

    rng = random.Random(int(cfg["seed"]))
    midi_files = list(midi_files)
    rng.shuffle(midi_files)
    val_file_count = int(cfg["data"]["val_file_count"])
    val_files = midi_files[:val_file_count]
    train_files = midi_files[val_file_count:]
    val_set = {str(p.resolve()) for p in val_files}

    train_root = Path(args.train_root) if args.train_root else ensure_dir(Path(cfg["paths"]["preprocessed_train_root"]))
    val_root = Path(args.val_root) if args.val_root else ensure_dir(Path(cfg["paths"]["preprocessed_val_root"]))
    ensure_dir(train_root)
    ensure_dir(val_root)

    teacher = ClampTeacher(cfg)
    rows_train: list[dict] = []
    rows_val: list[dict] = []
    seen_keys = set()
    ok = 0
    fail = 0
    # log number of midi files
    segments_per_file = int(cfg["data"].get("segments_per_file", 10))
    print(
        f"[preprocess] midi_files={len(midi_files)} segments_per_file={segments_per_file} "
        "(random triplet-aligned windows; see source.preproc_chunk_min/max)"
    )
    for midi_path in tqdm(midi_files, desc="preprocess"):
        try:
            _symusic_load(Path(midi_path))
            events = midi_to_events(str(midi_path))
            for seg_idx in range(segments_per_file):
                sampled = _try_sample_random_triplet_chunk(
                    events,
                    rng,
                    preproc_chunk_min,
                    preproc_chunk_max,
                )
                if sampled is None:
                    continue
                chunk_events, token_start_offset = sampled
                chunk_events = _reoffset_and_clamp_ar_chunk(chunk_events)
                if not chunk_events:
                    continue
                seg_key = _text_cache_key(
                    f"{Path(midi_path).resolve()}::seg{seg_idx}::off{token_start_offset}::len{len(chunk_events)}"
                )
                if seg_key in seen_keys:
                    continue
                seen_keys.add(seg_key)
                token_ids = _events_to_token_ids(chunk_events, vocab_size=ar_event_vocab_size)
                tgt = teacher.encode_global_from_events(chunk_events)
                dst_root = val_root if str(midi_path.resolve()) in val_set else train_root
                sample = {
                    "token_ids": token_ids.to(torch.long),
                    "teacher_global": tgt.to(torch.float32),
                    "midi_path": str(Path(midi_path).resolve()),
                    "token_start_offset": int(token_start_offset),
                }
                torch.save(sample, dst_root / f"{seg_key}.pt")
                record = {
                    "key": seg_key,
                    "midi_path": str(Path(midi_path).resolve()),
                    "split": "val" if dst_root == val_root else "train",
                    "segment_index": seg_idx,
                    "token_start_offset": int(token_start_offset),
                    "token_len": int(token_ids.size(0)),
                }
                if dst_root == val_root:
                    rows_val.append(record)
                else:
                    rows_train.append(record)
                ok += 1
        except Exception as e:
            print(f"[WARN] failed midi={midi_path} err={e}")
            fail += 1
    save_jsonl(train_root / "manifest.jsonl", rows_train)
    save_jsonl(val_root / "manifest.jsonl", rows_val)
    (val_root / "selected_val_midis.txt").write_text(
        "\n".join(str(Path(p).resolve()) for p in val_files) + "\n",
        encoding="utf-8",
    )

    train_keys = {r["key"] for r in rows_train}
    val_keys = {r["key"] for r in rows_val}
    overlap = train_keys & val_keys
    if overlap:
        raise RuntimeError(f"Found overlap between train/val keys: {len(overlap)}")
    if len(val_files) != val_file_count:
        raise RuntimeError(f"Expected {val_file_count} val files, got {len(val_files)}")

    print(
        "done: "
        f"ok_segments={ok} fail_midis={fail} "
        f"train_segments={len(rows_train)} val_segments={len(rows_val)} "
        f"train_root={train_root} val_root={val_root}"
    )


if __name__ == "__main__":
    main()

