from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


def list_preprocessed_files(
    preprocessed_root: str | Path,
    *,
    max_files: int | None = None,
) -> list[Path]:
    """List preprocessed sample files (single-file ``*.pt`` samples).

    Uses :func:`os.scandir` recursively (no lexicographic sort). Order follows
    the filesystem; with ``max_files``, scanning stops once enough files are found.
    """
    root = Path(preprocessed_root)
    files: list[Path] = []
    cap = int(max_files) if max_files is not None else 0
    stack = [root]
    while stack:
        cur = stack.pop()
        with os.scandir(cur) as it:
            for ent in it:
                if cap > 0 and len(files) >= cap:
                    break
                if ent.is_dir(follow_symlinks=False):
                    stack.append(Path(ent.path))
                    continue
                if not ent.is_file() or not ent.name.endswith(".pt"):
                    continue
                files.append(Path(ent.path))
        if cap > 0 and len(files) >= cap:
            break
    return files
class PreprocessedDistillDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        *,
        max_tokens: int = 0,
        min_train_tokens: int = 0,
        random_crop: bool = False,
    ):
        self.files = files
        self.max_tokens = int(max_tokens)
        self.min_train_tokens = int(min_train_tokens)
        self.random_crop = bool(random_crop)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        src_path = self.files[idx]
        src_obj = torch.load(src_path, map_location="cpu")
        if not isinstance(src_obj, dict):
            raise TypeError(
                f"Expected dict preprocessed sample in {src_path}, got {type(src_obj).__name__}"
            )
        if "token_ids" not in src_obj or "teacher_global" not in src_obj:
            raise KeyError(
                f"Missing required keys in {src_path}; expected 'token_ids' and 'teacher_global'"
            )
        token_ids = torch.as_tensor(src_obj["token_ids"], dtype=torch.long)
        target = torch.as_tensor(src_obj["teacher_global"], dtype=torch.float32)
        midi_path = src_obj.get("midi_path")
        token_start_offset = src_obj.get("token_start_offset")
        if self.max_tokens > 0 and token_ids.numel() > self.max_tokens:
            token_ids = token_ids[: self.max_tokens]
        if self.random_crop and token_ids.numel() > 1:
            low = max(2, self.min_train_tokens) if self.min_train_tokens > 0 else 2
            high = token_ids.numel()
            if low <= high:
                crop_len = random.randint(low, high)
                token_ids = token_ids[:crop_len]
        return {
            "token_ids": token_ids,
            "decoder_target_sequence": token_ids.clone(),
            "teacher_global_target": target,
            "key": src_path.stem,
            "midi_path": midi_path,
            "token_start_offset": token_start_offset,
        }


def collate_preprocessed_batch(batch: list[dict], *, pad_token_id: int) -> dict:
    seqs = [x["token_ids"] for x in batch]
    dec_tgts = [x["decoder_target_sequence"] for x in batch]
    tgts = [x["teacher_global_target"] for x in batch]
    keys = [x["key"] for x in batch]
    midi_paths = [x.get("midi_path") for x in batch]
    token_start_offsets = [x.get("token_start_offset") for x in batch]

    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())
    token_ids = torch.full((len(seqs), max_len), int(pad_token_id), dtype=torch.long)
    dec = torch.full((len(seqs), max_len), int(pad_token_id), dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        l = s.size(0)
        token_ids[i, :l] = s
        dec[i, :l] = dec_tgts[i]
        mask[i, :l] = True

    teacher_global = torch.stack(tgts, dim=0)
    return {
        "token_ids": token_ids,
        "attention_mask": mask,
        "decoder_target_sequence": dec,
        "teacher_global_target": teacher_global,
        "token_lengths": lengths,
        "keys": keys,
        "midi_paths": midi_paths,
        "token_start_offsets": token_start_offsets,
    }

