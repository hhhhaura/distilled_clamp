# distilled_clamp (ctrlm)

Distilled Anti-CLaMP2 student training code. Python package name: **`distilled_clamp`** (folder `distilled_clamp/` next to `scripts/` and `configs/`).

Downstream (e.g. TTM) should set **`distilled_root`** to this directory (`.../ctrlm/distilled_clamp`) so `from distilled_clamp.models.student import DistilledAntiClamp2Model` resolves.

**Paths in YAML:** set `export CTRLM_REPO=/abs/path/to/CTRL-M` and `export LAKH_MIDI_ROOT=...` before running scripts; string values in `configs/base.yaml` and `configs/smoke.yaml` use `${CTRLM_REPO}` / `${LAKH_MIDI_ROOT}` and are expanded when configs load. Optionally keep a local override YAML (gitignored) if you prefer not to use env vars.

**Conda (recommended):** use the project env that has PyTorch and anticipation installed, e.g. `conda activate anticipation`, then run `scripts/preprocess.py`, `train_from_preprocessed.py`, and `pytest` from `ctrlm/distilled_clamp`.

---

Distilled Anti-CLaMP2 pipeline with:

1. **Preprocess**: for each MIDI, draw **`data.segments_per_file`** random windows from the full event stream. Each window starts at a **triplet boundary** (anticipation `EVENT_SIZE=3`: time, duration, note), and its length is sampled in token counts between `source.preproc_chunk_min` and `source.preproc_chunk_max` (triplet-aligned). Same logic for train and val files. Teacher global targets are computed from the same cropped events.
2. **Train**: dual-branch student with differentiable fixed-token patching, RoPE M3 encoder (full fine-tune or frozen with optional LoRA), AR decoder, and distillation head.

### Input module choices

Configure encoder input compression with `input_module` in YAML:

- `input_module.type: patch` (default): `StudentInputModule` patch-wise cross-attention using `patching.patch_len`.
- `input_module.type: perceiver`: `PerceiverResamplingInputModule` with learned latent queries (`input_module.perceiver_num_latents`).

Perceiver mode keeps a fixed latent count regardless of source token length, analogous to Perceiver resampling in vision-language stacks.

**Checkpoint compatibility:** Embeddings saved as `nn.Embedding` can load legacy flat `input_module.embedding` tensors. The **RoPE M3 core** is not compatible with checkpoints from the older `nn.TransformerEncoder` stack; retrain from scratch and use an updated `distilled_cfg` in dab.

**Trainable M3 (config):** `model.freeze` controls M3 freezing and LoRA mode:

- `model.freeze.enable: false` -> full M3 fine-tuning (no LoRA injection).
- `model.freeze.enable: true` + `model.freeze.lora: false` -> freeze M3 with no LoRA adapters.
- `model.freeze.enable: true` + `model.freeze.lora: core_only` -> freeze M3, LoRA on attention `out_proj` only.
- `model.freeze.enable: true` + `model.freeze.lora: qkv` -> freeze M3, LoRA on `q_proj`/`k_proj`/`v_proj`/`out_proj`.
- LoRA hyperparameters live at `model.freeze.rank`, `model.freeze.alpha`, `model.freeze.dropout`.

**M3 alignment mode (config):** `model.m3_alignment` controls how `student_global` is formed for distillation:

- `model.m3_alignment: distilled_head` (default) -> masked mean pool + learned linear projection (`DistillationHead`) + L2 normalization.
- `model.m3_alignment: direct` -> masked mean pool of M3 memory + L2 normalization (no projection head).
- `direct` requires `model.d_model == target.dim` so the latent already lives in CLAMP3 target dimensionality.

## AR-only vocab used by the bridge

The bridge uses only the token block that can appear **after** `[AUTOREGRESS]` generation:

- valid IDs for AR generation: `[0, CONTROL_OFFSET)` (event time/duration/note/rest)
- excluded IDs: `[CONTROL_OFFSET, VOCAB_SIZE)` (anticipation control tokens + special flags)

With current `anticipation` constants this is:

- `CONTROL_OFFSET = 27513`
- `VOCAB_SIZE = 55028`

So the bridge student is trained with `source.vocab_size: 27513`.

## Phase 1: Preprocess

```bash
python scripts/preprocess.py --config configs/base.yaml
```

Outputs (default paths use a legacy `preprocessed_seg10_*` folder name):

- `preprocessed_seg10_train/token_ids`, `preprocessed_seg10_train/teacher_global`
- `preprocessed_seg10_val/token_ids`, `preprocessed_seg10_val/teacher_global`
- `manifest.jsonl` for each split (includes `token_start_offset`, `token_len`)
- `selected_val_midis.txt` in val root

Split policy:

- exactly `data.val_file_count` MIDI files in validation; the rest are train
- **`data.segments_per_file`** random crops per file (train and val), triplet-aligned token windows as above

## Phase 2: Train

```bash
python scripts/train_from_preprocessed.py --config configs/base.yaml --auto_resume
```

Behavior:

- console + TensorBoard logging every `log.log_every_steps` (default 50)
- checkpoint every `log.save_every_steps` (default 1000)
- checkpoints are saved under `paths.output_root/checkpoints/<module_design_tag>/` where tag encodes the selected input design (for example `input_patch_len63` or `input_perceiver_lat64`)
- validation at checkpoint steps; best checkpoint selected by total validation loss
- auto-resume from latest `step_*.pt` checkpoint when `--auto_resume` is provided (restores optimizer, scheduler, and AMP scaler when present)
- optional train-subset cap via `data.max_training_samples` (0 = use all preprocessed train pairs), override with `--max_training_samples`
- loss = `lambda_recon * CE + soft_distribution_distill + lambda_rkd * RKD`
- optional smoothness regularizer on memory-token perturbations:
  - set `loss.lambda_smooth > 0` and `loss.smooth_noise_std > 0`
  - choose `loss.smooth_loss_type: kl|mse`
- alignment diagnostics:
  - `alignment_cos` (mean cosine between student and matching teacher global embeddings)
  - `alignment_frac_above_threshold` (fraction above `loss.alignment_threshold`, default `0.8`)
- student input patching is done in-model with fixed `patching.patch_len` (default 63); optional `patching.max_patches > 0` truncates the patch sequence from the left
- **AMP** (`optim.use_amp`, CUDA), **gradient accumulation** (`optim.accumulation_steps`), **warmup + cosine decay** (`optim.warmup_steps`, `optim.lr_min`, `optim.cosine`)

### Contrastive / batch size

`InfoNCE` negatives come from the current forward batch (`B x B` similarity matrix). Gradient accumulation improves optimization stability but does not add extra in-batch negatives to one InfoNCE call. Not implemented here: symmetric InfoNCE, memory bank / MoCo-style encoders, or multi-GPU all-gather negatives.

### Rescanning preprocessed shards (streaming / growing datasets)

At the **start of each epoch**, you can re-list `token_ids` / `teacher_global` pairs from disk so new `.pt` files are picked up:

- `data.rescan_train_each_epoch: true` — refresh the **train** `DataLoader` each epoch.
- `data.rescan_val_each_epoch: true` — refresh **val** each epoch (same idea for val shards).
- `data.rescan_each_epoch: true` — shorthand for **both** train and val.

CLI overrides YAML: `--rescan-train-each-epoch` / `--no-rescan-train-each-epoch`, `--rescan-val-each-epoch` / `--no-rescan-val-each-epoch`.

Validation during training still **rebuilds val** at each **checkpoint** (`log.save_every_steps`) as before; `rescan_val_each_epoch` additionally refreshes val **every epoch** (useful if val shards change between epochs).

TensorBoard:

```bash
tensorboard --logdir ./outputs/tensorboard
```

## Rebuild after vocab-scope change

Because source-token semantics changed, rebuild both cached tensors and checkpoints:

```bash
cd /path/to/ctrlm/distilled_clamp
rm -rf preprocessed_seg10_train preprocessed_seg10_val outputs
python scripts/preprocess.py --config configs/base.yaml
python scripts/train_from_preprocessed.py --config configs/base.yaml --auto_resume
```

Optional smoke run (append results to `SMOKE_LOG.md`). Pin a free GPU if others are busy, e.g. device 3:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anticipation
export CTRLM_REPO=/abs/path/to/CTRL-M
export LAKH_MIDI_ROOT=/path/to/midi
export CUDA_VISIBLE_DEVICES=3
python scripts/preprocess.py --config configs/smoke.yaml
python scripts/train_from_preprocessed.py --config configs/smoke.yaml --auto_resume
python -m pytest tests/ -q
```

## Tests

```bash
cd /path/to/ctrlm/distilled_clamp
pytest tests/
```
# distilled_clamp
