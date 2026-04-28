from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RopeTransformerEncoder, RopeTransformerEncoderLayer


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if int(rank) <= 0:
            raise ValueError(f"LoRALinear rank must be > 0, got {rank}")
        self.base = base_linear
        self.rank = rank
        self.scaling = alpha / float(rank)
        in_features = self.base.in_features
        out_features = self.base.out_features
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for p in self.base.parameters():
            p.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        delta = self.lora_b.weight @ self.lora_a.weight
        return self.base.weight + delta * self.scaling

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base.weight, self.base.bias)
        x_in = self.lora_dropout(x)
        delta = self.lora_b(self.lora_a(x_in)) * self.scaling
        return base_out + delta


class StudentInputModule(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, patch_len: int, nhead: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_len = patch_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.patch_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
        )

    def forward(
        self,
        token_inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        inputs_are_one_hot: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.embedding.weight
        if inputs_are_one_hot:
            x = token_inputs.to(dtype=w.dtype)
            tok_emb = x @ w
        else:
            tok_emb = self.embedding(token_inputs.long())
        bsz, seqlen, d_model = tok_emb.shape
        pad_len = (self.patch_len - (seqlen % self.patch_len)) % self.patch_len
        if pad_len > 0:
            tok_emb = torch.cat(
                [tok_emb, torch.zeros(bsz, pad_len, d_model, device=tok_emb.device, dtype=tok_emb.dtype)],
                dim=1,
            )
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(bsz, pad_len, device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1,
            )
        num_patches = tok_emb.size(1) // self.patch_len
        patch_tokens = tok_emb.view(bsz * num_patches, self.patch_len, d_model)
        patch_mask = attention_mask.view(bsz * num_patches, self.patch_len)
        patch_valid_flat = patch_mask.any(dim=1)
        safe_patch_mask = patch_mask.clone()
        safe_patch_mask[~patch_valid_flat, 0] = True
        q = self.patch_query.expand(bsz * num_patches, -1, -1)
        patch_latent, _ = self.cross_attn(
            query=q,
            key=patch_tokens,
            value=patch_tokens,
            key_padding_mask=~safe_patch_mask.bool(),
        )
        patch_latent = patch_latent.squeeze(1).view(bsz, num_patches, d_model)
        patch_valid = patch_valid_flat.view(bsz, num_patches)
        patch_latent = patch_latent * patch_valid.unsqueeze(-1).to(patch_latent.dtype)
        return patch_latent, patch_valid

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Legacy: flat Parameter "embedding" -> nn.Embedding "embedding.weight"
        legacy = prefix + "embedding"
        weight_key = prefix + "embedding.weight"
        if legacy in state_dict and weight_key not in state_dict:
            state_dict[weight_key] = state_dict.pop(legacy)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class PerceiverInputModule(nn.Module):
    """Per-patch latent bank input module used by perceiver checkpoints.

    Each patch uses ``num_latents`` learned queries and the resulting latent sequence is flattened
    from ``(B, P, L, D)`` to ``(B, P*L, D)`` for downstream M3/decoder compatibility.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        patch_len: int,
        nhead: int = 8,
        num_latents: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_len = patch_len
        self.num_latents = int(num_latents)
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.latent_queries = nn.Parameter(torch.randn(1, self.num_latents, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
        )

    def forward(
        self,
        token_inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        inputs_are_one_hot: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.embedding.weight
        if inputs_are_one_hot:
            x = token_inputs.to(dtype=w.dtype)
            tok_emb = x @ w
        else:
            tok_emb = self.embedding(token_inputs.long())
        bsz, seqlen, d_model = tok_emb.shape
        pad_len = (self.patch_len - (seqlen % self.patch_len)) % self.patch_len
        if pad_len > 0:
            tok_emb = torch.cat(
                [tok_emb, torch.zeros(bsz, pad_len, d_model, device=tok_emb.device, dtype=tok_emb.dtype)],
                dim=1,
            )
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(bsz, pad_len, device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1,
            )
        num_patches = tok_emb.size(1) // self.patch_len
        patch_tokens = tok_emb.view(bsz * num_patches, self.patch_len, d_model)
        patch_mask = attention_mask.view(bsz * num_patches, self.patch_len)
        patch_valid_flat = patch_mask.any(dim=1)
        safe_patch_mask = patch_mask.clone()
        safe_patch_mask[~patch_valid_flat, 0] = True
        q = self.latent_queries.expand(bsz * num_patches, -1, -1)
        patch_latents, _ = self.cross_attn(
            query=q,
            key=patch_tokens,
            value=patch_tokens,
            key_padding_mask=~safe_patch_mask.bool(),
        )
        patch_latents = patch_latents.reshape(bsz, num_patches * self.num_latents, d_model)
        patch_valid = patch_valid_flat.view(bsz, num_patches)
        valid_expanded = patch_valid.unsqueeze(-1).expand(-1, -1, self.num_latents).reshape(bsz, num_patches * self.num_latents)
        patch_latents = patch_latents * valid_expanded.unsqueeze(-1).to(patch_latents.dtype)
        return patch_latents, valid_expanded

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        legacy = prefix + "embedding"
        weight_key = prefix + "embedding.weight"
        if legacy in state_dict and weight_key not in state_dict:
            state_dict[weight_key] = state_dict.pop(legacy)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class DistillationHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        m = memory_mask.unsqueeze(-1).to(memory.dtype)
        pooled = (memory * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
        z = self.proj(pooled)
        return F.normalize(z, dim=-1)


class DistillationMLPHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hdim = int(hidden_dim) if hidden_dim is not None else int(d_model) * 2
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hdim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hdim, out_dim)

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        m = memory_mask.unsqueeze(-1).to(memory.dtype)
        pooled = (memory * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
        x = self.norm(pooled)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        z = self.fc2(x)
        return F.normalize(z, dim=-1)


def _truncate_patches(
    patch_latents: torch.Tensor,
    patch_mask: torch.Tensor,
    max_patches: int,
    *,
    slots_per_patch: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_patches <= 0:
        return patch_latents, patch_mask
    max_slots = int(max_patches) * int(slots_per_patch)
    if patch_latents.size(1) <= max_slots:
        return patch_latents, patch_mask
    return patch_latents[:, :max_slots, :], patch_mask[:, :max_slots]


def _normalize_input_module_type(raw: str) -> str:
    kind = str(raw).strip().lower()
    if kind in {"patch", "student", "student_patch"}:
        return "patch"
    if kind in {"perceiver", "perceiver_resampling", "resampler"}:
        return "perceiver"
    return kind


class DistilledAntiClamp2Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 4,
        encoder_num_layers: int | None = None,
        decoder_num_layers: int | None = None,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        patch_len: int = 63,
        freeze_enable: bool = True,
        freeze_lora: str = "core_only",
        freeze_rank: int = 8,
        freeze_alpha: float = 16.0,
        freeze_dropout: float = 0.0,
        out_dim: int = 768,
        max_patches: int = 0,
        rope_theta: float = 10000.0,
        rope_max_positions: int = 4096,
        input_module_type: str = "patch",
        perceiver_num_latents: int = 64,
        m3_alignment: str = "distilled_head",
        align_mlp_hidden_dim: int = 0,
        align_mlp_dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_patches = int(max_patches)
        self.freeze_enable = bool(freeze_enable)
        self.freeze_lora = str(freeze_lora).strip().lower()
        self.input_module_type = _normalize_input_module_type(input_module_type)
        self.m3_alignment = m3_alignment
        if encoder_num_layers is None:
            encoder_num_layers = int(num_layers)
        if decoder_num_layers is None:
            decoder_num_layers = int(num_layers)
        self.encoder_num_layers = int(encoder_num_layers)
        self.decoder_num_layers = int(decoder_num_layers)
        if self.encoder_num_layers <= 0 or self.decoder_num_layers <= 0:
            raise ValueError(
                f"encoder_num_layers and decoder_num_layers must be > 0, got "
                f"encoder_num_layers={self.encoder_num_layers} decoder_num_layers={self.decoder_num_layers}"
            )
        if self.m3_alignment not in {"direct", "distilled_head", "mlp"}:
            raise ValueError(
                f"Unknown m3_alignment={m3_alignment!r} (normalized={self.m3_alignment!r}); "
                "expected one of: direct | distilled_head | mlp"
            )
        if self.m3_alignment == "direct" and int(d_model) != int(out_dim):
            raise ValueError(
                f"m3_alignment='direct' requires d_model == out_dim, got d_model={d_model} out_dim={out_dim}"
            )
        self.perceiver_num_latents = int(perceiver_num_latents)
        self._input_slots_per_patch = self.perceiver_num_latents if self.input_module_type == "perceiver" else 1
        if self.input_module_type == "perceiver":
            self.input_module = PerceiverInputModule(
                vocab_size=vocab_size,
                d_model=d_model,
                patch_len=patch_len,
                nhead=nhead,
                num_latents=self.perceiver_num_latents,
            )
        elif self.input_module_type == "patch":
            self.input_module = StudentInputModule(
                vocab_size=vocab_size,
                d_model=d_model,
                patch_len=patch_len,
                nhead=nhead,
            )
        else:
            raise ValueError(
                f"Unknown input_module_type={input_module_type!r} (normalized={self.input_module_type!r}); "
                "expected one of patch/perceiver aliases."
            )
        enc_layers = [
            RopeTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                rope_theta=rope_theta,
                max_positions=rope_max_positions,
            )
            for _ in range(self.encoder_num_layers)
        ]
        self.m3_core = RopeTransformerEncoder(enc_layers)
        self._configure_m3_trainability(
            freeze_enable=self.freeze_enable,
            freeze_lora=self.freeze_lora,
            rank=int(freeze_rank),
            alpha=float(freeze_alpha),
            dropout=float(freeze_dropout),
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.decoder_num_layers)
        self.decoder_tok_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_out = nn.Linear(d_model, vocab_size)
        if self.m3_alignment == "mlp":
            hidden_dim = int(align_mlp_hidden_dim) if int(align_mlp_hidden_dim) > 0 else None
            self.distill_head = DistillationMLPHead(
                d_model=d_model,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                dropout=float(align_mlp_dropout),
            )
        else:
            self.distill_head = DistillationHead(d_model=d_model, out_dim=out_dim)

    def _pool_memory(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        m = memory_mask.unsqueeze(-1).to(memory.dtype)
        return (memory * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)

    def _decode_from_memory(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        decoder_target_sequence: torch.Tensor,
    ) -> torch.Tensor:
        mem_mask = ~memory_mask.bool()  # True where padding (nn.Transformer convention)
        shifted = self._shift_right(decoder_target_sequence)
        dec_emb = self.decoder_tok_embedding(shifted)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(shifted.size(1), device=shifted.device)
        dec_out = self.decoder(
            tgt=dec_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=mem_mask,
        )
        return self.decoder_out(dec_out)

    def _memory_with_noise(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        *,
        smooth_noise_std: float,
    ) -> torch.Tensor:
        if float(smooth_noise_std) <= 0.0:
            return memory
        noise = torch.randn_like(memory) * float(smooth_noise_std)
        valid = memory_mask.unsqueeze(-1).to(dtype=memory.dtype)
        return memory + noise * valid

    def _set_m3_requires_grad(self, enabled: bool) -> None:
        for p in self.m3_core.parameters():
            p.requires_grad = enabled

    def _inject_lora_into_m3(
        self,
        rank: int,
        alpha: float,
        dropout: float,
        *,
        lora_qkv: bool,
    ) -> None:
        for layer in self.m3_core.layers:
            self_attn = layer.self_attn
            if lora_qkv:
                self_attn.q_proj = LoRALinear(self_attn.q_proj, rank=rank, alpha=alpha, dropout=dropout)
                self_attn.k_proj = LoRALinear(self_attn.k_proj, rank=rank, alpha=alpha, dropout=dropout)
                self_attn.v_proj = LoRALinear(self_attn.v_proj, rank=rank, alpha=alpha, dropout=dropout)
            self_attn.out_proj = LoRALinear(self_attn.out_proj, rank=rank, alpha=alpha, dropout=dropout)
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                mod = getattr(self_attn, name)
                if isinstance(mod, LoRALinear):
                    for p in mod.lora_a.parameters():
                        p.requires_grad = True
                    for p in mod.lora_b.parameters():
                        p.requires_grad = True

    def _configure_m3_trainability(
        self,
        *,
        freeze_enable: bool,
        freeze_lora: str,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        if not freeze_enable:
            self._set_m3_requires_grad(True)
            return

        mode = str(freeze_lora).strip().lower()
        if mode not in {"false", "core_only", "qkv"}:
            raise ValueError(f"Unsupported freeze_lora mode: {freeze_lora!r}")

        self._set_m3_requires_grad(False)
        if mode == "false":
            return
        self._inject_lora_into_m3(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            lora_qkv=(mode == "qkv"),
        )

    def _shift_right(self, tgt: torch.Tensor, bos_token_id: int = 0) -> torch.Tensor:
        out = torch.full_like(tgt, int(bos_token_id))
        out[:, 1:] = tgt[:, :-1]
        return out

    def forward(
        self,
        token_inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_target_sequence: torch.Tensor,
        inputs_are_one_hot: bool = False,
        smooth_noise_std: float = 0.0,
    ) -> dict:
        patch_latents, patch_mask = self.input_module(
            token_inputs=token_inputs,
            attention_mask=attention_mask,
            inputs_are_one_hot=inputs_are_one_hot,
        )
        patch_latents, patch_mask = _truncate_patches(
            patch_latents,
            patch_mask,
            self.max_patches,
            slots_per_patch=self._input_slots_per_patch,
        )
        mem_mask = ~patch_mask.bool()  # True where padding (nn.Transformer convention)
        memory = self.m3_core(patch_latents, src_key_padding_mask=mem_mask)
        logits = self._decode_from_memory(memory, patch_mask, decoder_target_sequence)
        if self.m3_alignment == "direct":
            student_global = F.normalize(self._pool_memory(memory, patch_mask), dim=-1)
        else:
            student_global = self.distill_head(memory, patch_mask)
        logits_noisy = None
        if float(smooth_noise_std) > 0.0:
            noisy_memory = self._memory_with_noise(
                memory,
                patch_mask,
                smooth_noise_std=float(smooth_noise_std),
            )
            logits_noisy = self._decode_from_memory(noisy_memory, patch_mask, decoder_target_sequence)
        return {
            "logits": logits,
            "logits_noisy": logits_noisy,
            "student_global": student_global,
            "memory": memory,
            "memory_mask": patch_mask,
        }
