from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    vocab = logits.size(-1)
    per_tok = F.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        reduction="none",
    )
    mask = valid_mask.reshape(-1).to(dtype=per_tok.dtype, device=per_tok.device)
    denom = mask.sum().clamp_min(1.0)
    return (per_tok * mask).sum() / denom


def soft_distribution_distill_loss(
    student_global: torch.Tensor,
    teacher_global: torch.Tensor,
    *,
    tau_teacher: float = 0.05,
    tau_student: float = 0.07,
) -> torch.Tensor:
    s = F.normalize(student_global, dim=-1)
    t = F.normalize(teacher_global, dim=-1)
    teacher_logits = (t @ t.t()) / float(tau_teacher)
    student_logits = (s @ t.t()) / float(tau_student)
    soft_targets = F.softmax(teacher_logits, dim=-1).detach()
    log_probs = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(log_probs, soft_targets, reduction="batchmean")


def rkd_distance_loss(
    student_global: torch.Tensor,
    teacher_global: torch.Tensor,
) -> torch.Tensor:
    s = F.normalize(student_global, dim=-1)
    t = F.normalize(teacher_global, dim=-1)
    n = int(s.size(0))
    if n < 2:
        return torch.zeros((), device=s.device, dtype=s.dtype)
    mask = ~torch.eye(n, dtype=torch.bool, device=s.device)
    s_dist = torch.cdist(s, s, p=2)[mask]
    t_dist = torch.cdist(t, t, p=2)[mask]
    s_dist = s_dist / s_dist.mean().clamp_min(1.0e-6)
    t_dist = t_dist / t_dist.mean().clamp_min(1.0e-6)
    return F.mse_loss(s_dist, t_dist)


def smoothness_loss(
    logits: torch.Tensor,
    logits_noisy: torch.Tensor,
    *,
    loss_type: str = "kl",
) -> torch.Tensor:
    kind = str(loss_type).strip().lower()
    if kind == "mse":
        return F.mse_loss(logits_noisy, logits)
    if kind == "kl":
        p = F.softmax(logits.detach(), dim=-1)
        q_log = F.log_softmax(logits_noisy, dim=-1)
        return F.kl_div(q_log, p, reduction="batchmean")
    raise ValueError(f"Unsupported smoothness loss_type={loss_type!r}; use 'kl' or 'mse'")


def reconstruction_smooth_loss(
    logits: torch.Tensor,
    decoder_targets: torch.Tensor,
    decoder_valid_mask: torch.Tensor,
    *,
    logits_noisy: torch.Tensor | None = None,
    lambda_recon: float = 1.0,
    lambda_smooth: float = 0.0,
    smooth_loss_type: str = "kl",
) -> tuple[torch.Tensor, dict]:
    l_recon = reconstruction_ce_loss(logits, decoder_targets, valid_mask=decoder_valid_mask)
    l_smooth = torch.zeros((), device=logits.device, dtype=logits.dtype)
    if float(lambda_smooth) > 0.0 and logits_noisy is not None:
        l_smooth = smoothness_loss(logits, logits_noisy, loss_type=smooth_loss_type)
    total = float(lambda_recon) * l_recon + float(lambda_smooth) * l_smooth
    return total, {
        "L_recon": float(l_recon.item()),
        "L_smooth": float(l_smooth.item()),
    }


def alignment_stats(
    student_global: torch.Tensor,
    teacher_global: torch.Tensor,
    *,
    threshold: float = 0.8,
) -> tuple[float, float]:
    s = F.normalize(student_global, dim=-1)
    t = F.normalize(teacher_global, dim=-1)
    cos = (s * t).sum(dim=-1)
    mean_cos = float(cos.mean().item())
    frac_above = float((cos > float(threshold)).float().mean().item())
    return mean_cos, frac_above


def joint_loss(
    logits: torch.Tensor,
    decoder_targets: torch.Tensor,
    decoder_valid_mask: torch.Tensor,
    student_global: torch.Tensor,
    teacher_global: torch.Tensor,
    logits_noisy: torch.Tensor | None = None,
    lambda_recon: float = 1.0,
    lambda_soft: float = 1.0,
    lambda_rkd: float = 1.0,
    lambda_smooth: float = 0.0,
    smooth_loss_type: str = "kl",
    alignment_threshold: float = 0.8,
    temperature: float = 0.07,
    teacher_temperature: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    l_recon = reconstruction_ce_loss(logits, decoder_targets, valid_mask=decoder_valid_mask)
    tau_student = float(temperature)
    tau_teacher = float(teacher_temperature) if float(teacher_temperature) > 0.0 else (tau_student / 2.0)
    l_soft = soft_distribution_distill_loss(
        student_global,
        teacher_global,
        tau_teacher=tau_teacher,
        tau_student=tau_student,
    )
    l_rkd = rkd_distance_loss(student_global, teacher_global)
    l_distill = float(lambda_soft) * l_soft + float(lambda_rkd) * l_rkd
    l_smooth = torch.zeros((), device=logits.device, dtype=logits.dtype)
    if float(lambda_smooth) > 0.0 and logits_noisy is not None:
        l_smooth = smoothness_loss(logits, logits_noisy, loss_type=smooth_loss_type)
    total = (
        float(lambda_recon) * l_recon
        + l_distill
        + float(lambda_smooth) * l_smooth
    )
    align_cos, align_frac = alignment_stats(
        student_global,
        teacher_global,
        threshold=alignment_threshold,
    )
    return total, {
        "L_recon": float(l_recon.item()),
        "L_distill": float(l_distill.item()),
        "L_soft": float(l_soft.item()),
        "L_rkd": float(l_rkd.item()),
        "L_smooth": float(l_smooth.item()),
        "alignment_cos": align_cos,
        "alignment_frac_above_threshold": align_frac,
    }

