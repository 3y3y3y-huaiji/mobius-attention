"""Minimal, differentiable RoPE phase interventions for controlled research.

The module deliberately separates three different operations:
- standard: ordinary RoPE on queries and keys;
- query: an additional query-only phase rotation, applied to every past key;
- segment: independent token phases on Q and K, giving segment-level selectivity.

No semantic detector, topology claim, custom cache, or model monkey-patching lives here.
"""

import math
from typing import Literal

import torch

PhaseMode = Literal["standard", "query", "segment"]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of each 2D RoPE plane."""
    if x.shape[-1] % 2:
        raise ValueError("The head dimension must be even.")
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply a duplicated-half rotary table to [batch, heads, tokens, dim]."""
    return x * cos + rotate_half(x) * sin


def rope_tables(
    positions: torch.Tensor,
    head_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build standard RoPE tables for a one-dimensional position tensor."""
    if head_dim % 2:
        raise ValueError("The head dimension must be even.")
    if positions.ndim != 1:
        raise ValueError("positions must have shape [tokens].")
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    frequencies = base ** (-torch.arange(0, head_dim, 2, device=positions.device, dtype=work_dtype) / head_dim)
    angles = positions.to(work_dtype).unsqueeze(-1) * frequencies
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def reversal_phase(
    seq_len: int,
    reversals: list[int] | tuple[int, ...],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a causal phase staircase; each reversal adds pi at its token."""
    positions = torch.arange(seq_len, device=device)
    if not reversals:
        return torch.zeros(seq_len, device=device, dtype=dtype)
    boundaries = torch.tensor(reversals, device=device).unsqueeze(1)
    if torch.any(boundaries < 0) or torch.any(boundaries >= seq_len):
        raise ValueError("Every reversal index must be inside the sequence.")
    return (positions.unsqueeze(0) >= boundaries).sum(dim=0).to(dtype) * math.pi


def _expand_phase(phase: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Expand [T], [B,T], or [B,H,T] to [B,H,T,1]."""
    if reference.ndim != 4:
        raise ValueError("Q and K must have shape [batch, heads, tokens, dim].")
    if phase.ndim == 1:
        phase = phase.view(1, 1, -1, 1)
    elif phase.ndim == 2:
        phase = phase.unsqueeze(1).unsqueeze(-1)
    elif phase.ndim == 3:
        phase = phase.unsqueeze(-1)
    elif phase.ndim == 4 and phase.shape[-1] == 1:
        pass
    else:
        raise ValueError("phase must have shape [T], [B,T], or [B,H,T].")
    if phase.shape[-2] != reference.shape[-2]:
        raise ValueError("The phase and token dimensions must match.")
    return phase.to(device=reference.device, dtype=reference.dtype)


def apply_phase(
    x: torch.Tensor,
    phase: torch.Tensor,
    channel_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply an additional norm-preserving rotation to selected RoPE planes."""
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("The head dimension must be even.")
    if channel_mask is None:
        mask = torch.ones(dim // 2, device=x.device, dtype=x.dtype)
    else:
        if channel_mask.ndim != 1 or channel_mask.numel() not in (dim // 2, dim):
            raise ValueError("channel_mask must contain one value per plane or per channel.")
        mask = channel_mask[: dim // 2].to(device=x.device, dtype=x.dtype)
    angles = _expand_phase(phase, x) * mask.view(1, 1, 1, -1)
    angles = torch.cat((angles, angles), dim=-1)
    return apply_rotary(x, angles.cos(), angles.sin())


def attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    q_phase: torch.Tensor | None = None,
    k_phase: torch.Tensor | None = None,
    mode: PhaseMode = "standard",
    channel_mask: torch.Tensor | None = None,
    rope_base: float = 10000.0,
) -> torch.Tensor:
    """Compute scaled attention logits under a precisely defined phase intervention."""
    if q.ndim != 4 or k.ndim != 4 or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q and K must be 4D and have the same head dimension.")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1]:
        raise ValueError("Q and K batch/head dimensions must match in this minimal prototype.")

    q_cos, q_sin = rope_tables(q_positions, q.shape[-1], rope_base, q.dtype)
    k_cos, k_sin = rope_tables(k_positions, k.shape[-1], rope_base, k.dtype)
    table_shape = (1, 1, -1, q.shape[-1])
    q_rot = apply_rotary(q, q_cos.to(q.dtype).view(*table_shape), q_sin.to(q.dtype).view(*table_shape))
    k_rot = apply_rotary(k, k_cos.to(k.dtype).view(*table_shape), k_sin.to(k.dtype).view(*table_shape))

    if mode == "query":
        if q_phase is None:
            raise ValueError("query mode requires q_phase.")
        q_rot = apply_phase(q_rot, q_phase, channel_mask)
    elif mode == "segment":
        if q_phase is None or k_phase is None:
            raise ValueError("segment mode requires both q_phase and k_phase.")
        q_rot = apply_phase(q_rot, q_phase, channel_mask)
        k_rot = apply_phase(k_rot, k_phase, channel_mask)
    elif mode != "standard":
        raise ValueError(f"Unknown phase mode: {mode}")

    return q_rot @ k_rot.transpose(-1, -2) / math.sqrt(q.shape[-1])


def causal_attention_weights(
    scores: torch.Tensor,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply an optional additive control and causal mask, then normalize."""
    if bias is not None:
        scores = scores + bias
    allowed = k_positions.view(1, 1, 1, -1) <= q_positions.view(1, 1, -1, 1)
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=-1)


__all__ = [
    "PhaseMode",
    "apply_phase",
    "apply_rotary",
    "attention_scores",
    "causal_attention_weights",
    "reversal_phase",
    "rope_tables",
    "rotate_half",
]
