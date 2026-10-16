# -*- coding: utf-8 -*-
"""
Standalone Metric Extractor for M3 Numerical Equivalence & Causal Isolation Challenger
Collects exact maximum and mean discrepancies for handoff report.
"""

import math
import json
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch

from src.mobius_attention import (
    ContinuousMobiusRoPEModulator,
    SemanticGaugeConnection,
    causal_viscous_ema,
    apply_rotary_angle_addition
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def run_metrics():
    results = {}

    # 1. Modulator & Attention Logits at L=4096 and L=8192
    for L in [4096, 8192]:
        for dtype, dtype_str in [(torch.float32, "float32"), (torch.float64, "float64")]:
            torch.manual_seed(100 + L + (0 if dtype == torch.float32 else 999))
            batch_size = 1
            num_heads = 2
            head_dim = 32
            scale = 1.0 / math.sqrt(head_dim)
            omega_max = 0.05
            nu = 0.35

            modulator = ContinuousMobiusRoPEModulator(
                head_dim=head_dim,
                anchor_ratio=0.5,
                omega_max=omega_max,
                nu=nu,
                enable_nonnegative_map=True
            )

            raw_torsion = torch.empty(batch_size, L, 1, dtype=dtype).uniform_(-omega_max, omega_max)
            omega = causal_viscous_ema(raw_torsion, nu=nu)
            phi = torch.cumsum(omega, dim=-2)

            q = torch.randn(batch_size, num_heads, L, head_dim, dtype=dtype)
            k = torch.randn(batch_size, num_heads, L, head_dim, dtype=dtype)
            q_map, k_map = modulator.apply_feature_map(q, k)

            pos_ids_prefill = torch.arange(L, dtype=torch.long).unsqueeze(0)
            cos_prefill, sin_prefill = modulator(q_map, position_ids=pos_ids_prefill, phi=phi)
            q_rot_prefill = apply_rotary_pos_emb(q_map, cos_prefill, sin_prefill)
            k_rot_prefill = apply_rotary_pos_emb(k_map, cos_prefill, sin_prefill)

            cos_decode = torch.empty_like(cos_prefill)
            sin_decode = torch.empty_like(sin_prefill)
            q_rot_decode = torch.empty_like(q_rot_prefill)
            k_rot_decode = torch.empty_like(k_rot_prefill)
            k_cache = torch.empty_like(k_rot_prefill)

            max_logit_err = 0.0
            sum_logit_err = 0.0
            num_logit_evals = 0

            # Evaluate sample points for logits
            eval_steps = list(range(0, 32)) + list(range(32, L - 32, 128)) + list(range(L - 32, L))

            for t in range(L):
                pos_t = torch.tensor([[t]], dtype=torch.long)
                phi_t = phi[:, t:t+1, :]
                q_t = q_map[:, :, t:t+1, :]
                k_t = k_map[:, :, t:t+1, :]

                c_t, s_t = modulator(q_t, position_ids=pos_t, phi=phi_t)
                qr_t = apply_rotary_pos_emb(q_t, c_t, s_t)
                kr_t = apply_rotary_pos_emb(k_t, c_t, s_t)

                cos_decode[:, t:t+1, :] = c_t
                sin_decode[:, t:t+1, :] = s_t
                q_rot_decode[:, :, t:t+1, :] = qr_t
                k_rot_decode[:, :, t:t+1, :] = kr_t
                k_cache[:, :, t:t+1, :] = kr_t

                if t in eval_steps:
                    cached_keys = k_cache[:, :, :t+1, :]
                    step_logits = torch.matmul(qr_t, cached_keys.transpose(-1, -2)) * scale
                    prefill_row = torch.matmul(
                        q_rot_prefill[:, :, t:t+1, :],
                        k_rot_prefill[:, :, :t+1, :].transpose(-1, -2)
                    ) * scale

                    diff = (step_logits - prefill_row).abs()
                    curr_max = diff.max().item()
                    if curr_max > max_logit_err:
                        max_logit_err = curr_max
                    sum_logit_err += diff.mean().item()
                    num_logit_evals += 1

            key_name = f"L{L}_{dtype_str}"
            results[key_name] = {
                "max_err_cos": (cos_prefill - cos_decode).abs().max().item(),
                "max_err_sin": (sin_prefill - sin_decode).abs().max().item(),
                "max_err_q_rot": (q_rot_prefill - q_rot_decode).abs().max().item(),
                "max_err_k_rot": (k_rot_prefill - k_rot_decode).abs().max().item(),
                "max_err_logits": max_logit_err,
                "mean_err_logits": sum_logit_err / num_logit_evals,
            }

    # 2. Causal Isolation Metrics
    # (a) Downstream extension key cache mutation
    L1, L2 = 4096, 8192
    q1 = torch.randn(1, 2, L1, 64, dtype=torch.float64)
    k1 = torch.randn(1, 2, L1, 64, dtype=torch.float64)
    modulator = ContinuousMobiusRoPEModulator(head_dim=64, anchor_ratio=0.5)
    raw_torsion1 = torch.empty(1, L1, 1, dtype=torch.float64).uniform_(-0.05, 0.05)
    omega1 = causal_viscous_ema(raw_torsion1, nu=0.5)
    phi1 = torch.cumsum(omega1, dim=-2)

    k_cache = torch.empty(1, 2, L2, 64, dtype=torch.float64)
    for t in range(L1):
        pos_t = torch.tensor([[t]], dtype=torch.long)
        c_t, s_t = modulator(k1[:, :, t:t+1, :], position_ids=pos_t, phi=phi1[:, t:t+1, :])
        k_cache[:, :, t:t+1, :] = apply_rotary_pos_emb(k1[:, :, t:t+1, :], c_t, s_t)

    snapshot = k_cache[:, :, :L1, :].clone()

    k2 = torch.randn(1, 2, L2 - L1, 64, dtype=torch.float64)
    raw_torsion2 = torch.empty(1, L2 - L1, 1, dtype=torch.float64).uniform_(-0.05, 0.05)
    omega2 = causal_viscous_ema(raw_torsion2, nu=0.5, initial_omega=omega1[:, -1:, :].squeeze(1))
    phi2 = phi1[:, -1:, :] + torch.cumsum(omega2, dim=-2)

    for step_idx, t in enumerate(range(L1, L2)):
        pos_t = torch.tensor([[t]], dtype=torch.long)
        c_t, s_t = modulator(k2[:, :, step_idx:step_idx+1, :], position_ids=pos_t, phi=phi2[:, step_idx:step_idx+1, :])
        k_cache[:, :, t:t+1, :] = apply_rotary_pos_emb(k2[:, :, step_idx:step_idx+1, :], c_t, s_t)

    cache_leakage = (k_cache[:, :, :L1, :] - snapshot).abs().max().item()
    results["causal_isolation_cache_mutation"] = cache_leakage

    # (b) Prefill prefix invariance
    raw_torsion_full = torch.empty(1, L2, 1, dtype=torch.float64).uniform_(-0.05, 0.05)
    omega_full = causal_viscous_ema(raw_torsion_full, nu=0.45)
    phi_full = torch.cumsum(omega_full, dim=-2)
    omega_prefix = causal_viscous_ema(raw_torsion_full[:, :L1, :], nu=0.45)
    phi_prefix = torch.cumsum(omega_prefix, dim=-2)

    results["prefill_prefix_invariance_omega_err"] = (omega_full[:, :L1, :] - omega_prefix).abs().max().item()
    results["prefill_prefix_invariance_phi_err"] = (phi_full[:, :L1, :] - phi_prefix).abs().max().item()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_metrics()
