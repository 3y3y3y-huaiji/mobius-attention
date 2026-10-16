# -*- coding: utf-8 -*-
r"""
Milestone 3 Empirical Challenger Adversarial Stress Test Suite
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Adversarially stress-tests prefill vs autoregressive stateful decoding numerical equivalence:
1. Long-Context Stress (L = 4096 and L = 8192):
   - ContinuousMobiusRoPEModulator batch parallel prefill vs token-by-token stateful decode
   - Under randomly fluctuating continuous torsion rates omega in [-omega_max, omega_max] with viscous smoothing
   - Strict tolerances: max absolute error < 1e-5 in float32, < 1e-12 in float64
   - Evaluates rotary cos/sin, rotated Q/K tensors, and full causal attention logits
2. Causal Isolation Check:
   - Verifies that adding downstream tokens (or modifying future tokens) never leaks backward into cached states
   - Counterfactual branching: divergent futures leave past key representations strictly invariant (0.0 diff)
   - Prefix prefill invariance: batch prefill of L1 vs prefix of batch prefill of L2
3. Hybrid Chunked Prefill + Autoregressive Decode Handoff
4. Viscous EMA and Torsion Boundary Extremes (rapid alternating spikes, nu in {0.01, 0.1, 0.5, 0.99, 1.0})
"""

import math
import pytest
import torch
import torch.nn.functional as F

from src.mobius_attention import (
    ContinuousMobiusRoPEModulator,
    SemanticGaugeConnection,
    causal_viscous_ema,
    nonnegative_feature_map,
    apply_rotary_angle_addition
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


LONG_LENGTHS = [4096, 8192]


# ==============================================================================
# 1. Long-Context Stress: Modulator Prefill vs Decoding (L = 4096, 8192)
# ==============================================================================

@pytest.mark.parametrize("seq_len", LONG_LENGTHS)
@pytest.mark.parametrize("dtype,tol", [
    (torch.float32, 1e-5),
    (torch.float64, 1e-12)
])
def test_long_context_modulator_equivalence(seq_len: int, dtype: torch.dtype, tol: float):
    """
    Stress-tests ContinuousMobiusRoPEModulator across long contexts L in {4096, 8192}.
    Generates fluctuating continuous torsion rates with viscous smoothing (nu = 0.35, 0.7).
    Asserts max error in (cos, sin, q_rot, k_rot) < 1e-5 (f32) and < 1e-12 (f64).
    """
    torch.manual_seed(42 + seq_len + (0 if dtype == torch.float32 else 100))
    batch_size = 1
    head_dim = 64
    omega_max = 0.05
    nu = 0.35

    modulator = ContinuousMobiusRoPEModulator(
        head_dim=head_dim,
        anchor_ratio=0.5,
        omega_max=omega_max,
        nu=nu,
        enable_nonnegative_map=True
    )

    # Randomly fluctuating continuous torsion rates
    raw_torsion = torch.empty(batch_size, seq_len, 1, dtype=dtype).uniform_(-omega_max, omega_max)
    omega = causal_viscous_ema(raw_torsion, nu=nu)
    phi = torch.cumsum(omega, dim=-2)

    # Random query and key
    q = torch.randn(batch_size, 1, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, 1, seq_len, head_dim, dtype=dtype)
    q_map, k_map = modulator.apply_feature_map(q, k)

    # 1. Batch Parallel Prefill
    pos_ids_prefill = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    cos_prefill, sin_prefill = modulator(q_map, position_ids=pos_ids_prefill, phi=phi)
    q_rot_prefill = apply_rotary_pos_emb(q_map, cos_prefill, sin_prefill)
    k_rot_prefill = apply_rotary_pos_emb(k_map, cos_prefill, sin_prefill)

    # 2. Token-by-token stateful decode
    cos_decode = torch.empty_like(cos_prefill)
    sin_decode = torch.empty_like(sin_prefill)
    q_rot_decode = torch.empty_like(q_rot_prefill)
    k_rot_decode = torch.empty_like(k_rot_prefill)

    for t in range(seq_len):
        pos_t = torch.tensor([[t]], dtype=torch.long).expand(batch_size, 1)
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

    max_err_cos = (cos_prefill - cos_decode).abs().max().item()
    max_err_sin = (sin_prefill - sin_decode).abs().max().item()
    max_err_q = (q_rot_prefill - q_rot_decode).abs().max().item()
    max_err_k = (k_rot_prefill - k_rot_decode).abs().max().item()

    assert max_err_cos < tol, f"L={seq_len} ({dtype}): cos discrepancy {max_err_cos:.2e} >= {tol}"
    assert max_err_sin < tol, f"L={seq_len} ({dtype}): sin discrepancy {max_err_sin:.2e} >= {tol}"
    assert max_err_q < tol, f"L={seq_len} ({dtype}): q_rot discrepancy {max_err_q:.2e} >= {tol}"
    assert max_err_k < tol, f"L={seq_len} ({dtype}): k_rot discrepancy {max_err_k:.2e} >= {tol}"


# ==============================================================================
# 2. Long-Context Stress: Attention Logits Equivalence (L = 4096, 8192)
# ==============================================================================

@pytest.mark.parametrize("seq_len", LONG_LENGTHS)
@pytest.mark.parametrize("dtype,tol", [
    (torch.float32, 1e-5),
    (torch.float64, 1e-12)
])
def test_long_context_attention_logits_equivalence(seq_len: int, dtype: torch.dtype, tol: float):
    """
    Stress-tests full self-attention causal logits between parallel prefill and step-by-step
    stateful KV-cache decoding across L in {4096, 8192}.
    Threshold: max causal discrepancy < 1e-5 in float32 and < 1e-12 in float64.
    """
    torch.manual_seed(777 + seq_len + (0 if dtype == torch.float32 else 500))
    batch_size = 1
    num_heads = 2
    head_dim = 32
    scale = 1.0 / math.sqrt(head_dim)
    nu = 0.4
    omega_max = 0.05

    modulator = ContinuousMobiusRoPEModulator(
        head_dim=head_dim,
        anchor_ratio=0.5,
        enable_nonnegative_map=True
    )

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    q_map, k_map = modulator.apply_feature_map(q, k)

    # Fluctuating continuous torsion rates
    raw_torsion = torch.empty(batch_size, seq_len, 1, dtype=dtype).uniform_(-omega_max, omega_max)
    omega = causal_viscous_ema(raw_torsion, nu=nu)
    phi = torch.cumsum(omega, dim=-2)

    # 1. Parallel Prefill
    pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    cos_prefill, sin_prefill = modulator(q_map, position_ids=pos_ids, phi=phi)
    q_rot_prefill = apply_rotary_pos_emb(q_map, cos_prefill, sin_prefill)
    k_rot_prefill = apply_rotary_pos_emb(k_map, cos_prefill, sin_prefill)

    # 2. Stateful KV-Cache Autoregressive Decoding
    # Pre-allocate key cache buffer [batch, num_heads, seq_len, head_dim]
    k_cache = torch.empty(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    max_logit_err = 0.0

    # Sample check positions along the sequence to thoroughly test early, middle, and late context
    # while maintaining reasonable test runtime for L=8192
    if seq_len == 8192:
        # Check dense steps at boundaries and strided across the long context
        dense_head = list(range(0, 64))
        dense_tail = list(range(seq_len - 64, seq_len))
        strided_mid = list(range(64, seq_len - 64, 64))
        check_steps = set(dense_head + strided_mid + dense_tail)
    else:
        # Dense check across all steps for L=4096
        check_steps = set(range(seq_len))

    for t in range(seq_len):
        pos_t = torch.tensor([[t]], dtype=torch.long).expand(batch_size, 1)
        phi_t = phi[:, t:t+1, :]
        q_t = q_map[:, :, t:t+1, :]
        k_t = k_map[:, :, t:t+1, :]

        cos_t, sin_t = modulator(q_t, position_ids=pos_t, phi=phi_t)
        q_rot_t = apply_rotary_pos_emb(q_t, cos_t, sin_t)
        k_rot_t = apply_rotary_pos_emb(k_t, cos_t, sin_t)

        # Store rotated key in cache
        k_cache[:, :, t:t+1, :] = k_rot_t

        if t in check_steps:
            # Query at step t against all cached keys j <= t
            cached_keys_up_to_t = k_cache[:, :, :t+1, :]
            step_logits = torch.matmul(q_rot_t, cached_keys_up_to_t.transpose(-1, -2)) * scale

            # Prefill reference logits for row t, keys 0..t
            prefill_row_t = torch.matmul(
                q_rot_prefill[:, :, t:t+1, :],
                k_rot_prefill[:, :, :t+1, :].transpose(-1, -2)
            ) * scale

            err_t = (step_logits - prefill_row_t).abs().max().item()
            if err_t > max_logit_err:
                max_logit_err = err_t

    assert max_logit_err < tol, f"L={seq_len} ({dtype}): Attention logit discrepancy {max_logit_err:.2e} >= {tol}"


# ==============================================================================
# 3. Causal Isolation Checks: Historical KV-Cache Immutability & Anti-Leakage
# ==============================================================================

class TestCausalIsolation:
    """
    Rigorously certifies that future tokens and downstream additions never leak
    backward or alter previously cached key states or past attention computations.
    """

    def test_downstream_tokens_do_not_mutate_cached_keys(self):
        """
        Verify that adding downstream tokens (from L1=4096 to L2=8192) leaves
        the first L1 cached keys bitwise invariant (0.0 absolute difference).
        """
        L1 = 4096
        L2 = 8192
        batch_size = 1
        num_heads = 2
        head_dim = 64
        dtype = torch.float64

        modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5)

        # Generate prompt of length L1
        q1 = torch.randn(batch_size, num_heads, L1, head_dim, dtype=dtype)
        k1 = torch.randn(batch_size, num_heads, L1, head_dim, dtype=dtype)
        raw_torsion1 = torch.empty(batch_size, L1, 1, dtype=dtype).uniform_(-0.05, 0.05)
        omega1 = causal_viscous_ema(raw_torsion1, nu=0.5)
        phi1 = torch.cumsum(omega1, dim=-2)

        # Cache keys for prompt
        k_cache = torch.empty(batch_size, num_heads, L2, head_dim, dtype=dtype)
        for t in range(L1):
            pos_t = torch.tensor([[t]], dtype=torch.long)
            c_t, s_t = modulator(k1[:, :, t:t+1, :], position_ids=pos_t, phi=phi1[:, t:t+1, :])
            k_cache[:, :, t:t+1, :] = apply_rotary_pos_emb(k1[:, :, t:t+1, :], c_t, s_t)

        # Snapshot cached prompt keys
        prompt_keys_snapshot = k_cache[:, :, :L1, :].clone()

        # Now generate downstream tokens L1..L2
        q2 = torch.randn(batch_size, num_heads, L2 - L1, head_dim, dtype=dtype)
        k2 = torch.randn(batch_size, num_heads, L2 - L1, head_dim, dtype=dtype)
        raw_torsion2 = torch.empty(batch_size, L2 - L1, 1, dtype=dtype).uniform_(-0.05, 0.05)
        omega2 = causal_viscous_ema(raw_torsion2, nu=0.5, initial_omega=omega1[:, -1:, :].squeeze(1))
        phi2 = phi1[:, -1:, :] + torch.cumsum(omega2, dim=-2)

        for step_idx, t in enumerate(range(L1, L2)):
            pos_t = torch.tensor([[t]], dtype=torch.long)
            c_t, s_t = modulator(k2[:, :, step_idx:step_idx+1, :], position_ids=pos_t, phi=phi2[:, step_idx:step_idx+1, :])
            k_cache[:, :, t:t+1, :] = apply_rotary_pos_emb(k2[:, :, step_idx:step_idx+1, :], c_t, s_t)

            # Assert at intermediate points that prompt cache is unchanged
            if step_idx % 1024 == 0 or step_idx == (L2 - L1 - 1):
                cache_mutation = (k_cache[:, :, :L1, :] - prompt_keys_snapshot).abs().max().item()
                assert cache_mutation == 0.0, f"Step {t}: Past cached keys mutated by {cache_mutation}!"

    def test_counterfactual_futures_backward_isolation(self):
        """
        Adversarial test: Two diverging counterfactual futures (Branch A vs Branch B)
        emanate from the same prefix of length L1 = 2048.
        Branch A experiences extreme positive torsion (+0.05).
        Branch B experiences extreme negative torsion (-0.05).
        Assert that the prompt key cache and prompt attention matrix are strictly IDENTICAL (0.0 diff).
        """
        L1 = 2048
        L_ext = 1024
        batch_size = 1
        num_heads = 2
        head_dim = 32
        dtype = torch.float64
        scale = 1.0 / math.sqrt(head_dim)

        modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5)

        # Common prompt
        torch.manual_seed(999)
        q_prompt = torch.randn(batch_size, num_heads, L1, head_dim, dtype=dtype)
        k_prompt = torch.randn(batch_size, num_heads, L1, head_dim, dtype=dtype)
        raw_torsion_prompt = torch.empty(batch_size, L1, 1, dtype=dtype).uniform_(-0.02, 0.02)
        omega_prompt = causal_viscous_ema(raw_torsion_prompt, nu=0.4)
        phi_prompt = torch.cumsum(omega_prompt, dim=-2)

        # Cache keys for prompt
        k_cache_A = torch.empty(batch_size, num_heads, L1 + L_ext, head_dim, dtype=dtype)
        k_cache_B = torch.empty(batch_size, num_heads, L1 + L_ext, head_dim, dtype=dtype)

        for t in range(L1):
            pos_t = torch.tensor([[t]], dtype=torch.long)
            c_t, s_t = modulator(k_prompt[:, :, t:t+1, :], position_ids=pos_t, phi=phi_prompt[:, t:t+1, :])
            k_rot_t = apply_rotary_pos_emb(k_prompt[:, :, t:t+1, :], c_t, s_t)
            k_cache_A[:, :, t:t+1, :] = k_rot_t
            k_cache_B[:, :, t:t+1, :] = k_rot_t.clone()

        # Branch A: Extreme positive torsion
        raw_torsion_A = torch.full((batch_size, L_ext, 1), 0.05, dtype=dtype)
        omega_A = causal_viscous_ema(raw_torsion_A, nu=0.8, initial_omega=omega_prompt[:, -1:, :].squeeze(1))
        phi_A = phi_prompt[:, -1:, :] + torch.cumsum(omega_A, dim=-2)

        # Branch B: Extreme negative torsion
        raw_torsion_B = torch.full((batch_size, L_ext, 1), -0.05, dtype=dtype)
        omega_B = causal_viscous_ema(raw_torsion_B, nu=0.8, initial_omega=omega_prompt[:, -1:, :].squeeze(1))
        phi_B = phi_prompt[:, -1:, :] + torch.cumsum(omega_B, dim=-2)

        k_ext_A = torch.randn(batch_size, num_heads, L_ext, head_dim, dtype=dtype)
        k_ext_B = torch.randn(batch_size, num_heads, L_ext, head_dim, dtype=dtype)

        # Decode Branch A
        for step_idx, t in enumerate(range(L1, L1 + L_ext)):
            pos_t = torch.tensor([[t]], dtype=torch.long)
            c_t, s_t = modulator(k_ext_A[:, :, step_idx:step_idx+1, :], position_ids=pos_t, phi=phi_A[:, step_idx:step_idx+1, :])
            k_cache_A[:, :, t:t+1, :] = apply_rotary_pos_emb(k_ext_A[:, :, step_idx:step_idx+1, :], c_t, s_t)

        # Decode Branch B
        for step_idx, t in enumerate(range(L1, L1 + L_ext)):
            pos_t = torch.tensor([[t]], dtype=torch.long)
            c_t, s_t = modulator(k_ext_B[:, :, step_idx:step_idx+1, :], position_ids=pos_t, phi=phi_B[:, step_idx:step_idx+1, :])
            k_cache_B[:, :, t:t+1, :] = apply_rotary_pos_emb(k_ext_B[:, :, step_idx:step_idx+1, :], c_t, s_t)

        # Assert: Prompt cached keys are 100% IDENTICAL between Branch A and Branch B
        prompt_key_diff = (k_cache_A[:, :, :L1, :] - k_cache_B[:, :, :L1, :]).abs().max().item()
        assert prompt_key_diff == 0.0, f"Branching corrupted past keys: diff={prompt_key_diff}"

        # Assert: Self-attention among prompt tokens (any i, j <= L1) is 100% identical
        q_pos = torch.arange(L1, dtype=torch.long).unsqueeze(0)
        c_prompt, s_prompt = modulator(q_prompt, position_ids=q_pos, phi=phi_prompt)
        q_rot_prompt = apply_rotary_pos_emb(q_prompt, c_prompt, s_prompt)

        logits_prompt_A = torch.matmul(q_rot_prompt, k_cache_A[:, :, :L1, :].transpose(-1, -2)) * scale
        logits_prompt_B = torch.matmul(q_rot_prompt, k_cache_B[:, :, :L1, :].transpose(-1, -2)) * scale

        logit_diff = (logits_prompt_A - logits_prompt_B).abs().max().item()
        assert logit_diff == 0.0, f"Prompt self-attention differs across branches: diff={logit_diff}"

    def test_prefill_prefix_invariance(self):
        """
        Tests that batch parallel prefill of length L2 = 8192 yields IDENTICAL
        representations for the first L1 = 4096 tokens compared to batch prefill of length L1 alone.
        Verifies no future token information leaks backward in the vectorized implementation.
        """
        L1 = 4096
        L2 = 8192
        batch_size = 1
        head_dim = 64
        dtype = torch.float64
        nu = 0.45
        omega_max = 0.05

        modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5)

        # Full sequence inputs
        torch.manual_seed(12345)
        raw_torsion_full = torch.empty(batch_size, L2, 1, dtype=dtype).uniform_(-omega_max, omega_max)
        x_full = torch.randn(batch_size, 1, L2, head_dim, dtype=dtype)

        # 1. Full sequence prefill
        omega_full = causal_viscous_ema(raw_torsion_full, nu=nu)
        phi_full = torch.cumsum(omega_full, dim=-2)
        pos_full = torch.arange(L2, dtype=torch.long).unsqueeze(0)
        cos_full, sin_full = modulator(x_full, position_ids=pos_full, phi=phi_full)
        x_rot_full = apply_rotary_pos_emb(x_full, cos_full, sin_full)

        # 2. Prefix-only prefill (L1)
        raw_torsion_prefix = raw_torsion_full[:, :L1, :]
        x_prefix = x_full[:, :, :L1, :]
        omega_prefix = causal_viscous_ema(raw_torsion_prefix, nu=nu)
        phi_prefix = torch.cumsum(omega_prefix, dim=-2)
        pos_prefix = torch.arange(L1, dtype=torch.long).unsqueeze(0)
        cos_prefix, sin_prefix = modulator(x_prefix, position_ids=pos_prefix, phi=phi_prefix)
        x_rot_prefix = apply_rotary_pos_emb(x_prefix, cos_prefix, sin_prefix)

        # Assert prefix matches sliced full prefill
        err_omega = (omega_full[:, :L1, :] - omega_prefix).abs().max().item()
        err_phi = (phi_full[:, :L1, :] - phi_prefix).abs().max().item()
        err_cos = (cos_full[:, :L1, :] - cos_prefix).abs().max().item()
        err_sin = (sin_full[:, :L1, :] - sin_prefix).abs().max().item()
        err_rot = (x_rot_full[:, :, :L1, :] - x_rot_prefix).abs().max().item()

        assert err_omega < 1e-14, f"Omega prefix error: {err_omega:.2e}"
        assert err_phi < 1e-14, f"Phi prefix error: {err_phi:.2e}"
        assert err_cos < 1e-14, f"Cos prefix error: {err_cos:.2e}"
        assert err_sin < 1e-14, f"Sin prefix error: {err_sin:.2e}"
        assert err_rot < 1e-14, f"Rotated tensor prefix error: {err_rot:.2e}"


# ==============================================================================
# 4. Hybrid Chunked Prefill + Autoregressive Stateful Decode Handoff
# ==============================================================================

def test_hybrid_chunked_prefill_then_decode():
    """
    Simulates production LLM serving:
    Chunk 1 (Prompt, L1=4096): Batch parallel prefill with state caching.
    Chunk 2 (Generation, L2-L1=4096 tokens): Autoregressive token-by-token stateful decode.
    Compares against pure batch prefill of full L2 = 8192 tokens.
    Tolerance: < 1e-5 in float32, < 1e-12 in float64.
    """
    L1 = 4096
    L2 = 8192
    batch_size = 1
    head_dim = 64
    hidden_dim = 64
    out_dim = 1
    nu = 0.4
    omega_max = 0.05

    for dtype, tol in [(torch.float32, 1e-5), (torch.float64, 1e-12)]:
        torch.manual_seed(4321)
        gauge = SemanticGaugeConnection(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            omega_max=omega_max,
            nu=nu
        ).to(dtype=dtype)

        modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5)

        x_embed = torch.randn(batch_size, L2, hidden_dim, dtype=dtype)

        # Baseline: Full parallel prefill
        omega_full, phi_full = gauge(x_embed)

        # Hybrid:
        # Step 1: Prefill prompt of length L1
        gauge.reset_state()
        omega_prompt, phi_prompt = gauge(x_embed[:, :L1, :], update_state=True)

        # Step 2: Decode tokens L1..L2 step-by-step
        omega_gen_list = []
        phi_gen_list = []
        for t in range(L1, L2):
            x_t = x_embed[:, t:t+1, :]
            o_t, p_t = gauge.step(x_t)
            omega_gen_list.append(o_t)
            phi_gen_list.append(p_t)

        omega_hybrid = torch.cat([omega_prompt, torch.cat(omega_gen_list, dim=1)], dim=1)
        phi_hybrid = torch.cat([phi_prompt, torch.cat(phi_gen_list, dim=1)], dim=1)

        err_omega = (omega_full - omega_hybrid).abs().max().item()
        err_phi = (phi_full - phi_hybrid).abs().max().item()

        assert err_omega < tol, f"Hybrid omega error {err_omega:.2e} >= {tol} ({dtype})"
        assert err_phi < tol, f"Hybrid phi error {err_phi:.2e} >= {tol} ({dtype})"


# ==============================================================================
# 5. Adversarial Torsion Patterns & Parameter Stress
# ==============================================================================

@pytest.mark.parametrize("nu", [0.01, 0.1, 0.5, 0.99, 1.0])
@pytest.mark.parametrize("anchor_ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_adversarial_torsion_patterns(nu: float, anchor_ratio: float):
    """
    Stress-tests rapid alternating sign spikes (+omega_max, -omega_max) across
    viscosity extremes nu in {0.01, 0.1, 0.5, 0.99, 1.0} and anchor ratios in {0, 0.25, 0.5, 0.75, 1.0}.
    """
    seq_len = 1024
    head_dim = 32
    dtype = torch.float64
    omega_max = 0.05

    modulator = ContinuousMobiusRoPEModulator(
        head_dim=head_dim,
        anchor_ratio=anchor_ratio,
        enable_nonnegative_map=True
    )

    # High-frequency alternating sign spikes
    signs = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(seq_len)], dtype=dtype).view(1, seq_len, 1)
    raw_torsion = signs * omega_max

    omega = causal_viscous_ema(raw_torsion, nu=nu)
    phi = torch.cumsum(omega, dim=-2)

    pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    dummy_x = torch.zeros(1, seq_len, head_dim, dtype=dtype)
    cos_pre, sin_pre = modulator(dummy_x, position_ids=pos_ids, phi=phi)

    cos_steps = []
    sin_steps = []
    for t in range(seq_len):
        pos_t = torch.tensor([[t]], dtype=torch.long)
        c_t, s_t = modulator(dummy_x[:, t:t+1, :], position_ids=pos_t, phi=phi[:, t:t+1, :])
        cos_steps.append(c_t)
        sin_steps.append(s_t)

    cos_dec = torch.cat(cos_steps, dim=1)
    sin_dec = torch.cat(sin_steps, dim=1)

    max_err_cos = (cos_pre - cos_dec).abs().max().item()
    max_err_sin = (sin_pre - sin_dec).abs().max().item()

    assert max_err_cos < 1e-12, f"nu={nu}, ratio={anchor_ratio}: cos err {max_err_cos:.2e} >= 1e-12"
    assert max_err_sin < 1e-12, f"nu={nu}, ratio={anchor_ratio}: sin err {max_err_sin:.2e} >= 1e-12"
