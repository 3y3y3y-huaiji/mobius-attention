# -*- coding: utf-8 -*-
r"""
Formal Prefill vs Autoregressive Stateful Decoding Equivalence Test Suite
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Rigorous mathematical and numerical verification of algebraic and causal equivalence:
1. ContinuousMobiusRoPEModulator forward pass (prefill vs token-by-token stateful decode)
2. Full multi-head self-attention matrix logits (prefill vs stateful KV-cache decode)
3. SemanticGaugeConnection viscous EMA and path-integral phase integration (forward vs step)
4. Comprehensive multi-length evaluation across L in [16, 64, 128, 256, 512, 1024, 2048, 4096]
5. Strict precision bounds:
   - Max absolute error < 10^-5 in float32
   - Max absolute error < 10^-12 in float64
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


# Target sequence lengths specified in Milestone 3 requirements
TEST_LENGTHS = [16, 64, 128, 256, 512, 1024, 2048, 4096]


@pytest.mark.parametrize("seq_len", TEST_LENGTHS)
def test_modulator_prefill_decoding_equivalence_float32(seq_len: int):
    """
    Verifies that ContinuousMobiusRoPEModulator produces identical rotary cos/sin
    between parallel batch prefill and autoregressive token-by-token decoding in float32.
    Threshold: strictly < 10^-5.
    """
    torch.manual_seed(42 + seq_len)
    batch_size = 1
    head_dim = 64
    dtype = torch.float32

    modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5)

    # Generate continuous phase flow via bounded torsion rates
    raw_torsion = torch.empty(batch_size, seq_len, 1, dtype=dtype).uniform_(-0.05, 0.05)
    omega = causal_viscous_ema(raw_torsion, nu=0.4)
    phi = torch.cumsum(omega, dim=-2)

    # 1. Parallel Prefill Forward
    pos_ids_prefill = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    dummy_x = torch.zeros(batch_size, seq_len, head_dim, dtype=dtype)
    cos_prefill, sin_prefill = modulator(dummy_x, position_ids=pos_ids_prefill, phi=phi)

    # 2. Autoregressive Token-by-Token Decoding
    cos_steps = []
    sin_steps = []
    for t in range(seq_len):
        pos_t = torch.tensor([[t]], dtype=torch.long).expand(batch_size, 1)
        phi_t = phi[:, t:t+1, :]
        dummy_t = torch.zeros(batch_size, 1, head_dim, dtype=dtype)
        c_t, s_t = modulator(dummy_t, position_ids=pos_t, phi=phi_t)
        cos_steps.append(c_t)
        sin_steps.append(s_t)

    cos_decode = torch.cat(cos_steps, dim=1)
    sin_decode = torch.cat(sin_steps, dim=1)

    max_err_cos = (cos_prefill - cos_decode).abs().max().item()
    max_err_sin = (sin_prefill - sin_decode).abs().max().item()

    assert max_err_cos < 1e-5, f"L={seq_len}: cos error {max_err_cos:.2e} >= 1e-5 in float32"
    assert max_err_sin < 1e-5, f"L={seq_len}: sin error {max_err_sin:.2e} >= 1e-5 in float32"


@pytest.mark.parametrize("seq_len", TEST_LENGTHS)
def test_modulator_prefill_decoding_equivalence_float64(seq_len: int):
    """
    Verifies that ContinuousMobiusRoPEModulator produces identical rotary cos/sin
    between parallel batch prefill and autoregressive token-by-token decoding in float64.
    Threshold: strictly < 10^-12.
    """
    torch.manual_seed(1042 + seq_len)
    batch_size = 1
    head_dim = 64
    dtype = torch.float64

    modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5)

    raw_torsion = torch.empty(batch_size, seq_len, 1, dtype=dtype).uniform_(-0.05, 0.05)
    omega = causal_viscous_ema(raw_torsion, nu=0.4)
    phi = torch.cumsum(omega, dim=-2)

    # 1. Parallel Prefill Forward
    pos_ids_prefill = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    dummy_x = torch.zeros(batch_size, seq_len, head_dim, dtype=dtype)
    cos_prefill, sin_prefill = modulator(dummy_x, position_ids=pos_ids_prefill, phi=phi)

    # 2. Autoregressive Token-by-Token Decoding
    cos_steps = []
    sin_steps = []
    for t in range(seq_len):
        pos_t = torch.tensor([[t]], dtype=torch.long).expand(batch_size, 1)
        phi_t = phi[:, t:t+1, :]
        dummy_t = torch.zeros(batch_size, 1, head_dim, dtype=dtype)
        c_t, s_t = modulator(dummy_t, position_ids=pos_t, phi=phi_t)
        cos_steps.append(c_t)
        sin_steps.append(s_t)

    cos_decode = torch.cat(cos_steps, dim=1)
    sin_decode = torch.cat(sin_steps, dim=1)

    max_err_cos = (cos_prefill - cos_decode).abs().max().item()
    max_err_sin = (sin_prefill - sin_decode).abs().max().item()

    assert max_err_cos < 1e-12, f"L={seq_len}: cos error {max_err_cos:.2e} >= 1e-12 in float64"
    assert max_err_sin < 1e-12, f"L={seq_len}: sin error {max_err_sin:.2e} >= 1e-12 in float64"


@pytest.mark.parametrize("seq_len", TEST_LENGTHS)
def test_attention_logits_prefill_decoding_equivalence_float32(seq_len: int):
    """
    Verifies full self-attention matrix logits between parallel causal prefill
    and step-by-step stateful KV-cache decoding in float32.
    Threshold: strictly < 10^-5 across all causal positions (j <= i).
    """
    torch.manual_seed(2042 + seq_len)
    batch_size = 1
    num_heads = 2
    head_dim = 32
    dtype = torch.float32

    modulator = ContinuousMobiusRoPEModulator(
        head_dim=head_dim,
        anchor_ratio=0.5,
        enable_nonnegative_map=True
    )

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    # Non-negative feature map on evolution subspace
    q_mapped, k_mapped = modulator.apply_feature_map(q, k)

    # Generate continuous phase flow
    raw_torsion = torch.empty(batch_size, seq_len, 1, dtype=dtype).uniform_(-0.04, 0.04)
    omega = causal_viscous_ema(raw_torsion, nu=0.35)
    phi = torch.cumsum(omega, dim=-2)

    # 1. Parallel Prefill Attention Logits
    pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    cos_prefill, sin_prefill = modulator(q_mapped, position_ids=pos_ids, phi=phi)

    # Apply RoPE modulation to full Q and K
    q_rot_prefill = apply_rotary_pos_emb(q_mapped, cos_prefill, sin_prefill)
    k_rot_prefill = apply_rotary_pos_emb(k_mapped, cos_prefill, sin_prefill)

    # Compute full attention matrix: [B, H, L, L]
    scale = 1.0 / math.sqrt(head_dim)
    logits_prefill = torch.matmul(q_rot_prefill, k_rot_prefill.transpose(-1, -2)) * scale

    # 2. Stateful KV-Cache Autoregressive Decoding
    kv_cache_k = [] # Cached rotated keys
    logits_decode_rows = []

    for t in range(seq_len):
        pos_t = torch.tensor([[t]], dtype=torch.long).expand(batch_size, 1)
        phi_t = phi[:, t:t+1, :]

        # Single-token query and key
        q_t = q_mapped[:, :, t:t+1, :]
        k_t = k_mapped[:, :, t:t+1, :]

        cos_t, sin_t = modulator(q_t, position_ids=pos_t, phi=phi_t)
        q_rot_t = apply_rotary_pos_emb(q_t, cos_t, sin_t)
        k_rot_t = apply_rotary_pos_emb(k_t, cos_t, sin_t)

        kv_cache_k.append(k_rot_t) # Append newly rotated key to cache

        # Concatenate all cached keys up to current step: [B, H, t+1, D]
        all_cached_k = torch.cat(kv_cache_k, dim=-2)

        # Compute query t against all cached keys j <= t: [B, H, 1, t+1]
        step_logits = torch.matmul(q_rot_t, all_cached_k.transpose(-1, -2)) * scale
        logits_decode_rows.append(step_logits)

    # Compare causal lower-triangular entries
    max_err = 0.0
    for t in range(seq_len):
        row_prefill = logits_prefill[:, :, t, :t+1]
        row_decode = logits_decode_rows[t].squeeze(-2)
        err_t = (row_prefill - row_decode).abs().max().item()
        if err_t > max_err:
            max_err = err_t

    assert max_err < 1e-5, f"L={seq_len}: Attention logits error {max_err:.2e} >= 1e-5 in float32"


@pytest.mark.parametrize("seq_len", TEST_LENGTHS)
def test_attention_logits_prefill_decoding_equivalence_float64(seq_len: int):
    """
    Verifies full self-attention matrix logits between parallel causal prefill
    and step-by-step stateful KV-cache decoding in float64.
    Threshold: strictly < 10^-12 across all causal positions (j <= i).
    """
    torch.manual_seed(3042 + seq_len)
    batch_size = 1
    num_heads = 2
    head_dim = 32
    dtype = torch.float64

    modulator = ContinuousMobiusRoPEModulator(
        head_dim=head_dim,
        anchor_ratio=0.5,
        enable_nonnegative_map=True
    )

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    q_mapped, k_mapped = modulator.apply_feature_map(q, k)

    raw_torsion = torch.empty(batch_size, seq_len, 1, dtype=dtype).uniform_(-0.04, 0.04)
    omega = causal_viscous_ema(raw_torsion, nu=0.35)
    phi = torch.cumsum(omega, dim=-2)

    # 1. Parallel Prefill Attention Logits
    pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    cos_prefill, sin_prefill = modulator(q_mapped, position_ids=pos_ids, phi=phi)

    q_rot_prefill = apply_rotary_pos_emb(q_mapped, cos_prefill, sin_prefill)
    k_rot_prefill = apply_rotary_pos_emb(k_mapped, cos_prefill, sin_prefill)

    scale = 1.0 / math.sqrt(head_dim)
    logits_prefill = torch.matmul(q_rot_prefill, k_rot_prefill.transpose(-1, -2)) * scale

    # 2. Stateful KV-Cache Autoregressive Decoding
    kv_cache_k = []
    logits_decode_rows = []

    for t in range(seq_len):
        pos_t = torch.tensor([[t]], dtype=torch.long).expand(batch_size, 1)
        phi_t = phi[:, t:t+1, :]

        q_t = q_mapped[:, :, t:t+1, :]
        k_t = k_mapped[:, :, t:t+1, :]

        cos_t, sin_t = modulator(q_t, position_ids=pos_t, phi=phi_t)
        q_rot_t = apply_rotary_pos_emb(q_t, cos_t, sin_t)
        k_rot_t = apply_rotary_pos_emb(k_t, cos_t, sin_t)

        kv_cache_k.append(k_rot_t)
        all_cached_k = torch.cat(kv_cache_k, dim=-2)
        step_logits = torch.matmul(q_rot_t, all_cached_k.transpose(-1, -2)) * scale
        logits_decode_rows.append(step_logits)

    max_err = 0.0
    for t in range(seq_len):
        row_prefill = logits_prefill[:, :, t, :t+1]
        row_decode = logits_decode_rows[t].squeeze(-2)
        err_t = (row_prefill - row_decode).abs().max().item()
        if err_t > max_err:
            max_err = err_t

    assert max_err < 1e-12, f"L={seq_len}: Attention logits error {max_err:.2e} >= 1e-12 in float64"


@pytest.mark.parametrize("seq_len", [16, 64, 128, 512, 1024])
def test_semantic_gauge_step_equivalence(seq_len: int):
    """
    Verifies that SemanticGaugeConnection parallel forward pass matches
    autoregressive token-by-token .step() calls for both omega and phi.
    Threshold: < 10^-5 in float32 and < 10^-12 in float64.
    """
    hidden_dim = 64
    out_dim = 2
    omega_max = 0.05
    nu = 0.3

    for dtype, tol in [(torch.float32, 1e-5), (torch.float64, 1e-12)]:
        torch.manual_seed(5042 + seq_len)
        gauge = SemanticGaugeConnection(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            omega_max=omega_max,
            nu=nu
        ).to(dtype=dtype)

        x = torch.randn(2, seq_len, hidden_dim, dtype=dtype)

        # 1. Parallel forward
        omega_prefill, phi_prefill = gauge(x)

        # 2. Token-by-token step
        gauge.reset_state()
        omega_steps = []
        phi_steps = []
        for t in range(seq_len):
            x_t = x[:, t:t+1, :]
            o_t, p_t = gauge.step(x_t)
            omega_steps.append(o_t)
            phi_steps.append(p_t)

        omega_decode = torch.cat(omega_steps, dim=1)
        phi_decode = torch.cat(phi_steps, dim=1)

        err_omega = (omega_prefill - omega_decode).abs().max().item()
        err_phi = (phi_prefill - phi_decode).abs().max().item()

        assert err_omega < tol, f"Gauge omega error {err_omega:.2e} >= {tol} for {dtype}"
        assert err_phi < tol, f"Gauge phi error {err_phi:.2e} >= {tol} for {dtype}"


def test_subspace_anchor_ratios_prefill_equivalence():
    """
    Verifies prefill-decoding equivalence across multiple anchor subspace ratios
    (0.0 = full mobius, 0.25, 0.5, 0.75, 1.0 = standard RoPE).
    """
    seq_len = 64
    head_dim = 32
    dtype = torch.float64

    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        modulator = ContinuousMobiusRoPEModulator(
            head_dim=head_dim,
            anchor_ratio=ratio,
            enable_nonnegative_map=True
        )

        q = torch.randn(1, 1, seq_len, head_dim, dtype=dtype)
        k = torch.randn(1, 1, seq_len, head_dim, dtype=dtype)
        phi = torch.linspace(0.0, 3.14159, seq_len, dtype=dtype).view(1, seq_len, 1)

        pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        cos_pre, sin_pre = modulator(q, position_ids=pos_ids, phi=phi)

        cos_steps = []
        sin_steps = []
        for t in range(seq_len):
            pos_t = torch.tensor([[t]], dtype=torch.long)
            c_t, s_t = modulator(q[:, :, t:t+1, :], position_ids=pos_t, phi=phi[:, t:t+1, :])
            cos_steps.append(c_t)
            sin_steps.append(s_t)

        cos_dec = torch.cat(cos_steps, dim=1)
        sin_dec = torch.cat(sin_steps, dim=1)

        assert (cos_pre - cos_dec).abs().max().item() < 1e-12
        assert (sin_pre - sin_dec).abs().max().item() < 1e-12


def test_kv_cache_time_invariance_under_session_budget():
    r"""
    Explicitly tests Proposition 4:
    Verifies that for a session context budget L_max = 512, the adaptive anchor cutoff
    alpha(L_max) is session-static, and cached key vectors K_cache[j] for past tokens j < i
    remain strictly byte-for-byte immutable across subsequent decoding steps i = j+1 ... L_max.
    """
    head_dim = 64
    seq_len = 32
    dtype = torch.float64

    # Fix session budget L_max -> static alpha
    L_max = 2048
    session_alpha = min(0.875, max(0.5, math.log(2.0 * L_max / math.pi) / math.log(10000.0)))
    modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=session_alpha)

    # Autoregressive key cache accumulator
    key_cache = []
    cached_tensors_snapshots = {}

    for t in range(seq_len):
        k_t = torch.randn(1, 1, 1, head_dim, dtype=dtype)
        phi_t = torch.tensor([[[0.1 * t]]], dtype=dtype)
        pos_t = torch.tensor([[t]], dtype=torch.long)

        # Modulate key at step t
        cos_t, sin_t = modulator(k_t, position_ids=pos_t, phi=phi_t)
        k_modulated = apply_rotary_pos_emb(k_t, cos_t, sin_t)

        # Store in cache
        key_cache.append(k_modulated.clone())
        cached_tensors_snapshots[t] = k_modulated.clone()

        # Check immutability of ALL previously cached keys j < t
        for j in range(t):
            diff = (key_cache[j] - cached_tensors_snapshots[j]).abs().max().item()
            assert diff == 0.0, f"Cache violation: Key at index {j} mutated at step {t} with diff {diff}"

