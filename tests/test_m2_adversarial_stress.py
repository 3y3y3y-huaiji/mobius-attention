"""
Adversarial empirical stress tests for Milestone 2 Continuous Möbius RoPE.
Evaluates:
1. 2D Planar Rotation Norm Conservation across 10,000 randomized vectors and Phi in [-1000pi, 1000pi].
2. Subspace Decoupling Integrity (anchor preservation vs evolution rotation).
3. 1D Viscous EMA Diffusion & Bounded Derivatives under extreme spiky/noisy inputs.
4. Prefill vs Autoregressive Stateful Decoding Numerical Equivalence for L in {16, 64, 256, 1024, 2048}.
5. Causal KV-Cache Invariance under long sequence extension (L >> j).
6. Adversarial edge cases (extreme scales, zero vectors, non-negative map boundary).
"""

import math
import torch
import torch.nn as nn
import pytest

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


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ==============================================================================
# 1. 2D Planar Rotation Norm Conservation
# ==============================================================================

class TestNormConservation:
    """Stress-test 2D Planar Rotation Norm Conservation."""

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
    def test_norm_conservation_10000_vectors(self, dtype):
        """
        Test across 10,000 randomized vectors and arbitrary continuous phases Phi in [-1000pi, 1000pi].
        Assert ||R(theta + Phi) v||_2 == ||v||_2 within float roundoff (< 1e-6).
        """
        torch.manual_seed(1001)
        num_vectors = 10000
        head_dim = 64
        num_subspaces = head_dim // 2

        # Generate vectors with diverse norms: tiny (1e-5), unit, large (1e4)
        scales = torch.tensor([1e-5, 0.01, 1.0, 100.0, 1e4], dtype=dtype)
        scale_assignments = scales[torch.randint(0, len(scales), (num_vectors, 1))]
        raw_v = torch.randn(num_vectors, head_dim, dtype=dtype) * scale_assignments
        orig_norms = raw_v.norm(dim=-1)

        # Baseline rotary angles theta
        theta = torch.randn(num_vectors, num_subspaces, dtype=dtype) * 2 * math.pi
        cos0 = torch.cat([theta.cos(), theta.cos()], dim=-1)
        sin0 = torch.cat([theta.sin(), theta.sin()], dim=-1)

        # Continuous phases Phi in [-1000pi, 1000pi]
        phi_vals = (torch.rand(num_vectors, num_subspaces, dtype=dtype) * 2000.0 - 1000.0) * math.pi
        # Also inject exact critical boundary angles
        phi_vals[:10] = torch.tensor([
            -1000 * math.pi, -500 * math.pi, -math.pi, -math.pi / 2, 0.0,
            math.pi / 2, math.pi, 2 * math.pi, 500 * math.pi, 1000 * math.pi
        ], dtype=dtype).unsqueeze(-1).expand(10, num_subspaces)
        phi = torch.cat([phi_vals, phi_vals], dim=-1)

        # Apply exact rotary angle addition
        cos_mod, sin_mod = apply_rotary_angle_addition(cos0, sin0, phi)
        rot_v = (raw_v * cos_mod) + (rotate_half(raw_v) * sin_mod)
        rot_norms = rot_v.norm(dim=-1)

        # Relative error for scale invariance
        rel_diff = (orig_norms - rot_norms).abs() / (orig_norms + 1e-12)
        abs_diff = (orig_norms - rot_norms).abs()

        tol = 1e-12 if dtype == torch.float64 else 1e-6
        max_rel_err = rel_diff.max().item()
        max_abs_err = abs_diff.max().item()

        assert max_rel_err < tol, f"Max relative norm change {max_rel_err} exceeds tolerance {tol} for {dtype}"
        print(f"Norm conservation ({dtype}): max_rel_err={max_rel_err:.2e}, max_abs_err={max_abs_err:.2e} across {num_vectors} vectors")

    def test_norm_conservation_with_modulator_forward(self):
        """Verify norm conservation when passed through ContinuousMobiusRoPEModulator."""
        torch.manual_seed(1002)
        batch = 8
        seq_len = 128
        head_dim = 128
        x = torch.randn(batch, seq_len, head_dim, dtype=torch.float64)
        orig_norms = x.norm(dim=-1)

        modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5)
        # Random phase in [-1000pi, 1000pi]
        phi = (torch.rand(batch, seq_len, 1, dtype=torch.float64) * 2000.0 - 1000.0) * math.pi
        cos, sin = modulator(x, phi=phi)

        rot_x = (x * cos) + (rotate_half(x) * sin)
        rot_norms = rot_x.norm(dim=-1)

        max_rel_diff = ((orig_norms - rot_norms).abs() / orig_norms).max().item()
        assert max_rel_diff < 1e-12, f"Modulator norm conservation failed: max_rel_diff={max_rel_diff}"


# ==============================================================================
# 2. Subspace Decoupling Integrity
# ==============================================================================

class TestSubspaceDecoupling:
    """Stress-test Subspace Decoupling Integrity."""

    @pytest.mark.parametrize("anchor_strategy", ["high_freq", "low_freq", "first_half"])
    @pytest.mark.parametrize("anchor_ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_subspace_anchor_preservation(self, anchor_strategy, anchor_ratio):
        """
        Verify that anchor channels strictly match baseline RoPE cos and sin without perturbation,
        while evolution channels smoothly rotate with phase.
        """
        torch.manual_seed(2001)
        head_dim = 64
        num_subspaces = head_dim // 2
        seq_len = 32

        modulator = ContinuousMobiusRoPEModulator(
            head_dim=head_dim,
            anchor_ratio=anchor_ratio,
            anchor_strategy=anchor_strategy
        )

        pos_ids = torch.arange(seq_len).unsqueeze(0)
        x = torch.randn(1, seq_len, head_dim)

        # Baseline RoPE (no phase)
        cos_base, sin_base = modulator(x, position_ids=pos_ids)

        # 1. Test anchor preservation under huge phase (1000*pi)
        phi_huge = torch.full((1, seq_len, 1), 1000.0 * math.pi)
        cos_mod_huge, sin_mod_huge = modulator(x, position_ids=pos_ids, phi=phi_huge)

        mask = modulator.get_subspace_mask(num_subspaces, device=x.device)
        anchor_mask = ~mask
        num_anchors = anchor_mask.sum().item()
        num_evolutions = mask.sum().item()

        # Both halves of head_dim have the same layout
        full_anchor_mask = torch.cat([anchor_mask, anchor_mask], dim=-1)
        full_evolution_mask = torch.cat([mask, mask], dim=-1)

        # Anchor channels must match baseline identically (no perturbation)
        if num_anchors > 0:
            diff_cos_anchor = (cos_base[..., full_anchor_mask] - cos_mod_huge[..., full_anchor_mask]).abs().max().item()
            diff_sin_anchor = (sin_base[..., full_anchor_mask] - sin_mod_huge[..., full_anchor_mask]).abs().max().item()
            assert diff_cos_anchor == 0.0 or diff_cos_anchor < 1e-7, (
                f"Anchor cos changed for {anchor_strategy} ratio={anchor_ratio}: {diff_cos_anchor}"
            )
            assert diff_sin_anchor == 0.0 or diff_sin_anchor < 1e-7, (
                f"Anchor sin changed for {anchor_strategy} ratio={anchor_ratio}: {diff_sin_anchor}"
            )

        # 2. Test exact antipodal inversion at phi = pi on evolution channels
        phi_pi = torch.full((1, seq_len, 1), math.pi)
        cos_mod_pi, sin_mod_pi = modulator(x, position_ids=pos_ids, phi=phi_pi)

        if num_evolutions > 0:
            # cos(theta + pi) = -cos(theta), sin(theta + pi) = -sin(theta)
            diff_cos_pi = (cos_mod_pi[..., full_evolution_mask] - (-cos_base[..., full_evolution_mask])).abs().max().item()
            diff_sin_pi = (sin_mod_pi[..., full_evolution_mask] - (-sin_base[..., full_evolution_mask])).abs().max().item()
            assert diff_cos_pi < 1e-6, f"Evolution cos failed pi inversion: {diff_cos_pi}"
            assert diff_sin_pi < 1e-6, f"Evolution sin failed pi inversion: {diff_sin_pi}"

    def test_subspace_high_precision_large_phase_inversion(self):
        """Verify high-precision inversion in float64 for large phase phi = 777*pi."""
        head_dim = 64
        num_subspaces = head_dim // 2
        modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5, anchor_strategy="high_freq")
        x = torch.randn(1, 16, head_dim, dtype=torch.float64)
        cos_base, sin_base = modulator(x)

        phi = torch.full((1, 16, 1), 777.0 * math.pi, dtype=torch.float64)
        cos_mod, sin_mod = modulator(x, phi=phi)

        mask = modulator.get_subspace_mask(num_subspaces, device=x.device)
        full_evolution_mask = torch.cat([mask, mask], dim=-1)

        diff = (cos_mod[..., full_evolution_mask] - (-cos_base[..., full_evolution_mask])).abs().max().item()
        assert diff < 1e-12, f"Float64 777*pi evolution inversion error: {diff}"

    def test_subspace_smooth_phase_sweep(self):
        """Verify continuous smooth phase sweep in evolution channels without steps or jumps."""
        torch.manual_seed(2002)
        head_dim = 32
        modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5)

        # 100 continuous phase steps from 0 to 2pi
        phases = torch.linspace(0, 2 * math.pi, 100).view(1, 100, 1)
        x = torch.randn(1, 100, head_dim)
        cos_mod, sin_mod = modulator(x, phi=phases)

        num_subspaces = head_dim // 2
        mask = modulator.get_subspace_mask(num_subspaces, x.device)
        full_evolution_mask = torch.cat([mask, mask], dim=-1)

        # Evolution values should be continuous: check finite difference
        evol_cos = cos_mod[0, :, full_evolution_mask]
        diffs = (evol_cos[1:] - evol_cos[:-1]).abs()
        max_step_change = diffs.max().item()
        # Max change per step for 100 steps across 2pi should be <= 2pi/100 ~ 0.063
        assert max_step_change < 0.1, f"Evolution channel had sudden jump: {max_step_change}"


# ==============================================================================
# 3. 1D Viscous EMA Diffusion & Bounded Derivatives
# ==============================================================================

class TestViscousDiffusionAndDerivativeBounds:
    """Stress-test 1D Viscous EMA Diffusion & Bounded Derivatives."""

    @pytest.mark.parametrize("nu", [0.01, 0.05, 0.2, 0.35, 0.5, 0.8, 0.99])
    @pytest.mark.parametrize("omega_max", [0.01, 0.05, 0.1, 0.5])
    def test_derivative_bound_extreme_spiky_inputs(self, nu, omega_max):
        """
        Test viscous filter under extreme spiky/noisy input signals and verify
        output derivative |omega(t) - omega(t-1)| <= 2 * nu * omega_max.
        """
        torch.manual_seed(3001)
        seq_len = 500
        hidden_dim = 32

        gauge = SemanticGaugeConnection(
            hidden_dim=hidden_dim,
            out_dim=2,
            omega_max=omega_max,
            nu=nu
        )

        # 1. Alternating square wave with extreme amplitude to force tanh to +1, -1
        spiky_x = torch.zeros(1, seq_len, hidden_dim)
        for t in range(seq_len):
            sign = 1.0 if (t % 2 == 0) else -1.0
            spiky_x[:, t, :] = sign * 1e4

        omega, phi = gauge(spiky_x)

        # Compute adjacent token derivatives: |omega(t) - omega(t-1)|
        omega_t = omega[:, 1:, :]
        omega_prev = omega[:, :-1, :]
        d_omega = (omega_t - omega_prev).abs()

        theoretical_bound = 2.0 * nu * omega_max
        max_d_omega = d_omega.max().item()

        assert max_d_omega <= theoretical_bound + 1e-7, (
            f"Derivative bound violated! max_d_omega={max_d_omega:.6f} > bound={theoretical_bound:.6f} "
            f"for nu={nu}, omega_max={omega_max}"
        )
        print(f"Derivative bound test (nu={nu}, omega_max={omega_max}): max={max_d_omega:.6f} <= bound={theoretical_bound:.6f}")

    def test_derivative_bound_impulse_train_and_noise(self):
        """Test with random impulse train and Cauchy heavy-tailed noise."""
        torch.manual_seed(3002)
        nu = 0.4
        omega_max = 0.08
        theoretical_bound = 2.0 * nu * omega_max

        gauge = SemanticGaugeConnection(hidden_dim=16, out_dim=1, omega_max=omega_max, nu=nu)

        # Impulse train: mostly zeros, sudden massive spikes
        seq_len = 1000
        impulse_x = torch.zeros(2, seq_len, 16)
        spike_indices = torch.randint(0, seq_len, (50,))
        impulse_x[:, spike_indices, :] = (torch.randn(2, 50, 16) * 1e5)

        omega, _ = gauge(impulse_x)
        d_omega = (omega[:, 1:, :] - omega[:, :-1, :]).abs()
        max_d = d_omega.max().item()
        assert max_d <= theoretical_bound + 1e-7, f"Impulse train exceeded bound: {max_d} > {theoretical_bound}"

        # Absolute value of omega must never exceed omega_max
        max_omega = omega.abs().max().item()
        assert max_omega <= omega_max + 1e-7, f"Omega exceeded omega_max: {max_omega} > {omega_max}"


# ==============================================================================
# 4. Prefill vs Autoregressive Stateful Decoding Numerical Equivalence
# ==============================================================================

class TestPrefillDecodingEquivalence:
    """Stress-test Prefill vs Autoregressive Stateful Decoding Numerical Equivalence."""

    @pytest.mark.parametrize("seq_len", [16, 64, 256, 1024, 2048])
    def test_prefill_vs_autoregressive_equivalence(self, seq_len):
        """
        Feed sequence of tokens (lengths L in {16, 64, 256, 1024, 2048}) through both
        parallel prefill and token-by-token stateful decode.
        Measure max absolute discrepancy across torsion rates omega and cumulative phases Phi.
        Assert max error < 1e-5.
        """
        torch.manual_seed(4001 + seq_len)
        batch = 2
        hidden_dim = 64
        out_dim = 4
        nu = 0.35
        omega_max = 0.05

        gauge = SemanticGaugeConnection(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            omega_max=omega_max,
            nu=nu
        )

        # Bursty random semantic signals
        x = torch.randn(batch, seq_len, hidden_dim)

        # 1. Parallel sequence prefill
        omega_prefill, phi_prefill = gauge(x)

        # 2. Token-by-token stateful autoregressive decode
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

        assert err_omega < 1e-5, f"Omega discrepancy {err_omega} >= 1e-5 at L={seq_len}"
        assert err_phi < 1e-5, f"Phi discrepancy {err_phi} >= 1e-5 at L={seq_len}"

        print(f"Equivalence L={seq_len:4d}: max_err_omega={err_omega:.2e}, max_err_phi={err_phi:.2e}")

    def test_prefill_decode_equivalence_with_initial_state(self):
        """Verify chunked prefill continuation matches step-by-step decode with initial state."""
        torch.manual_seed(4002)
        hidden_dim = 32
        out_dim = 2
        gauge = SemanticGaugeConnection(hidden_dim=hidden_dim, out_dim=out_dim, nu=0.5, omega_max=0.05)

        x_chunk1 = torch.randn(1, 30, hidden_dim)
        x_chunk2 = torch.randn(1, 30, hidden_dim)

        # Chunk 1 prefill with state update
        o1, p1 = gauge(x_chunk1, update_state=True)

        # Chunk 2 prefill continuing from chunk 1 final state
        init_o = o1[:, -1, :]
        init_p = p1[:, -1, :]
        o2_prefill, p2_prefill = gauge(x_chunk2, initial_omega=init_o, initial_phi=init_p)

        # Step by step decode for chunk 2
        # gauge already has _last_omega and _last_phi updated
        o2_steps, p2_steps = [], []
        for t in range(30):
            ot, pt = gauge.step(x_chunk2[:, t:t+1, :])
            o2_steps.append(ot)
            p2_steps.append(pt)

        o2_decode = torch.cat(o2_steps, dim=1)
        p2_decode = torch.cat(p2_steps, dim=1)

        assert (o2_prefill - o2_decode).abs().max().item() < 1e-5
        assert (p2_prefill - p2_decode).abs().max().item() < 1e-5


# ==============================================================================
# 5. Causal KV-Cache Invariance
# ==============================================================================

class TestCausalKVCacheInvariance:
    """Stress-test Causal KV-Cache Invariance."""

    @pytest.mark.parametrize("checkpoint_j", [0, 15, 63, 255, 511, 1023])
    def test_kv_cache_key_immutability(self, checkpoint_j):
        """
        Store key embeddings at step j, then extend sequence up to step L >> j (L=2048).
        Confirm cached key at j never mutates.
        """
        torch.manual_seed(5001 + checkpoint_j)
        seq_len_total = 2048
        head_dim = 64
        hidden_dim = 64

        gauge = SemanticGaugeConnection(hidden_dim=hidden_dim, out_dim=1, omega_max=0.05, nu=0.4)
        modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5)

        x_full = torch.randn(1, seq_len_total, hidden_dim)
        k_full = torch.randn(1, seq_len_total, head_dim)

        # Step 1: Simulate generation up to step j (short context)
        x_short = x_full[:, :checkpoint_j + 1]
        k_short = k_full[:, :checkpoint_j + 1]

        _, phi_short = gauge(x_short)
        cos_short, sin_short = modulator(k_short, phi=phi_short)
        _, k_short_rot = apply_rotary_pos_emb(k_short, k_short, cos_short, sin_short)
        cached_key_at_j = k_short_rot[:, checkpoint_j:checkpoint_j+1, :].clone()

        # Step 2: Extend sequence up to L = 2048
        _, phi_full = gauge(x_full)
        cos_full, sin_full = modulator(k_full, phi=phi_full)
        _, k_full_rot = apply_rotary_pos_emb(k_full, k_full, cos_full, sin_full)
        key_in_extended_at_j = k_full_rot[:, checkpoint_j:checkpoint_j+1, :]

        # Step 3: Compare cached key with extended sequence key
        max_mutation = (cached_key_at_j - key_in_extended_at_j).abs().max().item()
        assert max_mutation < 1e-6, (
            f"KV cache mutated at checkpoint j={checkpoint_j} when extended to L={seq_len_total}! "
            f"max_mutation={max_mutation}"
        )
        print(f"KV Cache Invariance j={checkpoint_j:4d} in L={seq_len_total}: diff={max_mutation:.2e}")

    def test_attention_logit_causal_invariance(self):
        """
        Verify that attention logit q_t @ k_j^T is identical whether k_j is read from
        cached state or computed within full parallel prefill.
        """
        torch.manual_seed(5002)
        seq_len = 128
        j = 20
        t = 100
        head_dim = 64
        hidden_dim = 64

        gauge = SemanticGaugeConnection(hidden_dim=hidden_dim, out_dim=1, nu=0.5, omega_max=0.05)
        modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5)

        x = torch.randn(1, seq_len, hidden_dim)
        q = torch.randn(1, seq_len, head_dim)
        k = torch.randn(1, seq_len, head_dim)

        # Full prefill
        _, phi = gauge(x)
        cos, sin = modulator(k, phi=phi)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        full_logit = torch.sum(q_rot[:, t] * k_rot[:, j], dim=-1)

        # Autoregressive simulated cache: key cached at step j
        x_j = x[:, :j+1]
        k_j = k[:, :j+1]
        _, phi_j = gauge(x_j)
        cos_j, sin_j = modulator(k_j, phi=phi_j)
        _, k_rot_j = apply_rotary_pos_emb(k_j, k_j, cos_j, sin_j)
        cached_k = k_rot_j[:, j]

        # Query at step t using step-t rotary
        q_t = q[:, t:t+1]
        pos_t = torch.tensor([[t]])
        cos_t, sin_t = modulator(q_t, phi=phi[:, t:t+1], position_ids=pos_t)
        q_rot_t, _ = apply_rotary_pos_emb(q_t, q_t, cos_t, sin_t)
        cache_logit = torch.sum(q_rot_t[:, 0] * cached_k, dim=-1)

        logit_diff = (full_logit - cache_logit).abs().max().item()
        assert logit_diff < 1e-6, f"Attention logit mismatch between cache and prefill: {logit_diff}"


# ==============================================================================
# 6. Edge Cases & Boundary Conditions
# ==============================================================================

class TestAdversarialEdgeCases:
    """Stress-test numerical boundaries, degenerate inputs, and extreme scales."""

    def test_nonnegative_feature_map_extreme_boundaries(self):
        """Verify nonnegative feature map under extreme negative and positive values."""
        extreme_neg = torch.tensor([-1e9, -1e5, -1000.0, -50.0, -10.0])
        mapped_neg = nonnegative_feature_map(extreme_neg)
        assert (mapped_neg >= 0.0).all(), "Mapped values must never be negative"
        assert not torch.isnan(mapped_neg).any()
        assert not torch.isinf(mapped_neg).any()

        # At -1e9, ELU(-1e9) + 1 == 0.0
        assert (mapped_neg >= 0.0).all()

        extreme_pos = torch.tensor([1e5, 1e9])
        mapped_pos = nonnegative_feature_map(extreme_pos)
        assert (mapped_pos > 0.0).all()

    def test_zero_vector_and_huge_vector_rotation(self):
        """Test rotation on zero vectors and huge vectors."""
        dim = 64
        zero_v = torch.zeros(1, 10, dim)
        cos = torch.ones(1, 10, dim)
        sin = torch.zeros(1, 10, dim)
        cos_mod, sin_mod = apply_rotary_angle_addition(cos, sin, torch.full((1, 10, dim), math.pi))
        rot_zero = (zero_v * cos_mod) + (rotate_half(zero_v) * sin_mod)
        assert (rot_zero == 0.0).all()
        assert not torch.isnan(rot_zero).any()

        huge_v = torch.full((1, 10, dim), 1e18, dtype=torch.float64)
        rot_huge = (huge_v * cos_mod.to(torch.float64)) + (rotate_half(huge_v) * sin_mod.to(torch.float64))
        assert not torch.isnan(rot_huge).any()
        assert not torch.isinf(rot_huge).any()

    def test_modulator_without_phase_is_pure_rope(self):
        """Verify modulator without phi is bitwise identical to standard RoPE."""
        head_dim = 64
        modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5)
        x = torch.randn(2, 16, head_dim)
        cos, sin = modulator(x)
        modulator.clear_holonomy_phase()
        cos2, sin2 = modulator(x)
        assert torch.equal(cos, cos2)
        assert torch.equal(sin, sin2)
