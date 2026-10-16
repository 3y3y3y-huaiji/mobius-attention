# -*- coding: utf-8 -*-
r"""
Continuous Flow & Geometric Invariants Test Suite
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Tests four foundational continuous geometric properties:
1. Continuous 2D Planar Rotation Euclidean Norm Preservation:
   ||R(\tilde{\theta}) v|| == ||v||  \forall \tilde{\theta} \in \mathbb{R}
2. Operator-level and Vector-level Lipschitz Continuity of Rotary Operator:
   ||\tilde{\mathbf{R}}(t+1) - \tilde{\mathbf{R}}(t)||_2 \le 2\sin\left(\frac{\theta_m + \omega_{\max}}{2}\right)
3. Smooth S-Curve Cumulative Phase Integration & Destructive Interference:
   \int \omega(t) dt = \pi,  |\omega(t)| \le \omega_{\max} \ll \pi
   Continuous gradual phase advection without step impulses
   Exponential relative odds ratio attenuation: \frac{\mathrm{Odds}_{\mathrm{mob}}}{\mathrm{Odds}_{\mathrm{std}}} \equiv \exp(-2 \text{Logit})
4. Non-Negative Feature Map & Baseline Similarity Preservation:
   \phi(u) = \text{ELU}(u) + 1 \ge 0 \implies \langle \phi(q), \phi(k) \rangle \ge 0
   Prevention of spurious false-positive reversal alignments
"""

import math
import random
import pytest
import numpy as np
import torch
import torch.nn.functional as F

from src.mobius_attention import (
    ContinuousMobiusRoPEModulator,
    SemanticGaugeConnection,
    causal_viscous_ema,
    nonnegative_feature_map,
    apply_rotary_angle_addition
)
from tests.common_math import (
    apply_rotary_angle_addition_np,
    nonnegative_feature_map_np,
    compute_lipschitz_rotation_bound,
    compute_rope_freqs,
    apply_rotary_emb
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


# =============================================================================
# 1. Continuous Rotation Norm Preservation
# =============================================================================

class TestContinuousRotationNormPreservation:
    """Verifies that 2D SO(2) angle addition identically preserves Euclidean vector norms."""

    def test_norm_preservation_torch_float64(self):
        """||R(theta + phi) v|| == ||v|| across continuous phase domain in float64 (< 1e-12)."""
        torch.manual_seed(101)
        dim = 64
        num_trials = 100
        v = torch.randn(num_trials, dim, dtype=torch.float64)
        orig_norm = v.norm(dim=-1)

        # Diverse continuous phase angles in radians: small, medium, pi, multiples of pi, irrational, negative
        test_phases = [
            0.0, 1e-4, 0.05, math.pi / 6, math.pi / 4, math.pi / 2,
            math.pi, 1.5 * math.pi, 2 * math.pi, math.e, math.sqrt(2),
            -math.pi / 3, -math.pi, 10.0 * math.pi, -42.1337
        ]

        modulator = ContinuousMobiusRoPEModulator(head_dim=dim, anchor_ratio=0.0)

        for phi_val in test_phases:
            phi_tensor = torch.full((num_trials, 1, 1), phi_val, dtype=torch.float64)
            cos, sin = modulator(v.unsqueeze(1), phi=phi_tensor)
            v_rot = apply_rotary_pos_emb(v.unsqueeze(1), cos, sin).squeeze(1)
            rot_norm = v_rot.norm(dim=-1)

            max_err = (orig_norm - rot_norm).abs().max().item()
            assert max_err < 1e-12, f"Norm preservation violated at phi={phi_val}: max_err={max_err:.2e}"

    def test_norm_preservation_torch_float32(self):
        """||R(theta + phi) v|| == ||v|| across continuous phase domain in float32 (< 1e-6)."""
        torch.manual_seed(102)
        dim = 128
        num_trials = 50
        v = torch.randn(num_trials, dim, dtype=torch.float32)
        orig_norm = v.norm(dim=-1)

        test_phases = [0.0, 0.05, math.pi / 2, math.pi, 2 * math.pi, -math.pi, 7.5]
        modulator = ContinuousMobiusRoPEModulator(head_dim=dim, anchor_ratio=0.0)

        for phi_val in test_phases:
            phi_tensor = torch.full((num_trials, 1, 1), phi_val, dtype=torch.float32)
            cos, sin = modulator(v.unsqueeze(1), phi=phi_tensor)
            v_rot = apply_rotary_pos_emb(v.unsqueeze(1), cos, sin).squeeze(1)
            rot_norm = v_rot.norm(dim=-1)

            max_err = (orig_norm - rot_norm).abs().max().item()
            assert max_err < 1e-6, f"Norm preservation violated at phi={phi_val}: max_err={max_err:.2e}"

    def test_norm_preservation_numpy_reference(self):
        """NumPy reference angle addition preserves vector norms identically (< 1e-12)."""
        np.random.seed(103)
        dim = 64
        seq_len = 20
        v = np.random.randn(seq_len, dim).astype(np.float64)
        orig_norm = np.linalg.norm(v, axis=-1)

        cos0, sin0 = compute_rope_freqs(dim, seq_len)
        phi_continuous = np.linspace(-3.0 * np.pi, 3.0 * np.pi, seq_len)
        phi_cat = np.concatenate([phi_continuous[:, None], phi_continuous[:, None]], axis=-1)
        phi_full = np.repeat(phi_cat, dim // 2, axis=-1)[..., :dim]

        cos_mod, sin_mod = apply_rotary_angle_addition_np(cos0, sin0, phi_full)
        v_rot = apply_rotary_emb(v, cos_mod, sin_mod)
        rot_norm = np.linalg.norm(v_rot, axis=-1)

        max_err = np.max(np.abs(orig_norm - rot_norm))
        assert max_err < 1e-12, f"NumPy norm preservation error: {max_err:.2e}"


# =============================================================================
# 2. Lipschitz Continuity of Rotation Operator
# =============================================================================

class TestLipschitzContinuity:
    r"""
    Verifies that the adjacent-token rotation operator difference satisfies:
    ||\tilde{\mathbf{R}}(t+1) - \tilde{\mathbf{R}}(t)||_2 \le 2\sin\left(\frac{\theta_m + \omega_{\max}}{2}\right)
    """

    def test_operator_norm_lipschitz_bound_randomized_trials(self):
        """1,000 randomized trials testing operator spectral norm <= theoretical Lipschitz bound."""
        random.seed(2026)
        np.random.seed(2026)
        omega_max = 0.05
        num_trials = 1000

        max_observed_ratio = 0.0

        for _ in range(num_trials):
            t = random.randint(0, 50000)
            # Base RoPE frequency theta_m in (0, pi]
            theta_m = random.uniform(1e-4, 1.2)
            # Continuous torsion rate in [-omega_max, omega_max]
            omega_t1 = random.uniform(-omega_max, omega_max)

            phi_t = random.uniform(-20.0, 20.0)
            phi_t1 = phi_t + omega_t1

            angle_t = t * theta_m + phi_t
            angle_t1 = (t + 1) * theta_m + phi_t1

            # 2D planar rotation matrices:
            R_t = np.array([
                [math.cos(angle_t), -math.sin(angle_t)],
                [math.sin(angle_t),  math.cos(angle_t)]
            ], dtype=np.float64)

            R_t1 = np.array([
                [math.cos(angle_t1), -math.sin(angle_t1)],
                [math.sin(angle_t1),  math.cos(angle_t1)]
            ], dtype=np.float64)

            # Operator spectral norm (L2 matrix norm)
            delta_R = R_t1 - R_t
            spectral_norm = np.linalg.norm(delta_R, ord=2)

            theoretical_bound = compute_lipschitz_rotation_bound(theta_m, omega_max)
            assert spectral_norm <= theoretical_bound + 1e-12, (
                f"Lipschitz bound violated: ||Delta R||_2={spectral_norm:.6f} > bound={theoretical_bound:.6f}"
            )

            ratio = spectral_norm / (theoretical_bound + 1e-15)
            if ratio > max_observed_ratio:
                max_observed_ratio = ratio

        assert max_observed_ratio <= 1.0 + 1e-12
        assert max_observed_ratio > 0.95, f"Peak ratio {max_observed_ratio} should approach bound closely"

    def test_vector_level_lipschitz_bound_torch(self):
        """||R(t+1) v - R(t) v||_2 <= 2 sin((theta_m + omega_max)/2) ||v||_2."""
        torch.manual_seed(2027)
        num_samples = 500
        omega_max = 0.05
        theta_m = 0.1

        v = torch.randn(num_samples, 2, dtype=torch.float64)
        v_norm = v.norm(dim=-1)

        t = 100
        omega = torch.empty(num_samples, 1, dtype=torch.float64).uniform_(-omega_max, omega_max)
        phi_t = torch.randn(num_samples, 1, dtype=torch.float64)
        phi_t1 = phi_t + omega

        th_t = t * theta_m + phi_t
        th_t1 = (t + 1) * theta_m + phi_t1

        # Rotate 2D vectors
        c_t, s_t = th_t.cos(), th_t.sin()
        c_t1, s_t1 = th_t1.cos(), th_t1.sin()

        v_rot_t = torch.cat([v[:, :1] * c_t - v[:, 1:] * s_t, v[:, :1] * s_t + v[:, 1:] * c_t], dim=-1)
        v_rot_t1 = torch.cat([v[:, :1] * c_t1 - v[:, 1:] * s_t1, v[:, :1] * s_t1 + v[:, 1:] * c_t1], dim=-1)

        diff_norm = (v_rot_t1 - v_rot_t).norm(dim=-1)
        lip_bound = 2.0 * math.sin((theta_m + omega_max) / 2.0) * v_norm

        assert (diff_norm <= lip_bound + 1e-12).all(), "Vector-level Lipschitz continuity violated"

    def test_viscous_ema_guarantees_bounded_torsion_rate(self):
        """1D Viscous EMA guarantees bounded derivative: |omega(t) - omega(t-1)| <= 2 nu omega_max."""
        torch.manual_seed(2028)
        seq_len = 1000
        nu = 0.35
        omega_max = 0.05

        # Worst-case adversarial alternating inputs: [-omega_max, +omega_max, -omega_max, ...]
        raw_g = torch.full((1, seq_len, 1), omega_max)
        raw_g[:, 1::2, :] = -omega_max

        omega = causal_viscous_ema(raw_g, nu=nu)
        diff = (omega[:, 1:, :] - omega[:, :-1, :]).abs()

        theoretical_max_jump = 2.0 * nu * omega_max
        max_jump = diff.max().item()

        assert max_jump <= theoretical_max_jump + 1e-6, (
            f"Viscous EMA jump {max_jump} exceeded theoretical bound {theoretical_max_jump}"
        )


# =============================================================================
# 3. Smooth S-Curve Cumulative Phase Integration & Destructive Interference
# =============================================================================

class TestSmoothSCurvePhaseIntegration:
    r"""
    Verifies smooth continuous phase advection:
    \int \omega(t) dt = \pi,  |\omega(t)| \le \omega_{\max} \ll \pi.
    Destructive interference at \Phi = \pi: \cos(\pi) = -1.0.
    Relative odds ratio attenuation: Odds_mob / Odds_std = exp(-2 * Logit).
    """

    def test_smooth_s_curve_integration_to_pi(self):
        """Constructs a smooth bell-shaped continuous torsion profile that integrates exactly to pi."""
        seq_len = 128
        t = np.arange(seq_len, dtype=np.float64)
        t_center = 64.0
        sigma = 10.0

        # Gaussian / logistic derivative torsion profile: omega(t) = (pi / (sigma * sqrt(2pi))) * exp(- (t - t_center)^2 / (2 sigma^2))
        raw_bell = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
        # Normalize area strictly to pi: sum omega(t) dt = pi
        omega_curve = raw_bell * (math.pi / np.sum(raw_bell))

        # Check maximum continuous torsion rate is bounded and small (no discrete jumps)
        peak_omega = np.max(np.abs(omega_curve))
        assert peak_omega < 0.15, f"Peak continuous torsion rate {peak_omega:.4f} must be small (<< pi)"
        assert peak_omega > 0.05

        # Check cumulative phase integral
        phi_curve = np.cumsum(omega_curve)
        assert math.isclose(phi_curve[0], omega_curve[0], abs_tol=1e-5)
        assert math.isclose(phi_curve[-1], math.pi, abs_tol=1e-12), (
            f"Cumulative integral {phi_curve[-1]} must equal pi exactly"
        )

        # Check S-curve smoothness: finite differences (first and second derivatives) are bounded and smooth
        d1 = np.diff(phi_curve)
        d2 = np.diff(d1)
        assert np.max(np.abs(d2)) < 0.01, "Second derivative of phase flow is bounded (no abrupt kinks)"

    def test_antipodal_destructive_interference_at_pi(self):
        """At cumulative phase Phi = pi, relative attention factor is cos(pi) = -1.0 identically."""
        torch.manual_seed(301)
        dim = 32
        modulator = ContinuousMobiusRoPEModulator(head_dim=dim, anchor_ratio=0.0)

        # Initial prompt token at t=0 with phi=0
        q = torch.randn(1, 1, dim, dtype=torch.float64)
        k = q.clone()

        cos_0, sin_0 = modulator(q, position_ids=torch.tensor([[0]]), phi=torch.tensor([[[0.0]]]))
        q_rot_0 = apply_rotary_pos_emb(q, cos_0, sin_0)
        k_rot_0 = apply_rotary_pos_emb(k, cos_0, sin_0)
        logit_0 = (q_rot_0 * k_rot_0).sum().item()

        # Target token at t=10 with antipodal phase phi = pi
        cos_pi, sin_pi = modulator(k, position_ids=torch.tensor([[0]]), phi=torch.tensor([[[math.pi]]]))
        k_rot_pi = apply_rotary_pos_emb(k, cos_pi, sin_pi)
        logit_pi = (q_rot_0 * k_rot_pi).sum().item()

        # Inversion parity: logit_pi == -logit_0
        assert math.isclose(logit_pi, -logit_0, rel_tol=1e-12), (
            f"Destructive interference failed: logit_0={logit_0}, logit_pi={logit_pi}"
        )

    def test_exponential_relative_odds_ratio_attenuation(self):
        r"""
        Relative Odds Ratio Attenuation Theorem:
        \frac{\mathrm{Odds}_{\mathrm{mob}}}{\mathrm{Odds}_{\mathrm{std}}} \equiv \exp\left(-2 \frac{\langle q_i, k_j \rangle}{\sqrt{d}}\right)
        """
        for base_logit in [0.5, 1.0, 2.0, 3.5, 5.0]:
            background_partition = 15.0

            p_std = math.exp(base_logit) / (math.exp(base_logit) + background_partition)
            p_mob = math.exp(-base_logit) / (math.exp(-base_logit) + background_partition)

            odds_std = p_std / (1.0 - p_std)
            odds_mob = p_mob / (1.0 - p_mob)

            odds_ratio = odds_mob / odds_std
            expected_odds_ratio = math.exp(-2.0 * base_logit)

            assert math.isclose(odds_ratio, expected_odds_ratio, rel_tol=1e-10), (
                f"Relative odds ratio {odds_ratio} differs from theoretical {expected_odds_ratio}"
            )


# =============================================================================
# 4. Non-Negative Feature Map & Baseline Similarity Preservation
# =============================================================================

class TestNonNegativeFeatureMap:
    r"""
    Verifies that the feature map \phi(u) = \text{ELU}(u) + 1 satisfies:
    1. \phi(u) \ge 0  \forall u \in \mathbb{R}
    2. \langle \phi(q), \phi(k) \rangle \ge 0  \forall q, k \in \mathbb{R}^d
    3. Prevents spurious false-positive reversals between opposite-polarity tokens
    """

    def test_nonnegative_feature_map_range(self):
        """phi(u) = ELU(u) + 1 >= 0 for extreme values [-1000, 1000]."""
        u = torch.linspace(-1000.0, 1000.0, 10000, dtype=torch.float64)
        phi_u = nonnegative_feature_map(u)

        assert (phi_u >= 0.0).all(), "phi(u) must be non-negative everywhere"
        # For u = 0, phi(0) = ELU(0) + 1 = 1.0
        assert math.isclose(nonnegative_feature_map(torch.tensor(0.0)).item(), 1.0, abs_tol=1e-7)
        # For u -> -inf, phi(u) -> 0.0
        assert nonnegative_feature_map(torch.tensor(-100.0)).item() >= 0.0

    def test_baseline_inner_product_positive_definiteness(self):
        """Baseline similarity <phi(q), phi(k)> is strictly non-negative for ANY arbitrary vectors."""
        torch.manual_seed(401)
        num_pairs = 1000
        dim = 64
        q = torch.randn(num_samples := num_pairs, dim, dtype=torch.float64) * 5.0
        k = torch.randn(num_pairs, dim, dtype=torch.float64) * 5.0

        q_mapped = nonnegative_feature_map(q)
        k_mapped = nonnegative_feature_map(k)

        inner_products = (q_mapped * k_mapped).sum(dim=-1)
        assert (inner_products >= 0.0).all(), "Inner products of non-negative representations must be >= 0"

    def test_prevention_of_spurious_false_positive_reversals(self):
        """
        Adversarial demonstration:
        Without feature map, two negated coordinates (-x) * (-y) = +xy produce
        spurious high attention between contradictory concepts.
        With non-negative feature map phi(u), spurious reversal is eliminated.
        """
        torch.manual_seed(402)
        # Construct two negatively correlated concept vectors
        q_raw = torch.tensor([[-2.0, -3.0, -4.0, -5.0]])
        k_raw = torch.tensor([[-2.0, -3.0, -4.0, -5.0]])

        # Without feature map:
        raw_dot = (q_raw * k_raw).sum().item()
        assert raw_dot == 54.0, "Without feature map, negative * negative = positive 54.0"

        # Opposite sign vector
        k_opposite = -q_raw
        raw_opposite_dot = (q_raw * k_opposite).sum().item()
        assert raw_opposite_dot == -54.0

        # With non-negative feature map:
        q_phi = nonnegative_feature_map(q_raw)
        k_phi = nonnegative_feature_map(k_raw)
        k_opp_phi = nonnegative_feature_map(k_opposite)

        dot_aligned = (q_phi * k_phi).sum().item()
        dot_opp = (q_phi * k_opp_phi).sum().item()

        # Both inner products are strictly >= 0, baseline similarity is well-defined
        assert dot_aligned >= 0.0
        assert dot_opp >= 0.0

    def test_integration_in_continuous_mobius_modulator(self):
        """Verifies apply_feature_map respects subspace decoupling."""
        torch.manual_seed(403)
        dim = 64
        num_subspaces = dim // 2
        anchor_ratio = 0.5
        num_anchor = int(round(num_subspaces * anchor_ratio))

        modulator = ContinuousMobiusRoPEModulator(
            head_dim=dim,
            anchor_ratio=anchor_ratio,
            enable_nonnegative_map=True
        )

        q = torch.randn(2, 10, dim)
        k = torch.randn(2, 10, dim)

        q_mapped, k_mapped = modulator.apply_feature_map(q, k)

        # Anchor channels (first num_anchor of each half) must remain untouched
        assert torch.equal(q_mapped[..., :num_anchor], q[..., :num_anchor])
        assert torch.equal(q_mapped[..., num_subspaces:num_subspaces + num_anchor],
                           q[..., num_subspaces:num_subspaces + num_anchor])

        # Evolution channels must be strictly >= 0
        assert (q_mapped[..., num_anchor:num_subspaces] >= 0.0).all()
        assert (k_mapped[..., num_anchor:num_subspaces] >= 0.0).all()
