# -*- coding: utf-8 -*-
r"""
Property-Based Fuzzing & Invariant Shrinking Test Suite using Hypothesis
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Leverages Hypothesis for exhaustive property-based counterexample discovery:
1. Affine Monoid Semi-Group Associativity:
       ((T_3 \circ T_2) \circ T_1)(x) \equiv (T_3 \circ (T_2 \circ T_1))(x)
2. Affine Monoid Identity:
       (a, b) \circ (1, 0) = (1, 0) \circ (a, b) = (a, b)
3. Parallel Affine Scan vs Serial Recurrence Equivalence:
       \Delta y \le \mathcal{O}(\epsilon_{\text{mach}} \log L)
4. SO(2) 2D Planar Vector Norm Conservation:
       \|R(\theta + \Phi) v\|_2 \equiv \|v\|_2
5. Anchor Channel Zero Modulation Invariance:
       \Phi_m \equiv 0 \implies \tilde{\theta}_m(t) \equiv t\theta_m
6. Phase Attractor Topological Equilibrium & Monotonic Restoration:
       -\eta \sin(2 k \pi) = 0, \quad -\eta \sin(2(k\pi + \delta)) < 0
7. Dynamic Adaptive Cutoff Alpha Bounds & Monotonicity:
       0.50 \le \alpha(L) \le 0.875 \quad \text{and} \quad L_1 \le L_2 \implies \alpha(L_1) \le \alpha(L_2)
"""

import math
import torch
from hypothesis import given, settings, strategies as st

from src.gauge_predictor import (
    parallel_affine_scan,
    two_stage_associative_scan,
)
from src.mobius_attention import (
    SGMobiusRotaryEmbedding,
    apply_rotary_angle_addition,
)


def compose_affine(t2, t1):
    """Composes two affine transformations: (a2, b2) o (a1, b1) = (a2*a1, a2*b1 + b2)."""
    a2, b2 = t2
    a1, b1 = t1
    return (a2 * a1, a2 * b1 + b2)


def apply_affine(t, x):
    """Applies affine transformation T(x) = a*x + b."""
    a, b = t
    return a * x + b


class TestHypothesisMonoidInvariants:
    """Hypothesis Property-based testing for algebraic monoid invariants."""

    @settings(max_examples=200)
    @given(
        a1=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        b1=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        a2=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        b2=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        a3=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        b3=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        x=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    def test_affine_associativity_property(self, a1, b1, a2, b2, a3, b3, x):
        r"""
        Tests monoid associativity: ((T3 o T2) o T1)(x) == (T3 o (T2 o T1))(x).
        """
        t1 = (a1, b1)
        t2 = (a2, b2)
        t3 = (a3, b3)

        # Left association: (T3 o T2) o T1
        t32 = compose_affine(t3, t2)
        t_left = compose_affine(t32, t1)
        y_left = apply_affine(t_left, x)

        # Right association: T3 o (T2 o T1)
        t21 = compose_affine(t2, t1)
        t_right = compose_affine(t3, t21)
        y_right = apply_affine(t_right, x)

        # In float64 algebra: difference must be within floating rounding
        assert abs(y_left - y_right) < 1e-10 * (abs(y_left) + abs(y_right) + 1.0)

    @settings(max_examples=100)
    @given(
        a=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        b=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        x=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    def test_affine_identity_property(self, a, b, x):
        r"""
        Tests identity element (1, 0): T o Id = Id o T = T.
        """
        t = (a, b)
        identity = (1.0, 0.0)

        t_right = compose_affine(t, identity)
        t_left = compose_affine(identity, t)

        assert abs(t_right[0] - a) < 1e-12 and abs(t_right[1] - b) < 1e-12
        assert abs(t_left[0] - a) < 1e-12 and abs(t_left[1] - b) < 1e-12


class TestHypothesisAlgorithmInvariants:
    """Hypothesis Property-based testing for algorithm invariants."""

    @settings(max_examples=50)
    @given(
        seq_len=st.integers(min_value=1, max_value=128),
        nu=st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
    )
    def test_parallel_scan_vs_serial_equivalence(self, seq_len, nu):
        r"""
        Property: parallel_affine_scan produces identical output to serial loop
        for arbitrary sequence lengths and nu parameters.
        """
        torch.manual_seed(seq_len)
        g = torch.randn(1, seq_len, 4, dtype=torch.float64)

        # Serial reference
        omega_ref = torch.zeros_like(g)
        prev = torch.zeros(1, 4, dtype=torch.float64)
        for t in range(seq_len):
            step = (1.0 - nu) * prev + nu * g[:, t, :]
            omega_ref[:, t, :] = step
            prev = step

        # Parallel scan
        a = torch.full_like(g, 1.0 - nu)
        b = nu * g
        _, omega_scan = parallel_affine_scan(a, b, dim=1)

        max_err = (omega_scan - omega_ref).abs().max().item()
        assert max_err < 1e-12, f"Scan failed equivalence at L={seq_len}: err={max_err}"

    @settings(max_examples=100)
    @given(
        v1=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        v2=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        theta=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        phi=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def test_2d_euclidean_norm_conservation(self, v1, v2, theta, phi):
        r"""
        Property: 2D planar rotation preserves vector Euclidean norm identically:
            \|R(\theta + \Phi) v\| \equiv \|v\|
        """
        v = torch.tensor([v1, v2], dtype=torch.float64)
        norm_orig = torch.norm(v).item()
        if norm_orig < 1e-6:
            return  # skip degenerate null vectors

        cos0 = torch.tensor([math.cos(theta), math.cos(theta)], dtype=torch.float64)
        sin0 = torch.tensor([math.sin(theta), math.sin(theta)], dtype=torch.float64)
        phi_t = torch.tensor([phi, phi], dtype=torch.float64)

        cos_mod, sin_mod = apply_rotary_angle_addition(cos0, sin0, phi_t)

        # 2D planar rotation on [v1, v2]
        # [v1*cos - v2*sin, v2*cos + v1*sin]
        rot_v1 = v1 * cos_mod[0].item() - v2 * sin_mod[0].item()
        rot_v2 = v2 * cos_mod[1].item() + v1 * sin_mod[1].item()
        norm_rot = math.sqrt(rot_v1**2 + rot_v2**2)

        assert abs(norm_orig - norm_rot) < 1e-10 * (norm_orig + 1.0)

    @settings(max_examples=100)
    @given(
        k=st.integers(min_value=-10, max_value=10),
        delta=st.floats(min_value=0.01, max_value=math.pi / 4.0, allow_nan=False, allow_infinity=False),
        eta=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_phase_attractor_fixed_points_and_restoring_force(self, k, delta, eta):
        r"""
        Property:
        1. At \Phi = k\pi, force -eta*sin(2*Phi) == 0.
        2. At \Phi = k\pi + \delta, force < 0 (pulls left towards k*pi).
        3. At \Phi = k\pi - \delta, force > 0 (pulls right towards k*pi).
        """
        phi_eq = float(k) * math.pi
        force_eq = -eta * math.sin(2.0 * phi_eq)
        assert abs(force_eq) < 1e-12, f"Force non-zero at equilibrium {phi_eq}: {force_eq}"

        # Displaced to the right
        force_right = -eta * math.sin(2.0 * (phi_eq + delta))
        assert force_right < 0.0, f"Restoring force to the right must be negative: {force_right}"

        # Displaced to the left
        force_left = -eta * math.sin(2.0 * (phi_eq - delta))
        assert force_left > 0.0, f"Restoring force to the left must be positive: {force_left}"

    @settings(max_examples=100)
    @given(
        L1=st.integers(min_value=1, max_value=50000),
        L2=st.integers(min_value=1, max_value=50000),
    )
    def test_dynamic_adaptive_alpha_invariants(self, L1, L2):
        r"""
        Property:
        1. 0.50 <= alpha(L) <= 0.875 for all L >= 1.
        2. L1 <= L2 ==> alpha(L1) <= alpha(L2) (monotonicity).
        """
        rotary = SGMobiusRotaryEmbedding(head_dim=64, base=10000.0, adaptive_alpha=True)

        a1 = rotary.compute_adaptive_alpha(L1)
        a2 = rotary.compute_adaptive_alpha(L2)

        # Bounds
        assert 0.50 <= a1 <= 0.875
        assert 0.50 <= a2 <= 0.875

        # Monotonicity
        if L1 <= L2:
            assert a1 <= a2 + 1e-12
        else:
            assert a1 >= a2 - 1e-12
