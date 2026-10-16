# -*- coding: utf-8 -*-
r"""
Phase 4 Empirical Challenger: Adversarial Stress Test Suite for Mathematical Invariants
Author: challenger_p4_math_adversarial
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Rigorously verifies and stress-tests:
1. Extreme angles & random phase flows:
   - Antipodal cancellation at pi (Delta Phi = pi + 2k*pi)
   - Cyclic restoration at 2*pi (Delta Phi = 2k*pi)
   - Norm invariance and relative rotation across 10,000 extreme randomized angles
2. Operator Lipschitz continuity under attractor -eta*sin(2*Phi):
   - Spectral norm difference ||R~(t+1) - R~(t)||_2
   - Verification of 2*sin((theta_m + omega_max)/2) bound under clamped and unclamped flows
   - Attractor Lyapunov stability and phase self-locking at {0, pi, 2pi}
3. Numerical exactness of Theorem 3 pairwise odds ratio attenuation:
   - Odds_mob(k:j) / Odds_std(k:j) == exp(2 * A_ij^mob)
   - Unconditional partition cancellation across background token sizes L up to 32,768
   - Arbitrary background logit regimes (uniform, peaked, heavy-tailed, high dynamic range)
4. Prefill vs step-by-step autoregressive decode equivalence:
   - Across context lengths L in [16, 64, 512, 2048, 8192]
   - Under non-zero eta and random subspace masks
   - Evaluates omega, phi, cos/sin, Q/K, attention logits, and Softmax probabilities (< 1e-10 in float64)
"""

import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Reconfigure stdout/stderr to utf-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.mobius_attention import (
    SGMobiusRotaryEmbedding,
    apply_rotary_angle_addition,
    nonnegative_feature_map,
)
from src.gauge_predictor import (
    SemanticGaugeConnection,
    causal_viscous_ema,
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


# ==============================================================================
# SUITE 1: Extreme Angles and Random Phase Flows
# ==============================================================================

def stress_test_extreme_angles_and_phase_flows(num_trials: int = 10000) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  [SUITE 1] EXTREME ANGLES & RANDOM PHASE FLOWS STRESS TEST")
    print("=" * 80)
    
    torch.manual_seed(1001)
    np.random.seed(1001)
    random.seed(1001)

    max_antipodal_err = 0.0
    max_cyclic_err = 0.0
    max_norm_err = 0.0
    max_relative_rot_err = 0.0
    max_subspace_cancellation_err = 0.0

    # 1. Antipodal Cancellation at pi + 2k*pi and Cyclic Restoration at 2k*pi
    head_dim = 64
    num_subspaces = head_dim // 2
    theta_base = 10000.0
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, num_subspaces, 1, dtype=torch.float64) * 2 / head_dim))

    k_multipliers = [-1000, -100, -50, -10, -3, -1, 0, 1, 2, 5, 10, 50, 100, 1000]

    for k in k_multipliers:
        # Antipodal test: delta_phi = (2k + 1) * pi
        phi_antipodal = (2 * k + 1) * math.pi
        phi_tensor = torch.tensor([[[phi_antipodal]]], dtype=torch.float64)

        q = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)
        k_vec = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)
        pos_i = torch.tensor([[10]], dtype=torch.long)
        pos_j = torch.tensor([[5]], dtype=torch.long)

        # Baseline RoPE (phi = 0)
        mod_base = SGMobiusRotaryEmbedding(head_dim=head_dim, anchor_ratio=0.0, pre_fuse_angles=True)
        cos_i0, sin_i0 = mod_base(q, position_ids=pos_i, phi=None)
        cos_j0, sin_j0 = mod_base(k_vec, position_ids=pos_j, phi=None)
        q_rot0 = apply_rotary_pos_emb(q, cos_i0, sin_i0)
        k_rot0 = apply_rotary_pos_emb(k_vec, cos_j0, sin_j0)
        dot_baseline = torch.sum(q_rot0 * k_rot0).item()

        # Modulated with antipodal twist on query
        cos_i_anti, sin_i_anti = mod_base(q, position_ids=pos_i, phi=phi_tensor)
        q_rot_anti = apply_rotary_pos_emb(q, cos_i_anti, sin_i_anti)
        dot_anti = torch.sum(q_rot_anti * k_rot0).item()

        # Check: dot_anti == -dot_baseline
        anti_err = abs(dot_anti - (-dot_baseline))
        max_antipodal_err = max(max_antipodal_err, anti_err)
        assert anti_err < 1e-10, f"Antipodal cancellation violated at k={k}: err={anti_err:.2e}"

        # Cyclic restoration test: delta_phi = 2k * pi
        phi_cyclic = 2 * k * math.pi
        phi_tensor_cyc = torch.tensor([[[phi_cyclic]]], dtype=torch.float64)
        cos_i_cyc, sin_i_cyc = mod_base(q, position_ids=pos_i, phi=phi_tensor_cyc)
        q_rot_cyc = apply_rotary_pos_emb(q, cos_i_cyc, sin_i_cyc)
        dot_cyc = torch.sum(q_rot_cyc * k_rot0).item()

        # Check: dot_cyc == dot_baseline
        cyc_err = abs(dot_cyc - dot_baseline)
        max_cyclic_err = max(max_cyclic_err, cyc_err)
        assert cyc_err < 1e-10, f"Cyclic restoration violated at k={k}: err={cyc_err:.2e}"

    print(f"  [✓] Antipodal Cancellation at π + 2kπ: Max Error = {max_antipodal_err:.2e} across all k ∈ {k_multipliers}")
    print(f"  [✓] Cyclic Restoration at 2kπ: Max Error = {max_cyclic_err:.2e} across all k ∈ {k_multipliers}")

    # 2. Subspace-Decoupled Cancellation: Anchor vs Evolution Subspaces
    for trial in range(1000):
        # Random subspace mask
        rand_mask = torch.randint(0, 2, (num_subspaces,), dtype=torch.bool)
        if rand_mask.sum() == 0 or rand_mask.sum() == num_subspaces:
            rand_mask[0] = False
            rand_mask[1] = True

        mod_subspace = SGMobiusRotaryEmbedding(head_dim=head_dim, subspace_mask=rand_mask, pre_fuse_angles=True)
        q = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)
        k_vec = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)
        pos = torch.tensor([[trial]], dtype=torch.long)

        # Baseline RoPE
        cos_base, sin_base = mod_subspace(q, position_ids=pos, phi=None)
        q_base = apply_rotary_pos_emb(q, cos_base, sin_base)
        k_base = apply_rotary_pos_emb(k_vec, cos_base, sin_base)

        # Separate anchor and mob components
        full_mask = torch.cat([rand_mask, rand_mask], dim=-1)
        dot_anchor = torch.sum(q_base[..., ~full_mask] * k_base[..., ~full_mask]).item()
        dot_mob = torch.sum(q_base[..., full_mask] * k_base[..., full_mask]).item()

        # Modulated with phi = pi on mob subspace
        phi_pi = torch.tensor([[[math.pi]]], dtype=torch.float64)
        cos_mod, sin_mod = mod_subspace(q, position_ids=pos, phi=phi_pi)
        q_mod = apply_rotary_pos_emb(q, cos_mod, sin_mod)
        dot_total_mod = torch.sum(q_mod * k_base).item()

        expected_dot = dot_anchor - dot_mob
        err_sub = abs(dot_total_mod - expected_dot)
        max_subspace_cancellation_err = max(max_subspace_cancellation_err, err_sub)
        assert err_sub < 1e-12, f"Subspace cancellation failed at trial {trial}: err={err_sub:.2e}"

    print(f"  [✓] Subspace Decoupled Cancellation: Max Error = {max_subspace_cancellation_err:.2e} across 1000 trials")

    # 3. Norm Invariance and Relative Rotation across 10,000 Randomized Angles
    # A. Standard & Extended Domain (|Phi| <= 10,000 rad, representing up to 200,000 inversions)
    max_relative_rot_err_std = 0.0
    for trial in range(num_trials):
        pos_i_val = random.randint(0, 32768)
        pos_j_val = random.randint(0, 32768)

        # Sample phase values up to 10,000 rad
        if trial % 2 == 0:
            phi_i = np.random.uniform(-10000.0, 10000.0, num_subspaces)
            phi_j = np.random.uniform(-10000.0, 10000.0, num_subspaces)
        else:
            phi_i = np.random.normal(0, 50.0, num_subspaces)
            phi_j = np.random.normal(0, 50.0, num_subspaces)

        q = np.random.randn(head_dim).astype(np.float64)
        k_vec = np.random.randn(head_dim).astype(np.float64)

        theta_i = pos_i_val * inv_freq.numpy()
        theta_j = pos_j_val * inv_freq.numpy()

        for m in range(num_subspaces):
            v = np.array([q[2*m], q[2*m+1]], dtype=np.float64)
            orig_norm = np.linalg.norm(v)
            angle = theta_i[m] + phi_i[m]
            c, s = np.cos(angle), np.sin(angle)
            v_rot = np.array([v[0]*c - v[1]*s, v[0]*s + v[1]*c], dtype=np.float64)
            rot_norm = np.linalg.norm(v_rot)
            norm_err = abs(orig_norm - rot_norm)
            max_norm_err = max(max_norm_err, norm_err)

            # Check Relative Rotation Invariance:
            w = np.array([k_vec[2*m], k_vec[2*m+1]], dtype=np.float64)
            angle_j = theta_j[m] + phi_j[m]
            c_j, s_j = np.cos(angle_j), np.sin(angle_j)
            w_rot = np.array([w[0]*c_j - w[1]*s_j, w[0]*s_j + w[1]*c_j], dtype=np.float64)
            dot_lhs = np.dot(v_rot, w_rot)

            delta_angle = angle_j - angle
            c_rel, s_rel = np.cos(delta_angle), np.sin(delta_angle)
            w_rel = np.array([w[0]*c_rel - w[1]*s_rel, w[0]*s_rel + w[1]*c_rel], dtype=np.float64)
            dot_rhs = np.dot(v, w_rel)

            rel_err = abs(dot_lhs - dot_rhs)
            max_relative_rot_err_std = max(max_relative_rot_err_std, rel_err)

    assert max_norm_err < 1e-12, f"Norm invariance violated: {max_norm_err:.2e}"
    assert max_relative_rot_err_std < 1e-10, f"Standard/extended relative rotation error too high: {max_relative_rot_err_std:.2e}"

    # B. Ultra-Extreme Stress Domain (|Phi| ~ 1,000,000 rad) to characterize float64 ULP argument reduction loss
    max_relative_rot_err_ultra = 0.0
    for _ in range(2000):
        pos_i_val = random.randint(0, 100000)
        pos_j_val = random.randint(0, 100000)
        phi_i = np.random.uniform(-1e6, 1e6, num_subspaces)
        phi_j = np.random.uniform(-1e6, 1e6, num_subspaces)

        q = np.random.randn(head_dim).astype(np.float64)
        k_vec = np.random.randn(head_dim).astype(np.float64)

        theta_i = pos_i_val * inv_freq.numpy()
        theta_j = pos_j_val * inv_freq.numpy()

        for m in range(num_subspaces):
            v = np.array([q[2*m], q[2*m+1]], dtype=np.float64)
            w = np.array([k_vec[2*m], k_vec[2*m+1]], dtype=np.float64)
            ang_i = theta_i[m] + phi_i[m]
            ang_j = theta_j[m] + phi_j[m]

            v_rot = np.array([v[0]*np.cos(ang_i) - v[1]*np.sin(ang_i), v[0]*np.sin(ang_i) + v[1]*np.cos(ang_i)])
            w_rot = np.array([w[0]*np.cos(ang_j) - w[1]*np.sin(ang_j), w[0]*np.sin(ang_j) + w[1]*np.cos(ang_j)])
            dot_lhs = np.dot(v_rot, w_rot)

            delta = ang_j - ang_i
            w_rel = np.array([w[0]*np.cos(delta) - w[1]*np.sin(delta), w[0]*np.sin(delta) + w[1]*np.cos(delta)])
            dot_rhs = np.dot(v, w_rel)

            max_relative_rot_err_ultra = max(max_relative_rot_err_ultra, abs(dot_lhs - dot_rhs))

    assert max_relative_rot_err_ultra < 2e-9, f"Ultra-extreme relative rotation error too high: {max_relative_rot_err_ultra:.2e}"

    print(f"  [✓] 2D Euclidean Norm Preservation: Max Error = {max_norm_err:.2e} across {num_trials} trials")
    print(f"  [✓] Relative Rotation Invariance (|Φ| ≤ 10,000 rad): Max Error = {max_relative_rot_err_std:.2e} (< 1e-10 threshold)")
    print(f"  [✓] Ultra-Extreme Scale Stress (|Φ| ≤ 1,000,000 rad): Max Error = {max_relative_rot_err_ultra:.2e} (< 2e-9, linear ULP scaling)")

    return {
        "max_antipodal_err": max_antipodal_err,
        "max_cyclic_err": max_cyclic_err,
        "max_subspace_cancellation_err": max_subspace_cancellation_err,
        "max_norm_err": max_norm_err,
        "max_relative_rot_err_std": max_relative_rot_err_std,
        "max_relative_rot_err_ultra": max_relative_rot_err_ultra,
    }


# ==============================================================================
# SUITE 2: Operator Lipschitz Continuity under Attractor -eta*sin(2*Phi)
# ==============================================================================

def stress_test_lipschitz_continuity_under_attractor(num_trials: int = 10000) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  [SUITE 2] OPERATOR LIPSCHITZ CONTINUITY & ATTRACTOR DYNAMICS STRESS TEST")
    print("=" * 80)
    
    np.random.seed(2002)
    random.seed(2002)
    torch.manual_seed(2002)

    max_clamped_bound_ratio = 0.0
    violations_clamped = 0
    max_spectral_norm_diff = 0.0

    # 1. Clamped Flow under Varying Attractor Strengths eta in [0.0, 100.0]
    eta_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    omega_max = 0.35  # rad/token

    for eta in eta_values:
        for _ in range(num_trials // len(eta_values)):
            theta_m = random.uniform(1e-5, math.pi - omega_max)  # ensures theta_m + omega_max <= pi
            prev_phi = random.uniform(-100.0, 100.0)
            prev_omega = random.uniform(-omega_max, omega_max)
            raw_g = random.uniform(-omega_max, omega_max)
            nu = random.uniform(0.01, 1.0)

            # Compute next step omega with attractor
            attractor = -eta * math.sin(2.0 * prev_phi)
            step_omega = (1.0 - nu) * prev_omega + nu * raw_g + attractor
            # Clamping to [-omega_max, omega_max]
            clamped_omega = max(-omega_max, min(omega_max, step_omega))

            t = random.randint(0, 10000)
            ang_t = t * theta_m + prev_phi
            ang_t1 = (t + 1) * theta_m + (prev_phi + clamped_omega)

            # Construct 2D rotation matrices
            R_t = np.array([
                [np.cos(ang_t), -np.sin(ang_t)],
                [np.sin(ang_t),  np.cos(ang_t)]
            ], dtype=np.float64)

            R_t1 = np.array([
                [np.cos(ang_t1), -np.sin(ang_t1)],
                [np.sin(ang_t1),  np.cos(ang_t1)]
            ], dtype=np.float64)

            diff_norm = np.linalg.norm(R_t1 - R_t, ord=2)
            theoretical_bound = 2.0 * math.sin((theta_m + omega_max) / 2.0)

            excess = diff_norm - theoretical_bound
            if excess > 1e-11:
                violations_clamped += 1

            ratio = diff_norm / theoretical_bound if theoretical_bound > 1e-12 else 0.0
            max_clamped_bound_ratio = max(max_clamped_bound_ratio, ratio)
            max_spectral_norm_diff = max(max_spectral_norm_diff, diff_norm)

    assert violations_clamped == 0, f"Found {violations_clamped} violations of Lipschitz bound under attractor!"
    print("  [✓] Clamped Flow under Attractor (-eta*sin(2Phi)):")
    print(f"      Violations = {violations_clamped} across {num_trials} trials (eta in {eta_values})")
    print(f"      Max Observed Ratio to Theoretical Bound: {max_clamped_bound_ratio:.6f} (<= 1.0)")
    print(f"      Max Spectral Norm Difference: {max_spectral_norm_diff:.6f} (<= 2.0)")

    # 2. Universal Bound: 2*sin(min(pi, theta_m + |omega|)/2) <= 2.0
    # Stress test when theta_m + omega_max > pi
    max_universal_bound_ratio = 0.0
    violations_universal = 0
    for _ in range(5000):
        theta_m = random.uniform(math.pi / 2, math.pi * 0.99)
        omega_val = random.uniform(-math.pi, math.pi)
        prev_phi = random.uniform(-10.0, 10.0)

        ang_t = prev_phi
        ang_t1 = prev_phi + theta_m + omega_val

        R_t = np.array([[np.cos(ang_t), -np.sin(ang_t)], [np.sin(ang_t), np.cos(ang_t)]], dtype=np.float64)
        R_t1 = np.array([[np.cos(ang_t1), -np.sin(ang_t1)], [np.sin(ang_t1), np.cos(ang_t1)]], dtype=np.float64)

        diff_norm = np.linalg.norm(R_t1 - R_t, ord=2)
        universal_bound = 2.0 * math.sin(min(math.pi, abs(theta_m + omega_val)) / 2.0)

        if diff_norm > universal_bound + 1e-12:
            violations_universal += 1
        ratio = diff_norm / universal_bound if universal_bound > 1e-12 else 0.0
        max_universal_bound_ratio = max(max_universal_bound_ratio, ratio)

    assert violations_universal == 0, f"Universal bound failed: {violations_universal} violations"
    print(f"  [✓] Universal Spectral Norm Bound: 2*sin(min(pi, delta)/2) strictly holds across all frequencies (max ratio: {max_universal_bound_ratio:.6f})")

    # 3. Attractor Basin Stability & Phase Self-Locking at {0, pi, 2pi}
    convergence_results = {}
    for target in [0.0, math.pi, 2.0 * math.pi]:
        # Test perturbations within basin (-pi/2, +pi/2) around target
        perturbations = [-1.4, -0.8, -0.4, -0.1, 0.1, 0.4, 0.8, 1.4]
        converged = 0
        for delta in perturbations:
            phi = target + delta
            omega = 0.0
            nu = 0.3
            eta = 0.08
            om_max = 0.15

            # Autonomous evolution without external semantic drive
            for step in range(80):
                force = -eta * math.sin(2.0 * phi)
                omega = (1.0 - nu) * omega + force
                omega = max(-om_max, min(om_max, omega))
                phi += omega

            if abs(phi - target) < 0.02:
                converged += 1

        convergence_results[target] = f"{converged}/{len(perturbations)}"
        assert converged == len(perturbations), f"Failed convergence at target {target}: {converged}/{len(perturbations)}"

    print(f"  [✓] Attractor Basin Stability & Self-Locking at {{0, pi, 2pi}}: 100% Convergence {convergence_results}")

    return {
        "max_clamped_bound_ratio": max_clamped_bound_ratio,
        "violations_clamped": violations_clamped,
        "max_universal_bound_ratio": max_universal_bound_ratio,
        "convergence_results": convergence_results,
    }


# ==============================================================================
# SUITE 3: Numerical Exactness of Theorem 3 Pairwise Odds Ratio Attenuation
# ==============================================================================

def stress_test_theorem3_odds_ratio_attenuation() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  [SUITE 3] THEOREM 3: PAIRWISE ODDS RATIO ATTENUATION & PARTITION CANCELLATION")
    print("=" * 80)
    
    np.random.seed(3003)
    random.seed(3003)

    max_rel_error_odds = 0.0
    max_partition_divergence = 0.0
    min_partition_divergence = 1e9

    # Test configurations across different context lengths L
    context_lengths = [2, 10, 64, 256, 1024, 4096, 16384, 32768]
    background_regimes = ["uniform", "peaked", "high_dynamic_range", "heavy_tailed"]

    results_table = []

    for L in context_lengths:
        for regime in background_regimes:
            # Generate logits for L tokens
            if regime == "uniform":
                bg_logits = np.random.uniform(-1.0, 1.0, L)
            elif regime == "peaked":
                bg_logits = np.random.normal(0, 1.0, L)
                bg_logits[L // 2] += 15.0  # sharp dominant token
            elif regime == "high_dynamic_range":
                bg_logits = np.random.uniform(-40.0, 40.0, L)
            elif regime == "heavy_tailed":
                bg_logits = np.random.standard_cauchy(L)
                bg_logits = np.clip(bg_logits, -50.0, 50.0)

            # Choose target index k and decoy index j
            k_idx = 0
            j_idx = min(1, L - 1)
            if k_idx == j_idx and L > 1:
                j_idx = 1

            # Subspace decomposition: A_std = A_anchor + A_mob
            # For decoy: inverted under antipodal half-twist -> A_mob_inv = A_anchor - A_mob
            a_anchor_k = float(np.random.uniform(-5.0, 5.0))
            a_mob_k = float(np.random.uniform(-5.0, 5.0))
            a_std_k = a_anchor_k + a_mob_k

            a_anchor_j = float(np.random.uniform(-5.0, 5.0))
            a_mob_j = float(np.random.uniform(0.5, 6.0))  # positive evolution logit
            a_std_j = a_anchor_j + a_mob_j
            a_mob_j_inv = a_anchor_j - a_mob_j  # inverted

            # Build full logits array under standard RoPE
            logits_std = bg_logits.copy()
            logits_std[k_idx] = a_std_k
            if L > 1:
                logits_std[j_idx] = a_std_j

            # Build full logits array under SG-Möbius
            logits_mob = bg_logits.copy()
            logits_mob[k_idx] = a_std_k  # in-phase target
            if L > 1:
                logits_mob[j_idx] = a_mob_j_inv  # antipodal decoy

            # 1. Full Softmax partition functions and probabilities
            # Max-subtraction for numerical stability
            max_std = np.max(logits_std)
            exp_std = np.exp(logits_std - max_std)
            Z_std = np.sum(exp_std)
            P_std = exp_std / Z_std

            max_mob = np.max(logits_mob)
            exp_mob = np.exp(logits_mob - max_mob)
            Z_mob = np.sum(exp_mob)
            P_mob = exp_mob / Z_mob

            if L > 1:
                # Empirical pairwise odds
                odds_std = P_std[k_idx] / P_std[j_idx]
                odds_mob = P_mob[k_idx] / P_mob[j_idx]
                empirical_ratio = odds_mob / odds_std

                # Theoretical prediction: exp(2 * A_ij^mob)
                theoretical_ratio = math.exp(2.0 * a_mob_j)

                rel_err = abs(empirical_ratio - theoretical_ratio) / theoretical_ratio
                max_rel_error_odds = max(max_rel_error_odds, rel_err)

                # Partition function ratio: Z_mob / Z_std
                # Note: unnormalized partition ratio is (sum exp(logits_mob)) / (sum exp(logits_std))
                log_Z_mob_unnorm = max_mob + math.log(Z_mob)
                log_Z_std_unnorm = max_std + math.log(Z_std)
                part_ratio = math.exp(np.clip(log_Z_mob_unnorm - log_Z_std_unnorm, -50.0, 50.0))

                max_partition_divergence = max(max_partition_divergence, part_ratio)
                min_partition_divergence = min(min_partition_divergence, part_ratio)

                assert rel_err < 1e-10, f"Theorem 3 odds ratio violated at L={L}, {regime}: err={rel_err:.2e}"

        results_table.append((L, max_rel_error_odds))

    print(f"  [✓] Pairwise Odds Ratio Attenuation Theorem 3 Verified across context lengths L ∈ {context_lengths}:")
    print(f"      Max Relative Error vs exp(2*A_ij^mob): {max_rel_error_odds:.2e} (< 1e-10 threshold)")
    print(f"      Full Partition Function Partition Divergence (Z_mob / Z_std): [{min_partition_divergence:.4e}, {max_partition_divergence:.4e}]")
    print("      Certification: Even when full partition sums diverge by orders of magnitude, partition Z cancels 100% identically!")

    return {
        "max_rel_error_odds": max_rel_error_odds,
        "max_partition_divergence": max_partition_divergence,
        "min_partition_divergence": min_partition_divergence,
        "results_table": results_table,
    }


# ==============================================================================
# SUITE 4: Prefill vs Autoregressive Decode Equivalence across L in [16, 64, 512, 2048, 8192]
# ==============================================================================

def stress_test_prefill_vs_decode_equivalence() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  [SUITE 4] PREFILL VS AUTOREGRESSIVE DECODE EQUIVALENCE (L in [16, 64, 512, 2048, 8192])")
    print("=" * 80)
    
    context_lengths = [16, 64, 512, 2048, 8192]
    head_dim = 64
    hidden_dim = 64
    num_heads = 2
    dtype = torch.float64
    omega_max = 0.05
    nu = 0.35
    eta = 0.08  # strictly non-zero eta

    detailed_metrics = []

    for L in context_lengths:
        torch.manual_seed(4004 + L)
        np.random.seed(4004 + L)
        random.seed(4004 + L)

        # 1. Random Subspace Mask (non-trivial random boolean mask)
        num_subspaces = head_dim // 2
        # Ensure at least 1 anchor and 1 mob channel
        rand_mask = torch.randint(0, 2, (num_subspaces,), dtype=torch.bool)
        if rand_mask.sum() == 0:
            rand_mask[0] = True
        elif rand_mask.sum() == num_subspaces:
            rand_mask[0] = False

        anchor_count = int((~rand_mask).sum().item())
        mob_count = int(rand_mask.sum().item())

        # Instantiate modules
        gauge = SemanticGaugeConnection(
            hidden_dim=hidden_dim,
            out_dim=1,
            omega_max=omega_max,
            nu=nu,
            eta=eta,
            normalize_input=True
        ).to(dtype=dtype)

        modulator = SGMobiusRotaryEmbedding(
            head_dim=head_dim,
            subspace_mask=rand_mask,
            omega_max=omega_max,
            nu=nu,
            eta=eta,
            enable_nonnegative_map=True,
            pre_fuse_angles=True
        )

        # Generate inputs
        batch_size = 1
        x_embed = torch.randn(batch_size, L, hidden_dim, dtype=dtype)
        q = torch.randn(batch_size, num_heads, L, head_dim, dtype=dtype)
        k = torch.randn(batch_size, num_heads, L, head_dim, dtype=dtype)

        q_map, k_map = modulator.apply_feature_map(q, k)

        t0 = time.time()

        # -------------------------------------------------------------
        # Part A: Batch Parallel Prefill
        # -------------------------------------------------------------
        omega_prefill, phi_prefill = gauge(x_embed)
        pos_prefill = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        cos_prefill, sin_prefill = modulator(q_map, position_ids=pos_prefill, phi=phi_prefill)
        q_rot_prefill = apply_rotary_pos_emb(q_map, cos_prefill, sin_prefill)
        k_rot_prefill = apply_rotary_pos_emb(k_map, cos_prefill, sin_prefill)

        scale = 1.0 / math.sqrt(head_dim)

        # -------------------------------------------------------------
        # Part B: Token-by-Token Autoregressive Decode
        # -------------------------------------------------------------
        gauge.reset_state()
        omega_decode = torch.empty_like(omega_prefill)
        phi_decode = torch.empty_like(phi_prefill)
        cos_decode = torch.empty_like(cos_prefill)
        sin_decode = torch.empty_like(sin_prefill)
        q_rot_decode = torch.empty_like(q_rot_prefill)
        k_rot_decode = torch.empty_like(k_rot_prefill)

        # Allocate KV-cache buffer
        k_cache = torch.empty(batch_size, num_heads, L, head_dim, dtype=dtype)

        # Causal attention logits verification:
        # For L <= 512: verify full L x L causal attention matrix
        # For L >= 2048: verify dense boundaries (first 64, last 64) and extensive strided check
        if L <= 512:
            check_steps = list(range(L))
        else:
            dense_head = list(range(0, 64))
            dense_tail = list(range(L - 64, L))
            strided_mid = list(range(64, L - 64, 32))
            check_steps = sorted(list(set(dense_head + strided_mid + dense_tail)))

        max_logit_err = 0.0
        max_prob_err = 0.0

        for t in range(L):
            x_t = x_embed[:, t:t+1, :]
            w_t, p_t = gauge.step(x_t)
            omega_decode[:, t:t+1, :] = w_t
            phi_decode[:, t:t+1, :] = p_t

            pos_t = torch.tensor([[t]], dtype=torch.long).expand(batch_size, 1)
            q_t = q_map[:, :, t:t+1, :]
            k_t = k_map[:, :, t:t+1, :]

            c_t, s_t = modulator(q_t, position_ids=pos_t, phi=p_t)
            qr_t = apply_rotary_pos_emb(q_t, c_t, s_t)
            kr_t = apply_rotary_pos_emb(k_t, c_t, s_t)

            cos_decode[:, t:t+1, :] = c_t
            sin_decode[:, t:t+1, :] = s_t
            q_rot_decode[:, :, t:t+1, :] = qr_t
            k_rot_decode[:, :, t:t+1, :] = kr_t
            k_cache[:, :, t:t+1, :] = kr_t

            if t in check_steps:
                # Query t attending over all cached keys j <= t
                cached_k_up_to_t = k_cache[:, :, :t+1, :]
                step_logits = torch.matmul(qr_t, cached_k_up_to_t.transpose(-1, -2)) * scale
                prefill_logits_row = torch.matmul(
                    q_rot_prefill[:, :, t:t+1, :],
                    k_rot_prefill[:, :, :t+1, :].transpose(-1, -2)
                ) * scale

                logit_diff = (step_logits - prefill_logits_row).abs().max().item()
                if logit_diff > max_logit_err:
                    max_logit_err = logit_diff

                # Softmax probabilities
                prob_step = F.softmax(step_logits, dim=-1)
                prob_prefill = F.softmax(prefill_logits_row, dim=-1)
                prob_diff = (prob_step - prob_prefill).abs().max().item()
                if prob_diff > max_prob_err:
                    max_prob_err = prob_diff

        # Error metrics across entire sequence
        err_omega = (omega_prefill - omega_decode).abs().max().item()
        err_phi = (phi_prefill - phi_decode).abs().max().item()
        err_cos = (cos_prefill - cos_decode).abs().max().item()
        err_sin = (sin_prefill - sin_decode).abs().max().item()
        err_q = (q_rot_prefill - q_rot_decode).abs().max().item()
        err_k = (k_rot_prefill - k_rot_decode).abs().max().item()

        elapsed = time.time() - t0

        # Assert all errors strictly < 1e-10
        assert err_omega < 1e-10, f"L={L}: omega discrepancy {err_omega:.2e} >= 1e-10"
        assert err_phi < 1e-10, f"L={L}: phi discrepancy {err_phi:.2e} >= 1e-10"
        assert err_cos < 1e-10, f"L={L}: cos discrepancy {err_cos:.2e} >= 1e-10"
        assert err_sin < 1e-10, f"L={L}: sin discrepancy {err_sin:.2e} >= 1e-10"
        assert err_q < 1e-10, f"L={L}: q_rot discrepancy {err_q:.2e} >= 1e-10"
        assert err_k < 1e-10, f"L={L}: k_rot discrepancy {err_k:.2e} >= 1e-10"
        assert max_logit_err < 1e-10, f"L={L}: logit discrepancy {max_logit_err:.2e} >= 1e-10"
        assert max_prob_err < 1e-10, f"L={L}: prob discrepancy {max_prob_err:.2e} >= 1e-10"

        metric = {
            "L": L,
            "anchor_mob_split": f"{anchor_count}/{mob_count}",
            "err_omega": err_omega,
            "err_phi": err_phi,
            "err_cos": err_cos,
            "err_sin": err_sin,
            "err_q": err_q,
            "err_k": err_k,
            "max_logit_err": max_logit_err,
            "max_prob_err": max_prob_err,
            "elapsed_sec": elapsed,
        }
        detailed_metrics.append(metric)

        print(f"  [✓] L={L:4d} | Mask ({anchor_count} anc / {mob_count} mob) | "
              f"Omega Err: {err_omega:.2e} | Phi Err: {err_phi:.2e} | "
              f"Logit Err: {max_logit_err:.2e} | Prob Err: {max_prob_err:.2e} | "
              f"Time: {elapsed:.2f}s")

    print("\n  Summary: All context lengths passed with strict numerical equivalence (< 1e-10 error)!")
    return {
        "detailed_metrics": detailed_metrics
    }


def test_suite1_extreme_angles_and_phase_flows():
    res = stress_test_extreme_angles_and_phase_flows(num_trials=5000)
    assert res["max_antipodal_err"] < 1e-10
    assert res["max_cyclic_err"] < 1e-10
    assert res["max_subspace_cancellation_err"] < 1e-10
    assert res["max_norm_err"] < 1e-12
    assert res["max_relative_rot_err_std"] < 1e-10


def test_suite2_lipschitz_continuity_under_attractor():
    res = stress_test_lipschitz_continuity_under_attractor(num_trials=5000)
    assert res["violations_clamped"] == 0
    assert res["max_clamped_bound_ratio"] <= 1.0 + 1e-11


def test_suite3_theorem3_odds_ratio_attenuation():
    res = stress_test_theorem3_odds_ratio_attenuation()
    assert res["max_rel_error_odds"] < 1e-10


def test_suite4_prefill_vs_decode_equivalence():
    res = stress_test_prefill_vs_decode_equivalence()
    for metric in res["detailed_metrics"]:
        assert metric["err_omega"] < 1e-10
        assert metric["err_phi"] < 1e-10
        assert metric["max_logit_err"] < 1e-10
        assert metric["max_prob_err"] < 1e-10


def main():
    print("=" * 80)
    print("  PHASE 4 EMPIRICAL CHALLENGER: ADVERSARIAL STRESS TEST HARNESS")
    print("=" * 80)
    
    t_start = time.time()
    
    res1 = stress_test_extreme_angles_and_phase_flows(num_trials=10000)
    res2 = stress_test_lipschitz_continuity_under_attractor(num_trials=10000)
    res3 = stress_test_theorem3_odds_ratio_attenuation()
    res4 = stress_test_prefill_vs_decode_equivalence()

    t_total = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"  ALL 4 ADVERSARIAL SUITES PASSED EMPIRICALLY IN {t_total:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
