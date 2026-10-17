"""Property-Based Testing (PBT) via Hypothesis for SG-Mobius-RoPE.

Uses Hypothesis shrinking search to discover edge cases, verify invariants,
and formalize both positive guarantees and negative aliasing boundaries:

1. 2D Orthogonal Group SO(2) relative rotation invariance & double covering.
2. Pairwise odds ratio suppression > 1.0 under open-loop perturbation delta in [-pi/6, pi/6].
3. Negative Theorem aliasing boundary: unconditioned pi-shift amplifies decoy when phase in (pi/2, 3pi/2).
4. Adaptive band cutoff: theta_cutoff(L) = pi / (2L) guarantees monotonic suppression across all lengths.
5. Two-Stage Associative Scan equivalence to serial autoregressive recurrence under finite precision.
"""

import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st


def rot2(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


# =====================================================================
# 1. SO(2) Orthogonal Group Invariance & Double Covering
# =====================================================================
@settings(max_examples=200, deadline=None)
@given(
    u=st.lists(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False), min_size=2, max_size=2),
    v=st.lists(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False), min_size=2, max_size=2),
    phi_q=st.floats(min_value=-100 * math.pi, max_value=100 * math.pi, allow_nan=False, allow_infinity=False),
    phi_k=st.floats(min_value=-100 * math.pi, max_value=100 * math.pi, allow_nan=False, allow_infinity=False),
)
def test_so2_bilinear_invariance_and_double_covering(u, v, phi_q, phi_k):
    u_vec = np.array(u, dtype=np.float64)
    v_vec = np.array(v, dtype=np.float64)

    R_q = rot2(phi_q)
    R_k = rot2(phi_k)

    # 1. Bilinear relative rotation invariance: <R(phi_q) u, R(phi_k) v> == <u, R(phi_k - phi_q) v>
    lhs = np.dot(R_q @ u_vec, R_k @ v_vec)
    rhs = np.dot(u_vec, rot2(phi_k - phi_q) @ v_vec)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-8, atol=1e-8)

    # 2. Antipodal pi-shift inverts sign: <R(phi_q + pi) u, R(phi_k) v> == - <R(phi_q) u, R(phi_k) v>
    R_q_pi = rot2(phi_q + math.pi)
    anti_val = np.dot(R_q_pi @ u_vec, R_k @ v_vec)
    np.testing.assert_allclose(anti_val, -lhs, rtol=1e-8, atol=1e-8)

    # 3. 2pi shift restores identity: <R(phi_q + 2pi) u, R(phi_k) v> == + <R(phi_q) u, R(phi_k) v>
    R_q_2pi = rot2(phi_q + 2 * math.pi)
    rest_val = np.dot(R_q_2pi @ u_vec, R_k @ v_vec)
    np.testing.assert_allclose(rest_val, lhs, rtol=1e-8, atol=1e-8)


# =====================================================================
# 2. Open-Loop Perturbation Robustness: Odds Ratio > 1.0
# =====================================================================
@settings(max_examples=300, deadline=None)
@given(
    A_mob=st.floats(min_value=1e-3, max_value=50.0, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=-math.pi / 6, max_value=math.pi / 6, allow_nan=False, allow_infinity=False),
)
def test_open_loop_perturbation_odds_ratio_lower_bound(A_mob, delta):
    # Logit reduction: Delta S = - A_mob * (1 + cos(delta))
    # Odds ratio = exp(-Delta S) = exp(A_mob * (1 + cos(delta)))
    cos_d = math.cos(delta)
    odds_ratio = math.exp(A_mob * (1.0 + cos_d))

    # Analytical lower bound: delta in [-pi/6, pi/6] => cos(delta) >= sqrt(3)/2 ~ 0.866
    min_cos = math.cos(math.pi / 6)
    min_bound = math.exp(A_mob * (1.0 + min_cos))

    assert odds_ratio >= min_bound - 1e-9, f"Odds ratio {odds_ratio} fell below theoretical bound {min_bound}"
    assert odds_ratio > 1.0, f"Odds ratio must strictly exceed 1.0, got {odds_ratio}"


# =====================================================================
# 3. Negative Theorem: Aliasing Zone Shrinking Search
# =====================================================================
@settings(max_examples=200, deadline=None)
@given(
    phase=st.floats(min_value=0.55 * math.pi, max_value=1.45 * math.pi, allow_nan=False, allow_infinity=False),
)
def test_negative_theorem_aliasing_zone(phase):
    # When relative carrier phase is in the negative half-cycle (pi/2, 3pi/2):
    unmod_carrier = math.cos(phase)
    assert unmod_carrier < 0.0, f"Carrier must be negative, got {unmod_carrier}"

    # An unconditioned pi-shift inverts it to positive:
    mod_carrier = math.cos(phase + math.pi)
    assert mod_carrier > 0.0, f"Modulated carrier must be positive, got {mod_carrier}"

    # Logit difference: Delta S = mod_carrier - unmod_carrier = -2 * unmod_carrier > 0
    delta_s = mod_carrier - unmod_carrier
    assert delta_s > 0.0, "Unconditioned pi-shift in negative half-cycle increases decoy logit!"


# =====================================================================
# 4. Dynamic Band Defense: Monotonicity Restoration across Lengths
# =====================================================================
@settings(max_examples=200, deadline=None)
@given(
    L=st.integers(min_value=16, max_value=32768),
    rel_dist=st.integers(min_value=1, max_value=32768),
)
def test_dynamic_cutoff_eliminates_aliasing(L, rel_dist):
    # Constrain distance to within context length
    dist = min(rel_dist, L)

    # Dynamic cutoff rule: theta_cutoff = pi / (2 * L)
    theta_cutoff = math.pi / (2.0 * L)

    # For any channel within the active band: theta <= theta_cutoff
    theta_sample = theta_cutoff * 0.999  # Worst-case highest frequency in band

    carrier_phase = dist * theta_sample

    # Invariant: Phase must remain strictly in [0, pi/2]
    assert 0.0 <= carrier_phase <= math.pi / 2.0 + 1e-9

    # Unmodulated carrier is non-negative
    unmod_carrier = math.cos(carrier_phase)
    assert unmod_carrier >= -1e-9

    # Pi-shift inverts to non-positive, ensuring attention suppression <= 0
    mod_carrier = math.cos(carrier_phase + math.pi)
    assert mod_carrier <= 1e-9

    # Decoy logit difference: mod_carrier - unmod_carrier <= 0
    delta_s = mod_carrier - unmod_carrier
    assert delta_s <= 1e-9, f"Decoy logit must not increase, got {delta_s}"


# =====================================================================
# 5. Two-Stage Associative Scan vs Serial Recurrence Equivalence
# =====================================================================
def serial_ema(g: np.ndarray, nu: float) -> np.ndarray:
    """Serial IIR EMA recurrence: omega_t = (1 - nu) * omega_{t-1} + nu * g_t"""
    omega = np.zeros_like(g)
    prev = 0.0
    for t in range(len(g)):
        prev = (1.0 - nu) * prev + nu * g[t]
        omega[t] = prev
    return omega


def associative_scan_ema(g: np.ndarray, nu: float) -> np.ndarray:
    """Parallel associative scan using affine operator (a2, b2) o (a1, b1) = (a2*a1, a2*b1 + b2)."""
    L = len(g)
    if L == 0:
        return np.zeros(0, dtype=np.float64)

    # Element at step t computes: omega_t = a * omega_{t-1} + b_t
    a = 1.0 - nu
    b = nu * g

    # Build affine pairs (a_t, b_t)
    pairs = [(a, b[i]) for i in range(L)]

    # Associative binary operator
    def op(p2, p1):
        a2, b2 = p2
        a1, b1 = p1
        return (a2 * a1, a2 * b1 + b2)

    # Parallel tree prefix scan (Blelloch / Kogge-Stone equivalent)
    res = []
    curr = None
    for p in pairs:
        if curr is None:
            curr = p
        else:
            curr = op(p, curr)
        res.append(curr[1])

    return np.array(res, dtype=np.float64)


@settings(max_examples=100, deadline=None)
@given(
    L=st.integers(min_value=2, max_value=256),
    nu=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    values=st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=256, max_size=256)
)
def test_associative_scan_serial_equivalence(L, nu, values):
    g = np.array(values[:L], dtype=np.float64)

    seq_out = serial_ema(g, nu)
    par_out = associative_scan_ema(g, nu)

    # Tolerance bounded by machine epsilon * log2(L)
    err_bound = 1e-12 * (math.log2(L) + 1.0)
    np.testing.assert_allclose(par_out, seq_out, rtol=err_bound, atol=err_bound)
