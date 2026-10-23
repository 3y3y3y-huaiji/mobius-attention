# -*- coding: utf-8 -*-
"""
Numerical verification of Theorem 7 (thm:multi_turn_contraction) and Corollary 8 (cor:k_turn_acr)
of paper/main.tex (L399-L455): multi-turn parity-alternating partition function contraction,
uniform in-phase amplification, decoy cushioning, and ACR compounding.

Self-contained reference model (no src/ import, no network, no GPU):
- K reversal positions t_1 < ... < t_K <= i inject a nominal phase shift pi each.
- R(l) = #{r : t_r <= l}; segments S_s = [t_s, t_{s+1}) with t_0 = 0, t_{K+1} = i + 1.
- For l in S_s: (K - s) even -> in-phase, tilde_A = A_std; (K - s) odd -> anti-phase, tilde_A = A_std - 2 A_mob.
- Bounded perturbation |delta_l| <= pi/6:
  anti-phase  tilde_A = A_std - (1 + cos delta) A_mob -+ sin delta B_mob,
  in-phase    tilde_A = A_std - (1 - cos delta) A_mob +- sin delta B_mob.
"""

import numpy as np
import pytest

TOL = 1e-10
N_TOKENS = 40
SEEDS = (0, 1, 2)


def reversal_positions(rng, k_turns, n_tokens=N_TOKENS):
    """K reversal positions 1 <= t_1 < ... < t_K <= i = n_tokens - 1."""
    pos = rng.choice(np.arange(1, n_tokens), size=k_turns, replace=False)
    return np.sort(pos)


def reversal_counts(t_pos, n_tokens=N_TOKENS):
    """R(l) = #{r : t_r <= l} for l = 0 .. n_tokens - 1 (segment index s = R(l))."""
    return np.searchsorted(t_pos, np.arange(n_tokens), side="right")


def inverted_mask(k_turns, rev_counts):
    """I_inv = {l : K - R(l) odd}: voided anti-phase tokens of Theorem 7(i)."""
    return (k_turns - rev_counts) % 2 == 1


def nominal_logits(a_std, a_mob, inv_mask):
    """Theorem 7(i): anti-phase tilde_A = A_std - 2 A_mob, in-phase tilde_A = A_std."""
    return np.where(inv_mask, a_std - 2.0 * a_mob, a_std)


def perturbed_logits(a_std, a_mob, b_mob, delta, sign, inv_mask):
    """Theorem 7(ii): anti-phase A_std - (1+cos d)A - s sin d B, in-phase A_std - (1-cos d)A + s sin d B."""
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    anti = a_std - (1.0 + cos_d) * a_mob - sign * sin_d * b_mob
    in_phase = a_std - (1.0 - cos_d) * a_mob + sign * sin_d * b_mob
    return np.where(inv_mask, anti, in_phase)


def sample_instance(rng, k_turns, n_tokens=N_TOKENS):
    """Draw reversal layout and logits with A_std in [-1, 1] and A_mob > 0 everywhere (coherence hypothesis)."""
    t_pos = reversal_positions(rng, k_turns, n_tokens)
    rev = reversal_counts(t_pos, n_tokens)
    inv = inverted_mask(k_turns, rev)
    a_std = rng.uniform(-1.0, 1.0, size=n_tokens)
    a_mob = rng.uniform(0.1, 1.5, size=n_tokens)
    return t_pos, rev, inv, a_std, a_mob


@pytest.mark.parametrize("k_turns", range(1, 11))
def test_closed_form_partition_contraction(k_turns):
    """Theorem 7(i) closed form: Z~ = sum exp(tilde_A) = Z_std - Delta Z_tot with Delta Z_tot over I_inv."""
    for seed in SEEDS:
        rng = np.random.default_rng(100 * k_turns + seed)
        _, _, inv, a_std, a_mob = sample_instance(rng, k_turns)
        std_mass = np.exp(a_std)
        tilde = nominal_logits(a_std, a_mob, inv)
        z_std = float(std_mass.sum())
        z_direct = float(np.exp(tilde).sum())
        dz_terms = np.where(inv, std_mass * (1.0 - np.exp(-2.0 * a_mob)), 0.0)
        dz_tot = float(dz_terms.sum())
        z_closed = z_std - dz_tot
        assert abs(z_direct - z_closed) < TOL
        assert z_direct < z_std
        assert dz_tot > 0.0
        # pointwise: voided masses shrink by exactly exp(A_std)(1 - exp(-2 A_mob)); revived masses are untouched
        mass = np.exp(tilde)
        assert np.max(np.abs(mass[inv] - (std_mass[inv] - dz_terms[inv]))) < 1e-12
        assert np.array_equal(mass[~inv], std_mass[~inv])


def test_parity_alternating_segment_partition_k3():
    """Theorem 7(i) segment parity for K = 3: S_0/S_2 voided (K-s odd), S_1/S_3 revived (K-s even), token-wise."""
    rng = np.random.default_rng(3)
    k_turns = 3
    t_pos, rev, inv, a_std, a_mob = sample_instance(rng, k_turns)
    for s in range(k_turns + 1):
        lo = 0 if s == 0 else int(t_pos[s - 1])
        hi = int(t_pos[s]) if s < k_turns else N_TOKENS
        seg = np.arange(lo, hi)
        assert seg.size > 0
        assert np.all(rev[seg] == s)
        assert np.all(inv[seg] == ((k_turns - s) % 2 == 1))
    # explicit token-wise formula selection via R(l) and parity of K - s
    expected = np.empty(N_TOKENS)
    for idx in range(N_TOKENS):
        s_val = int(rev[idx])
        if (k_turns - s_val) % 2 == 1:
            expected[idx] = a_std[idx] - 2.0 * a_mob[idx]
        else:
            expected[idx] = a_std[idx]
    assert np.array_equal(nominal_logits(a_std, a_mob, inv), expected)
    # S_0 and S_2 are voided, S_1 and S_3 are revived
    for s in (0, 2):
        lo = 0 if s == 0 else int(t_pos[s - 1])
        hi = int(t_pos[s]) if s < k_turns else N_TOKENS
        assert np.all(inv[lo:hi])
    for s in (1, 3):
        lo = int(t_pos[s - 1])
        hi = int(t_pos[s]) if s < k_turns else N_TOKENS
        assert not np.any(inv[lo:hi])


@pytest.mark.parametrize("k_turns", range(1, 11))
def test_uniform_in_phase_amplification(k_turns):
    """Corollary 8(1): P_mob/P_std = T = 1/(1 - Delta Z_tot/Z_std) on every in-phase token, identity-free."""
    for seed in SEEDS:
        rng = np.random.default_rng(100 * k_turns + seed)
        _, _, inv, a_std, a_mob = sample_instance(rng, k_turns)
        tilde = nominal_logits(a_std, a_mob, inv)
        z_std = float(np.exp(a_std).sum())
        z_tilde = float(np.exp(tilde).sum())
        dz_tot = float(np.where(inv, np.exp(a_std) * (1.0 - np.exp(-2.0 * a_mob)), 0.0).sum())
        t_amp = 1.0 / (1.0 - dz_tot / z_std)
        assert t_amp > 1.0
        p_std = np.exp(a_std) / z_std
        p_mob = np.exp(tilde) / z_tilde
        ratios = p_mob[~inv] / p_std[~inv]
        assert np.max(np.abs(ratios - t_amp)) < TOL
        assert np.max(np.abs(ratios - ratios[0])) < TOL


@pytest.mark.parametrize("k_turns", range(1, 11))
def test_acr_compounding_identity(k_turns):
    """Corollary 8(3): ACR_mob/ACR_std = exp(2 A_j^mob) = T * D_cushion, invariant to K and background masses."""
    a_j_mob = 0.7
    for seed in (0, 1):
        rng = np.random.default_rng(1000 * k_turns + seed)
        _, _, inv, a_std, a_mob = sample_instance(rng, k_turns)
        j = int(np.flatnonzero(inv)[0])
        k_tok = int(np.flatnonzero(~inv)[0])
        a_mob[j] = a_j_mob
        tilde = nominal_logits(a_std, a_mob, inv)
        z_std = float(np.exp(a_std).sum())
        z_tilde = float(np.exp(tilde).sum())
        t_amp = z_std / z_tilde
        p_std = np.exp(a_std) / z_std
        p_mob = np.exp(tilde) / z_tilde
        acr_ratio = (p_mob[k_tok] / p_mob[j]) / (p_std[k_tok] / p_std[j])
        d_cushion = 1.0 / (np.exp(-2.0 * a_j_mob) * t_amp)
        assert abs(acr_ratio - np.exp(2.0 * a_j_mob)) < 1e-12
        assert abs(t_amp * d_cushion - np.exp(2.0 * a_j_mob)) < 1e-12
        assert abs(acr_ratio - t_amp * d_cushion) < 1e-12
        assert t_amp > 1.0
        assert d_cushion > 1.0


@pytest.mark.parametrize("k_turns", range(1, 11))
def test_perturbed_sufficient_condition(k_turns):
    """Theorem 7(ii): |delta| <= pi/6 with (1+cos d)A_mob > |sin d B| dominance plus global balance => Z~ < Z_std."""
    rng = np.random.default_rng(500 + k_turns)
    for _trial in range(3):
        anti_contraction = -np.inf
        in_phase_leak = np.inf
        for _attempt in range(500):
            _, _, inv, a_std, a_mob = sample_instance(rng, k_turns)
            b_mob = rng.uniform(-1.0, 1.0, size=N_TOKENS) * 0.3 * a_mob
            delta = rng.uniform(-np.pi / 6.0, np.pi / 6.0, size=N_TOKENS)
            sign = rng.choice([-1.0, 1.0], size=N_TOKENS)
            assert np.all(np.abs(delta) <= np.pi / 6.0 + 1e-15)
            dominance = (1.0 + np.cos(delta)) * a_mob > np.abs(np.sin(delta) * b_mob)
            assert np.all(dominance[inv])
            tilde = perturbed_logits(a_std, a_mob, b_mob, delta, sign, inv)
            std_mass = np.exp(a_std)
            mass = np.exp(tilde)
            anti_contraction = float(np.sum(std_mass[inv] - mass[inv]))
            in_phase_leak = float(np.sum(np.maximum(0.0, mass[~inv] - std_mass[~inv])))
            if anti_contraction > in_phase_leak:
                break
        else:
            pytest.fail("could not sample an instance meeting the global contraction condition")
        assert anti_contraction > in_phase_leak
        z_std = float(np.exp(a_std).sum())
        z_tilde = float(np.exp(tilde).sum())
        assert z_tilde < z_std


@pytest.mark.parametrize("case", ["dominance_violated", "delta_beyond_pi_over_6"])
def test_perturbed_boundary_counterexample(case):
    """Theorem 7(ii) boundary: once dominance or |delta| <= pi/6 fails, contraction is no longer guaranteed."""
    if case == "dominance_violated":
        # K = 1, t_1 = 2: voided tokens {0, 1}; token 0 breaks (1+cos d)A_mob > |sin d B| and gains mass.
        t_pos = np.array([2])
        a_std = np.zeros(4)
        a_mob = np.full(4, 0.02)
        b_mob = np.array([-20.0, 0.0, 0.0, 0.0])
        delta = np.full(4, np.pi / 6.0)
        sign = np.ones(4)
        assert not (1.0 + np.cos(delta[0])) * a_mob[0] > abs(np.sin(delta[0]) * b_mob[0])
    else:
        # |delta| = pi/3 > pi/6: the +- sin delta B leakage on the revived token overwhelms the contraction.
        t_pos = np.array([3])
        a_std = np.zeros(4)
        a_mob = np.full(4, 0.01)
        b_mob = np.array([0.0, 0.0, 0.0, 5.0])
        delta = np.full(4, np.pi / 3.0)
        sign = np.ones(4)
        assert np.all(np.abs(delta) > np.pi / 6.0)
    rev = reversal_counts(t_pos, n_tokens=4)
    inv = inverted_mask(1, rev)
    tilde = perturbed_logits(a_std, a_mob, b_mob, delta, sign, inv)
    z_std = float(np.exp(a_std).sum())
    z_tilde = float(np.exp(tilde).sum())
    assert z_tilde >= z_std


@pytest.mark.parametrize("seed", range(3))
def test_k1_regression_matches_corollary4(seed):
    """Theorem 7 for K = 1: Delta Z_tot degenerates term-by-term to Corollary 4 Delta Z_P and exp(2A) = T * D."""
    rng = np.random.default_rng(900 + seed)
    k_turns = 1
    t_pos, rev, inv, a_std, a_mob = sample_instance(rng, k_turns)
    t1 = int(t_pos[0])
    prefix = np.arange(N_TOKENS) < t1
    assert np.array_equal(inv, prefix)
    assert np.all(rev[prefix] == 0)
    assert np.all(rev[~prefix] == 1)
    terms_thm = np.where(inv, np.exp(a_std) * (1.0 - np.exp(-2.0 * a_mob)), 0.0)
    terms_cor4 = np.where(prefix, np.exp(a_std) * (1.0 - np.exp(-2.0 * a_mob)), 0.0)
    assert np.array_equal(terms_thm, terms_cor4)
    dz_tot = float(terms_thm.sum())
    dz_p = float(terms_cor4.sum())
    assert dz_tot == dz_p
    tilde = nominal_logits(a_std, a_mob, inv)
    z_std = float(np.exp(a_std).sum())
    z_tilde = float(np.exp(tilde).sum())
    t_amp = 1.0 / (1.0 - dz_p / z_std)
    assert abs(t_amp - z_std / z_tilde) < TOL
    j = t1 - 1
    k_tok = t1
    a_j = a_mob[j]
    d_cushion = 1.0 / (np.exp(-2.0 * a_j) * t_amp)
    assert abs(np.exp(2.0 * a_j) - t_amp * d_cushion) < 1e-12
    p_std = np.exp(a_std) / z_std
    p_mob = np.exp(tilde) / z_tilde
    acr_ratio = (p_mob[k_tok] / p_mob[j]) / (p_std[k_tok] / p_std[j])
    assert abs(acr_ratio - t_amp * d_cushion) < 1e-12
    assert abs(acr_ratio - np.exp(2.0 * a_j)) < 1e-12
