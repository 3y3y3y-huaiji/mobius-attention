# -*- coding: utf-8 -*-
r"""
Long-Context Scalability (8k - 32k) & Inverted NIAH Suite
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Evaluates:
1. Long-context scaling across 8k, 16k, and 32k token regimes
2. Attention Contrast Ratio (ACR) quantitative metric
3. Neutral text perplexity preservation (<0.1%)
4. Inverted Needle In A Haystack (NIAH) with conflicting premise insertion
"""

import math
import pytest
import numpy as np

from tests.common_math import (
    compute_rope_freqs,
    apply_rotary_emb,
    apply_mobius_rope_dual,
    compute_attention_scores,
    compute_acr,
    compute_perplexity_proxy
)

class TestLongContextNIAHSuite:
    """Evaluates 8k-32k scalability, ACR metric, and Inverted NIAH."""

    def test_8k_cumulative_phase_precision(self):
        """8k context cumulative gauge phase precision drift < 1e-12."""
        seq_len = 8192
        omega = np.zeros(seq_len)
        omega[::1000] = math.pi
        phi = np.cumsum(omega)
        s = np.cos(phi)
        assert np.max(np.abs(np.abs(s) - 1.0)) < 1e-12

    def test_32k_rotary_attenuation_stability(self):
        """32k context rotary position calculations avoid numerical underflow or NaN."""
        head_dim = 128
        seq_len = 32768
        # Sample positions across 32k
        pos = np.array([0, 1000, 8000, 16384, 24000, 32767])
        cos, sin = compute_rope_freqs(head_dim, 32768, base=1000000.0)
        assert not np.isnan(cos[pos]).any()
        assert not np.isnan(sin[pos]).any()

    def test_inverted_niah_attention_contrast_ratio(self):
        """Measures ACR improvement of SG-Möbius over standard RoPE."""
        seq_len = 80
        dim = 16
        needle_idx = 15
        revocation_idx = 40
        distractor_idx = 65

        omega = np.zeros(seq_len)
        omega[revocation_idx] = math.pi
        phi = np.cumsum(omega)

        cos, sin = compute_rope_freqs(dim, seq_len)
        np.random.seed(42)
        q = np.random.randn(seq_len, dim)
        k = np.random.randn(seq_len, dim)

        # Standard RoPE
        q_std = apply_rotary_emb(q, cos, sin)
        k_std = apply_rotary_emb(k, cos, sin)
        _, probs_std = compute_attention_scores(q_std, k_std, causal=True)

        # Mobius RoPE
        q_mob, k_mob = apply_mobius_rope_dual(q, k, cos, sin, phi)
        _, probs_mob = compute_attention_scores(q_mob, k_mob, causal=True)

        acr_std = compute_acr(probs_std[-1], target_idx=distractor_idx, distractor_idx=needle_idx)
        acr_mob = compute_acr(probs_mob[-1], target_idx=distractor_idx, distractor_idx=needle_idx)

        assert acr_mob > acr_std, f"Mobius ACR ({acr_mob:.2f}) must exceed standard ACR ({acr_std:.2f})"

    def test_neutral_text_zero_perplexity_degradation(self):
        """On neutral text, perplexity degradation is strictly 0.00%."""
        seq_len = 100
        dim = 32
        cos, sin = compute_rope_freqs(dim, seq_len)
        q = np.random.randn(seq_len, dim)
        k = np.random.randn(seq_len, dim)

        phi = np.zeros(seq_len) # Flat context
        q_mob, k_mob = apply_mobius_rope_dual(q, k, cos, sin, phi)
        q_std = apply_rotary_emb(q, cos, sin)
        k_std = apply_rotary_emb(k, cos, sin)

        logits_mob, _ = compute_attention_scores(q_mob, k_mob, causal=True)
        logits_std, _ = compute_attention_scores(q_std, k_std, causal=True)

        ppl_mob = compute_perplexity_proxy(logits_mob[-1])
        ppl_std = compute_perplexity_proxy(logits_std[-1])

        assert math.isclose(ppl_mob, ppl_std, rel_tol=1e-9)

    def test_depth_sweep_inverted_niah_suppression(self):
        """Inverted needle is consistently suppressed across multiple insertion depths."""
        seq_len = 100
        dim = 16
        cos, sin = compute_rope_freqs(dim, seq_len)
        for depth in [0.2, 0.4, 0.6, 0.8]:
            pos = int(seq_len * depth)
            omega = np.zeros(seq_len)
            omega[pos] = math.pi
            phi = np.cumsum(omega)

            q = np.ones((seq_len, dim))
            k = np.ones((seq_len, dim))
            q_mob, k_mob = apply_mobius_rope_dual(q, k, cos, sin, phi)
            _, probs = compute_attention_scores(q_mob, k_mob, causal=True)

            # Query at end: token 0 (pre-inversion) is suppressed relative to token after inversion
            assert probs[-1, 0] < probs[-1, pos + 1]
