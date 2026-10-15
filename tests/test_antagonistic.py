# -*- coding: utf-8 -*-
r"""
Antagonistic Stress-Testing Benchmark Suite
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Evaluates:
1. Nested negation sequences (e.g. 'not untrue', 'never invalid')
2. Negations scoped within quotation marks
3. Rapid consecutive contrastive tokens
4. Refractory hysteresis gating (preventing even-parity oscillation)
5. Mixed-polarity text flow attention distributions
"""

import math
import pytest
import numpy as np

from tests.common_math import (
    compute_rope_freqs,
    apply_rotary_emb,
    SemanticGaugeConnectionRef,
    apply_mobius_rope_dual,
    compute_attention_scores
)

class TestAntagonisticSuite:
    """Antagonistic stress cases for SG-Möbius-RoPE."""

    def test_nested_negation_topological_inversion(self):
        """Nested negation flips polarity back to positive alignment."""
        gauge = SemanticGaugeConnectionRef(refractory_window=0)
        # Sequence: [premise, not, untrue, conclusion]
        omega = np.zeros(10)
        omega[3] = math.pi # 'not'
        omega[4] = math.pi # 'untrue'
        phi, s = gauge.compute_holonomy(omega)

        assert s[2] == 1.0  # Before negations
        assert s[3] == -1.0 # First negation
        assert s[4] == 1.0  # Negation of negation
        assert s[9] == 1.0  # End of text

    def test_quotation_mark_boundary_isolation(self):
        """Contrastive keywords inside quotations do not disrupt outer context."""
        tokens = ["The", "judge", "ruled", '"', "the", "claim", "is", "false", '"', "and", "dismissed", "it"]
        raw_triggers = [7] # 'false' is at index 7 inside quotes
        gauge = SemanticGaugeConnectionRef()
        scoped = gauge.filter_with_quotation_scope(tokens, raw_triggers)

        # Quoted trigger must be excluded
        assert 7 not in scoped

    def test_rapid_consecutive_contrastive_burst(self):
        """Refractory window compresses multi-word contrastive burst into a single inversion."""
        seq_len = 50
        # Multi-word burst: 'However', 'nonetheless', 'on', 'the', 'contrary'
        burst = [15, 16, 17, 18, 19]
        gauge = SemanticGaugeConnectionRef(refractory_window=8)
        filtered = gauge.filter_with_hysteresis(burst, seq_len)

        # Only the first token fires
        assert filtered == [15]
        omega = np.zeros(seq_len)
        omega[filtered] = math.pi
        _, s = gauge.compute_holonomy(omega)
        assert s[-1] == -1.0 # Odd parity (-1) preserved

    def test_contrastive_with_punctuation(self):
        """Punctuation attached to contrastive words handled properly."""
        tokens = ["Initial", "premise.", "However,", "new", "evidence."]
        gauge = SemanticGaugeConnectionRef()
        scoped = gauge.filter_with_quotation_scope(tokens, raw_triggers=[2])
        assert scoped == [2]

    def test_triplet_alternating_negation(self):
        """Three alternating negations yields net odd parity (-1.0)."""
        gauge = SemanticGaugeConnectionRef(refractory_window=0)
        omega = np.zeros(20)
        omega[[5, 10, 15]] = math.pi
        phi, s = gauge.compute_holonomy(omega)

        assert s[4] == 1.0
        assert s[7] == -1.0
        assert s[12] == 1.0
        assert s[18] == -1.0
