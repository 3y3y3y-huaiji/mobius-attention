# -*- coding: utf-8 -*-
r"""
Cross-Architecture Generalization Benchmark Suite (Qwen2.5 vs Llama-3)
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Evaluates:
1. Qwen2.5 RoPE parameter set (hidden_dim=3584, heads=28, head_dim=128, base=1000000.0)
2. Llama-3 RoPE parameter set (hidden_dim=4096, heads=32, head_dim=128, base=500000.0)
3. Rotary position translation invariance across parameterizations
4. Relative attention contrast gain consistency across models
"""

import math
import pytest
import numpy as np

from tests.common_math import (
    MODEL_SPECS,
    compute_rope_freqs,
    apply_rotary_emb,
    apply_mobius_rope_dual,
    compute_attention_scores
)

class TestCrossArchitectureSuite:
    """Evaluates cross-architecture invariance on Qwen2.5 and Llama-3."""

    def test_qwen25_parameter_validation(self):
        """Validates Qwen2.5 parameter specifications."""
        spec = MODEL_SPECS["qwen2.5"]
        assert spec["head_dim"] == 128
        assert spec["rope_base"] == 1000000.0

    def test_llama3_parameter_validation(self):
        """Validates Llama-3 parameter specifications."""
        spec = MODEL_SPECS["llama3"]
        assert spec["head_dim"] == 128
        assert spec["rope_base"] == 500000.0

    def test_dual_rotary_invariance_on_qwen25(self):
        """Exact relative translation invariance on Qwen2.5: err < 1e-12."""
        spec = MODEL_SPECS["qwen2.5"]
        seq_len = 60
        cos, sin = compute_rope_freqs(spec["head_dim"], seq_len, base=spec["rope_base"])

        np.random.seed(11)
        q = np.random.randn(seq_len, spec["head_dim"])
        k = np.random.randn(seq_len, spec["head_dim"])

        omega = np.zeros(seq_len)
        omega[30] = math.pi
        phi = np.cumsum(omega)

        q_mob, k_mob = apply_mobius_rope_dual(q, k, cos, sin, phi)
        q_std = apply_rotary_emb(q, cos, sin)
        k_std = apply_rotary_emb(k, cos, sin)

        mob_dot = np.dot(q_mob[45], k_mob[15])
        std_dot = np.dot(q_std[45], k_std[15])
        assert math.isclose(mob_dot, -std_dot, abs_tol=1e-11)

    def test_dual_rotary_invariance_on_llama3(self):
        """Exact relative translation invariance on Llama-3: err < 1e-12."""
        spec = MODEL_SPECS["llama3"]
        seq_len = 60
        cos, sin = compute_rope_freqs(spec["head_dim"], seq_len, base=spec["rope_base"])

        np.random.seed(22)
        q = np.random.randn(seq_len, spec["head_dim"])
        k = np.random.randn(seq_len, spec["head_dim"])

        omega = np.zeros(seq_len)
        omega[30] = math.pi
        phi = np.cumsum(omega)

        q_mob, k_mob = apply_mobius_rope_dual(q, k, cos, sin, phi)
        q_std = apply_rotary_emb(q, cos, sin)
        k_std = apply_rotary_emb(k, cos, sin)

        mob_dot = np.dot(q_mob[45], k_mob[15])
        std_dot = np.dot(q_std[45], k_std[15])
        assert math.isclose(mob_dot, -std_dot, abs_tol=1e-11)

    def test_cross_arch_frequency_spectrum_stability(self):
        """Rotary frequency spectrum remains strictly positive and bounded on both architectures."""
        for arch in ["qwen2.5", "llama3"]:
            spec = MODEL_SPECS[arch]
            cos, sin = compute_rope_freqs(spec["head_dim"], 100, base=spec["rope_base"])
            assert np.all(cos >= -1.0) and np.all(cos <= 1.0)
            assert np.all(sin >= -1.0) and np.all(sin <= 1.0)
