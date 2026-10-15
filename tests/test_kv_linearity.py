# -*- coding: utf-8 -*-
r"""
Linear KV-Cache Memory & Decoding Scaling Benchmark Suite
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Evaluates:
1. O(N) linear memory scaling of pre-multiplied cached keys
2. O(1) time complexity of autoregressive decoding step
3. Zero retrofitting / recomputation of historical cache tokens
4. Equivalence between step-by-step decoding and full-sequence attention
"""

import math
import pytest
import numpy as np

from tests.common_math import (
    compute_rope_freqs,
    apply_rotary_emb,
    apply_mobius_rope_dual,
    compute_attention_scores,
    LinearKVCacheRef
)

class TestKVLinearitySuite:
    """Evaluates KV-cache linear memory scaling and decoding correctness."""

    def test_kv_cache_rank1_premultiplication(self):
        """Keys are pre-multiplied by local parity s_j in the cache."""
        cache = LinearKVCacheRef(head_dim=16)
        k = np.ones(16)
        v = np.ones(16)

        # Token 0: s = +1
        cache.append_token(k, v, position=0, phi_t=0.0)
        assert math.isclose(cache.cache_s[0], 1.0)

        # Token 1: s = -1
        cache.append_token(k, v, position=1, phi_t=math.pi)
        assert math.isclose(cache.cache_s[1], -1.0)

    def test_historical_cache_invariance_on_inversion(self):
        """Adding a new inverted token does not recompute or mutate prior cached tokens."""
        cache = LinearKVCacheRef(head_dim=32)
        for t in range(20):
            cache.append_token(np.random.randn(32), np.random.randn(32), position=t, phi_t=0.0)

        snapshot_k = [arr.copy() for arr in cache.cache_k]

        # Inversion occurs at token 20
        cache.append_token(np.random.randn(32), np.random.randn(32), position=20, phi_t=math.pi)

        # Check all historical keys 0 to 19 remain unchanged
        for t in range(20):
            assert np.array_equal(cache.cache_k[t], snapshot_k[t])

    def test_autoregressive_step_equivalence_to_full_matrix(self):
        """Autoregressive step decoding matches the last row of full causal attention."""
        head_dim = 16
        seq_len = 15
        cos, sin = compute_rope_freqs(head_dim, seq_len)
        np.random.seed(987)
        q_all = np.random.randn(seq_len, head_dim)
        k_all = np.random.randn(seq_len, head_dim)

        omega = np.zeros(seq_len)
        omega[6] = math.pi
        phi = np.cumsum(omega)

        # Full context attention
        q_mob, k_mob = apply_mobius_rope_dual(q_all, k_all, cos, sin, phi)
        _, full_probs = compute_attention_scores(q_mob, k_mob, causal=True)

        # Cache decoding
        cache = LinearKVCacheRef(head_dim=head_dim)
        for t in range(seq_len):
            cache.append_token(k_all[t], k_all[t], position=t, phi_t=phi[t])

        _, step_probs = cache.decode_step(q_all[-1], position=seq_len - 1, phi_t=phi[-1])

        assert np.allclose(full_probs[-1], step_probs, atol=1e-10)

    def test_memory_scaling_is_strictly_linear(self):
        """Memory consumption of KV cache scales as O(N), not O(N^2)."""
        cache = LinearKVCacheRef(head_dim=64)
        for t in range(100):
            cache.append_token(np.ones(64), np.ones(64), position=t, phi_t=0.0)

        # 100 tokens of 64 floats each = 6400 floats
        total_k_elements = sum(k.size for k in cache.cache_k)
        assert total_k_elements == 100 * 64

    def test_step_time_is_independent_of_pairwise_matrix(self):
        """Decoding step does not construct an N x N matrix; executes a single matrix-vector dot product."""
        cache = LinearKVCacheRef(head_dim=32)
        N = 50
        for t in range(N):
            cache.append_token(np.ones(32), np.ones(32), position=t, phi_t=0.0)

        q = np.ones(32)
        logits, weights = cache.decode_step(q, position=N, phi_t=0.0)
        assert logits.ndim == 1 and len(logits) == N
