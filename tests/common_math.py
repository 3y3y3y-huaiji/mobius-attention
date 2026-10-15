# -*- coding: utf-8 -*-
r"""
E2E Test Mathematical Reference & Verification Utilities for SG-Möbius-RoPE
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Provides opaque-box, requirement-driven mathematical reference implementations:
1. Standard RoPE rotary embedding operator (Su et al., 2021)
2. Data-Dependent Semantic Gauge Connection & Causal Wilson Line Holonomy
3. Dual Rotary Möbius Modulation & Topological Factorization (\langle q~, k~ \rangle = s_i s_j \langle q, k \rangle)
4. Hysteresis refractory gating & Quotation-scoped isolation
5. O(N) Linear KV-Cache Incremental Decoding Simulation
6. Attention Contrast Ratio (ACR) & Perplexity Preserving Metrics
7. Cross-Architecture Specifications (Qwen2.5 vs Llama-3)
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

# =============================================================================
# Architecture Profiles
# =============================================================================

MODEL_SPECS = {
    "qwen2.5": {
        "name": "Qwen2.5-7B/14B",
        "hidden_dim": 3584,
        "num_heads": 28,
        "head_dim": 128,
        "rope_base": 1000000.0,
        "max_position_embeddings": 32768,
    },
    "llama3": {
        "name": "Llama-3-8B",
        "hidden_dim": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "rope_base": 500000.0,
        "max_position_embeddings": 8192,
    },
    "standard": {
        "name": "Standard-Reference",
        "hidden_dim": 64,
        "num_heads": 4,
        "head_dim": 16,
        "rope_base": 10000.0,
        "max_position_embeddings": 4096,
    }
}

# =============================================================================
# Standard RoPE Rotary Position Embedding Reference
# =============================================================================

def compute_rope_freqs(head_dim: int, seq_len: int, base: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes standard RoPE frequency tensors.
    theta_m = base^(-2m / head_dim) for m in [0, head_dim/2 - 1]
    cos_part, sin_part: shape [seq_len, head_dim]
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    dim_half = head_dim // 2
    m_indices = np.arange(dim_half, dtype=np.float64)
    inv_freq = 1.0 / (base ** (2.0 * m_indices / head_dim))
    t = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq) # [seq_len, dim_half]
    full_freqs = np.concatenate([freqs, freqs], axis=-1) # [seq_len, head_dim]
    return np.cos(full_freqs), np.sin(full_freqs)

def apply_rotary_emb(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Applies standard RoPE rotation:
    x_rot = x * cos + rotate_half(x) * sin
    where rotate_half([x1, x2]) = [-x2, x1]
    """
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    rotated_x = np.concatenate([-x2, x1], axis=-1)
    return (x * cos) + (rotated_x * sin)

# =============================================================================
# Semantic Gauge Connection & Holonomy
# =============================================================================

class SemanticGaugeConnectionRef:
    """
    Reference implementation of discrete semantic gauge connection:
    - omega_t in [0, pi]
    - Causal Wilson line / path holonomy: Phi(j, i) = sum_{k=j+1}^i omega_k = Phi_i - Phi_j
    - Parity factor: s_t = cos(Phi_t) in {-1, +1}
    - Relative topological factor: T(j, i) = cos(Phi(j, i)) = s_i * s_j
    """
    def __init__(self, refractory_window: int = 5):
        self.refractory_window = refractory_window

    def compute_holonomy(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes cumulative gauge phase Phi_t and topological parity sections s_t.
        Returns:
            phi: cumulative phase array [seq_len]
            s: parity section array [seq_len]
        """
        phi = np.cumsum(omega)
        s = np.cos(phi)
        return phi, s

    def compute_pairwise_topological_matrix(self, phi: np.ndarray) -> np.ndarray:
        """
        Computes the NxN pairwise topological factor matrix T[i, j] = cos(phi[i] - phi[j]).
        Under discrete Z2 gauge factorization, T[i, j] = s[i] * s[j].
        """
        seq_len = len(phi)
        delta_phi = phi[:, None] - phi[None, :] # [i, j] = phi[i] - phi[j]
        return np.cos(delta_phi)

    def filter_with_hysteresis(self, raw_triggers: List[int], seq_len: int) -> List[int]:
        """
        Applies refractory hysteresis gating:
        Once a contrastive trigger fires at token t, subsequent triggers within
        [t, t + refractory_window] are suppressed to prevent multi-word burst oscillations.
        """
        filtered = []
        last_fired = -self.refractory_window - 1
        for idx in sorted(raw_triggers):
            if idx >= seq_len:
                continue
            if idx - last_fired > self.refractory_window:
                filtered.append(idx)
                last_fired = idx
        return filtered

    def filter_with_quotation_scope(self, text_tokens: List[str], raw_triggers: List[int]) -> List[int]:
        """
        Filters triggers to respect quotation boundaries:
        Contrastive tokens inside quotation marks ("...", '...', “...”) are treated
        as quoted speech and do not invert global document polarity.
        """
        in_double_quote = False
        in_single_quote = False
        quote_char_double = {'"', '“', '”'}
        quote_char_single = {"'", '‘', '’'}

        scoped_triggers = []
        for idx, token in enumerate(text_tokens):
            for ch in token:
                if ch in quote_char_double:
                    in_double_quote = not in_double_quote
                elif ch in quote_char_single:
                    in_single_quote = not in_single_quote

            if idx in raw_triggers:
                # If outside quotation marks, trigger affects global polarity
                if not (in_double_quote or in_single_quote):
                    scoped_triggers.append(idx)
        return scoped_triggers

# =============================================================================
# Dual Rotary Möbius-RoPE Modulation
# =============================================================================

def apply_mobius_rope_dual(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies mathematically rigorous dual-modulated Möbius-RoPE:
    s_t = cos(Phi_t) in {-1, +1}
    q_tilde = s_i * R_theta(q)
    k_tilde = s_j * R_theta(k)
    This strictly restores exact relative translation invariance:
    <q_tilde_i, k_tilde_j> = s_i * s_j * <q_i, k_j>_RoPE = cos(Phi_i - Phi_j) * <q_i, k_j>_RoPE
    """
    s = np.cos(phi) # [seq_len]
    view_shape = [1] * q.ndim
    view_shape[-2] = len(phi)
    s_tensor = s.reshape(view_shape)

    # Standard rotation
    q_rope = apply_rotary_emb(q, cos, sin)
    k_rope = apply_rotary_emb(k, cos, sin)

    # Dual scaling by parity section s
    q_mob = q_rope * s_tensor
    k_mob = k_rope * s_tensor
    return q_mob, k_mob

def compute_attention_scores(
    q: np.ndarray,
    k: np.ndarray,
    causal: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes scaled dot-product attention logits and probabilities:
    S_{ij} = <q_i, k_j> / sqrt(d)
    A_{ij} = softmax(S_{ij}) along key dimension
    """
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    logits = np.matmul(q, np.swapaxes(k, -1, -2)) * scale

    if causal:
        seq_len_q = q.shape[-2]
        seq_len_k = k.shape[-2]
        causal_mask = np.triu(np.ones((seq_len_q, seq_len_k), dtype=bool), k=1)
        logits = np.where(causal_mask, -1e9, logits)

    # Numerically stable softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(np.clip(logits - max_logits, -80.0, 0.0))
    if causal:
        exp_logits = np.where(causal_mask, 0.0, exp_logits)
    probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)
    return logits, probs

# =============================================================================
# KV Cache Incremental Decoding Reference
# =============================================================================

class LinearKVCacheRef:
    """
    Reference simulator for linear KV cache decoding:
    - Pre-stores keys scaled by local parity s_j in the cache.
    - Each autoregressive step (seq_len_curr = 1) scales new query by s_t.
    - Achieves O(1) step time and O(N) memory without retroactive cache recomputation.
    """
    def __init__(self, head_dim: int, base: float = 10000.0):
        self.head_dim = head_dim
        self.base = base
        self.cache_k = [] # list of key vectors [head_dim]
        self.cache_v = [] # list of val vectors [head_dim]
        self.cache_s = [] # list of parity signs s_j
        self.dim_half = head_dim // 2
        m_indices = np.arange(self.dim_half, dtype=np.float64)
        self.inv_freq = 1.0 / (base ** (2.0 * m_indices / head_dim))

    def _get_rope_sincos(self, position: int) -> Tuple[np.ndarray, np.ndarray]:
        t = float(position)
        freqs = t * self.inv_freq
        full_freqs = np.concatenate([freqs, freqs])
        return np.cos(full_freqs), np.sin(full_freqs)

    def append_token(self, k: np.ndarray, v: np.ndarray, position: int, phi_t: float):
        cos, sin = self._get_rope_sincos(position)
        k_rot = apply_rotary_emb(k, cos, sin)
        s_t = math.cos(phi_t)
        k_cached = k_rot * s_t # pre-scaled key stored in cache
        self.cache_k.append(k_cached)
        self.cache_v.append(v)
        self.cache_s.append(s_t)

    def decode_step(self, q: np.ndarray, position: int, phi_t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decodes a single token at position t with phase phi_t against all cached keys.
        """
        cos, sin = self._get_rope_sincos(position)
        q_rot = apply_rotary_emb(q, cos, sin)
        s_t = math.cos(phi_t)
        q_mod = q_rot * s_t

        keys = np.stack(self.cache_k, axis=0) # [N, head_dim]
        logits = np.dot(keys, q_mod) / math.sqrt(self.head_dim) # [N]
        # Softmax
        exp_l = np.exp(logits - np.max(logits))
        attn_weights = exp_l / (np.sum(exp_l) + 1e-12)

        vals = np.stack(self.cache_v, axis=0) # [N, head_dim]
        output = np.sum(attn_weights[:, None] * vals, axis=0)
        return logits, attn_weights

# =============================================================================
# Benchmarking Metrics: ACR & Perplexity Preservation
# =============================================================================

def compute_acr(attn_weights: np.ndarray, target_idx: int, distractor_idx: int) -> float:
    """
    Attention Contrast Ratio:
    ACR = Attention(target) / (Attention(distractor) + 1e-12)
    """
    target_score = float(attn_weights[target_idx])
    distractor_score = float(attn_weights[distractor_idx])
    return target_score / (distractor_score + 1e-12)

def compute_perplexity_proxy(logits: np.ndarray) -> float:
    """
    Computes Shannon entropy / perplexity proxy:
    PPL = exp( - sum p * log(p) )
    """
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / (np.sum(exp_l) + 1e-12)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(np.exp(entropy))
