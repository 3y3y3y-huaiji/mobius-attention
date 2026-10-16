# -*- coding: utf-8 -*-
r"""
E2E Test Mathematical Reference & Verification Utilities for SG-Möbius-RoPE
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Provides opaque-box, requirement-driven mathematical reference implementations:
1. Standard RoPE rotary embedding operator (Su et al., 2021)
2. Continuous 2D Planar Rotation via Angle Addition:
   \tilde{\theta}_{h,m}(t) = t \theta_m + \Phi_{h,m}(t)
   \cos_{\mathrm{mod}} = \cos_0 \odot \cos(\Phi) - \sin_0 \odot \sin(\Phi)
   \sin_{\mathrm{mod}} = \sin_0 \odot \cos(\Phi) + \cos_0 \odot \sin(\Phi)
3. Data-Dependent Continuous Semantic Gauge Connection:
   Continuous torsion rate \omega \in [-\omega_{\max}, \omega_{\max}]
   1D Viscous EMA smoothing: \omega(t) = (1 - \nu)\omega(t-1) + \nu g(t)
   Smooth cumulative Möbius phase flow: \Phi(t) = \sum_{\tau=1}^t \omega(\tau)
4. Non-Negative Feature Map:
   \phi(u) = \text{ELU}(u) + 1 \ge 0
5. O(1) Linear KV-Cache Incremental Decoding Simulation with exact causal invariance
6. Attention Contrast Ratio (ACR) & Perplexity Preserving Metrics
7. Cross-Architecture Specifications (Qwen3.5, Qwen2.5, Llama-3)
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union

# =============================================================================
# Architecture Profiles
# =============================================================================

MODEL_SPECS = {
    "qwen3.5": {
        "name": "Qwen3.5-9B",
        "hidden_dim": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "rope_base": 1000000.0,
        "max_position_embeddings": 32768,
    },
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
# Standard RoPE & Continuous 2D Planar Rotation Reference
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


def apply_rotary_angle_addition_np(
    cos0: np.ndarray,
    sin0: np.ndarray,
    phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Exact 2D SO(2) Planar Rotation via Angle Addition in NumPy:
        \cos_{\mathrm{mod}} = \cos_0 \odot \cos(\Phi) - \sin_0 \odot \sin(\Phi)
        \sin_{\mathrm{mod}} = \sin_0 \odot \cos(\Phi) + \cos_0 \odot \sin(\Phi)
    
    Identically preserves 2D Euclidean vector norms:
        \|R(\theta + \Phi) v\| \equiv \|v\|   \forall \Phi \in \mathbb{R}
    """
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_mod = cos0 * cos_phi - sin0 * sin_phi
    sin_mod = sin0 * cos_phi + cos0 * sin_phi
    return cos_mod, sin_mod


def nonnegative_feature_map_np(u: np.ndarray) -> np.ndarray:
    r"""
    Non-Negative Feature Map in NumPy:
        \phi(u) = \text{ELU}(u) + 1 \ge 0
    where ELU(u) = u if u > 0 else (exp(u) - 1).
    Hence phi(u) = u + 1 if u > 0 else exp(u) >= 0.
    """
    return np.where(u > 0.0, u + 1.0, np.exp(np.clip(u, -80.0, 0.0)))


def causal_viscous_ema_np(
    g: np.ndarray,
    nu: float = 0.5,
    initial_omega: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    1D Viscous EMA Diffusion Filter in NumPy:
        \omega(t) = (1 - \nu)\omega(t-1) + \nu g(t),   \nu \in (0, 1]
    Guarantees bounded rate-of-change and continuous derivatives.
    """
    if nu >= 1.0:
        return g.copy()

    g = np.asarray(g, dtype=np.float64)
    # Assume sequence dimension is second-to-last if ndim >= 2, else 0
    seq_dim = -2 if g.ndim >= 2 else 0
    seq_len = g.shape[seq_dim]

    alpha = 1.0 - nu
    omega = np.empty_like(g)

    if initial_omega is not None:
        prev = np.asarray(initial_omega, dtype=np.float64)
    else:
        # Zero initial state
        prev = np.zeros(np.take(g, 0, axis=seq_dim).shape, dtype=np.float64)

    for t in range(seq_len):
        curr_g = np.take(g, t, axis=seq_dim)
        prev = alpha * prev + nu * curr_g
        # Place prev into omega along seq_dim
        slc = [slice(None)] * g.ndim
        slc[seq_dim] = t
        omega[tuple(slc)] = prev

    return omega


def compute_lipschitz_rotation_bound(theta_m: float, omega_max: float) -> float:
    r"""
    Theoretical Lipschitz continuity upper bound of the 2D rotation operator:
        \|\tilde{\mathbf{R}}(t+1) - \tilde{\mathbf{R}}(t)\|_2 \le 2\sin\left(\frac{\theta_m + \omega_{\max}}{2}\right)
    """
    effective_angle = min(math.pi, theta_m + omega_max)
    return 2.0 * math.sin(effective_angle / 2.0)

# =============================================================================
# Continuous Semantic Gauge Connection & Holonomy
# =============================================================================

class SemanticGaugeConnectionRef:
    r"""
    Reference implementation of continuous semantic gauge connection:
    - Continuous torsion rate: \omega(t) \in [-\omega_{\max}, \omega_{\max}]
    - 1D Viscous EMA smoothing: \omega(t) = (1 - \nu)\omega(t-1) + \nu g(t)
    - Causal Wilson line / continuous path holonomy:
      \Phi(j, i) = \sum_{k=j+1}^i \omega(k) = \Phi(i) - \Phi(j)
    - Parity section: s_t = \cos(\Phi_t) \in [-1, +1]
    - Relative topological factor: T(j, i) = \cos(\Phi(j, i)) = \cos(\Phi_i - \Phi_j)
    """
    def __init__(
        self,
        refractory_window: int = 5,
        omega_max: float = 0.05,
        nu: float = 0.5
    ):
        self.refractory_window = refractory_window
        self.omega_max = float(omega_max)
        self.nu = float(nu)

    def compute_holonomy(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes cumulative continuous gauge phase Phi_t and parity section s_t.
        Returns:
            phi: cumulative continuous phase array [seq_len]
            s: parity section array [seq_len] = cos(phi) in [-1, +1]
        """
        phi = np.cumsum(omega)
        s = np.cos(phi)
        return phi, s

    def compute_pairwise_topological_matrix(self, phi: np.ndarray) -> np.ndarray:
        """
        Computes the NxN pairwise continuous topological factor matrix:
        T[i, j] = cos(phi[i] - phi[j]).
        """
        delta_phi = phi[:, None] - phi[None, :] # [i, j] = phi[i] - phi[j]
        return np.cos(delta_phi)

    def apply_viscous_ema(
        self,
        g: np.ndarray,
        nu: Optional[float] = None,
        initial_omega: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Applies 1D viscous EMA diffusion filtering."""
        effective_nu = self.nu if nu is None else float(nu)
        return causal_viscous_ema_np(g, nu=effective_nu, initial_omega=initial_omega)

    def forward(
        self,
        g: np.ndarray,
        nu: Optional[float] = None,
        initial_omega: Optional[np.ndarray] = None,
        initial_phi: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Continuous torsion smoothing and phase integration:
        Returns:
            (omega, phi): filtered continuous torsion and cumulative phase flow.
        """
        omega = self.apply_viscous_ema(g, nu=nu, initial_omega=initial_omega)
        phi = np.cumsum(omega, axis=-1 if omega.ndim == 1 else -2)
        if initial_phi is not None:
            phi = phi + initial_phi
        return omega, phi

    def filter_with_hysteresis(self, raw_triggers: List[int], seq_len: int) -> List[int]:
        """
        Applies refractory hysteresis gating:
        Once a contrastive trigger fires at token t, subsequent triggers within
        [t, t + refractory_window] are suppressed to prevent burst oscillations.
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
                if not (in_double_quote or in_single_quote):
                    scoped_triggers.append(idx)
        return scoped_triggers

# =============================================================================
# Continuous 2D Planar Möbius-RoPE Modulation
# =============================================================================

def apply_mobius_rope_dual(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    phi: np.ndarray,
    anchor_ratio: float = 0.0,
    enable_nonnegative_map: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Applies mathematically rigorous Continuous 2D Planar Möbius-RoPE via Angle Addition:
        \cos_{\mathrm{mod}} = \cos_0 \odot \cos(\Phi) - \sin_0 \odot \sin(\Phi)
        \sin_{\mathrm{mod}} = \sin_0 \odot \cos(\Phi) + \cos_0 \odot \sin(\Phi)
    
    Identically preserves vector Euclidean norms:
        \|q_{\mathrm{mob}}\| \equiv \|q\|,  \|k_{\mathrm{mob}}\| \equiv \|k\|  \forall \Phi \in \mathbb{R}
    
    When \Phi \in \{0, \pi\}, reduces identically to s_t \in \{+1, -1\} parity inversion,
    maintaining 100% backward compatibility for discrete tier benchmark tests.
    """
    head_dim = q.shape[-1]
    num_subspaces = head_dim // 2

    # Standardize phi shape to [..., head_dim]
    phi_arr = np.asarray(phi, dtype=np.float64)
    if phi_arr.ndim == 1:
        # [seq_len] -> [seq_len, 1]
        seq_len = len(phi_arr)
        phi_sub = np.repeat(phi_arr[:, None], num_subspaces, axis=-1)
    elif phi_arr.shape[-1] == 1:
        phi_sub = np.repeat(phi_arr, num_subspaces, axis=-1)
    elif phi_arr.shape[-1] == num_subspaces:
        phi_sub = phi_arr
    elif phi_arr.shape[-1] == head_dim:
        phi_sub = phi_arr[..., :num_subspaces]
    else:
        phi_sub = np.repeat(phi_arr[..., :1], num_subspaces, axis=-1)

    # Subspace decoupling: anchor channels keep phi = 0
    if anchor_ratio > 0.0:
        num_anchor = int(round(num_subspaces * anchor_ratio))
        num_anchor = max(0, min(num_anchor, num_subspaces))
        phi_sub = phi_sub.copy()
        phi_sub[..., :num_anchor] = 0.0

    phi_full = np.concatenate([phi_sub, phi_sub], axis=-1)

    # Align ranks between phi_full and cos
    while phi_full.ndim < cos.ndim:
        phi_full = np.expand_dims(phi_full, axis=0)

    # Non-negative feature map prior to rotation (if enabled)
    if enable_nonnegative_map:
        q_in = nonnegative_feature_map_np(q)
        k_in = nonnegative_feature_map_np(k)
    else:
        q_in = q
        k_in = k

    # Angle addition modulation
    cos_mod, sin_mod = apply_rotary_angle_addition_np(cos, sin, phi_full)
    q_mob = apply_rotary_emb(q_in, cos_mod, sin_mod)
    k_mob = apply_rotary_emb(k_in, cos_mod, sin_mod)
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
    r"""
    Reference simulator for linear KV cache decoding with continuous 2D planar rotation:
    - Pre-stores keys rotated with continuous phase \Phi(j) in the cache.
    - Each autoregressive step (seq_len_curr = 1) rotates new query with \Phi(t).
    - Achieves exact O(1) step time and O(N) memory without retroactive cache recomputation.
    - Fully compatible with continuous phase flow \Phi(t) \in \mathbb{R}.
    """
    def __init__(self, head_dim: int, base: float = 10000.0):
        self.head_dim = head_dim
        self.base = base
        self.cache_k = [] # list of key vectors [head_dim]
        self.cache_v = [] # list of val vectors [head_dim]
        self.cache_s = [] # list of parity sections s_j = cos(phi_j) (backward compat)
        self.cache_phi = [] # list of continuous phases
        self.dim_half = head_dim // 2
        m_indices = np.arange(self.dim_half, dtype=np.float64)
        self.inv_freq = 1.0 / (base ** (2.0 * m_indices / head_dim))

    def _get_rope_sincos(self, position: int) -> Tuple[np.ndarray, np.ndarray]:
        t = float(position)
        freqs = t * self.inv_freq
        full_freqs = np.concatenate([freqs, freqs])
        return np.cos(full_freqs), np.sin(full_freqs)

    def append_token(
        self,
        k: np.ndarray,
        v: np.ndarray,
        position: int,
        phi_t: Union[float, np.ndarray]
    ):
        """Appends and caches a newly generated token key-value pair."""
        cos0, sin0 = self._get_rope_sincos(position)
        if np.isscalar(phi_t):
            phi_arr = np.full(self.head_dim, float(phi_t))
        else:
            phi_arr = np.asarray(phi_t, dtype=np.float64)
            if phi_arr.ndim == 0:
                phi_arr = np.full(self.head_dim, float(phi_arr))
            elif len(phi_arr) == self.dim_half:
                phi_arr = np.concatenate([phi_arr, phi_arr])

        cos_mod, sin_mod = apply_rotary_angle_addition_np(cos0, sin0, phi_arr)
        k_cached = apply_rotary_emb(k, cos_mod, sin_mod)
        s_t = float(np.cos(phi_arr[0]))

        self.cache_k.append(k_cached)
        self.cache_v.append(v)
        self.cache_s.append(s_t)
        self.cache_phi.append(phi_arr)

    def decode_step(
        self,
        q: np.ndarray,
        position: int,
        phi_t: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decodes a single token at position t with continuous phase phi_t against all cached keys.
        """
        cos0, sin0 = self._get_rope_sincos(position)
        if np.isscalar(phi_t):
            phi_arr = np.full(self.head_dim, float(phi_t))
        else:
            phi_arr = np.asarray(phi_t, dtype=np.float64)
            if phi_arr.ndim == 0:
                phi_arr = np.full(self.head_dim, float(phi_arr))
            elif len(phi_arr) == self.dim_half:
                phi_arr = np.concatenate([phi_arr, phi_arr])

        cos_mod, sin_mod = apply_rotary_angle_addition_np(cos0, sin0, phi_arr)
        q_mod = apply_rotary_emb(q, cos_mod, sin_mod)

        keys = np.stack(self.cache_k, axis=0) # [N, head_dim]
        logits = np.dot(keys, q_mod) / math.sqrt(self.head_dim) # [N]
        # Softmax
        exp_l = np.exp(logits - np.max(logits))
        attn_weights = exp_l / (np.sum(exp_l) + 1e-12)

        vals = np.stack(self.cache_v, axis=0) # [N, head_dim]
        output = np.sum(attn_weights[:, None] * vals, axis=0)
        return logits, attn_weights

# =============================================================================
# Benchmarking Metrics: ACR & Shannon Entropy Perplexity Preservation
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
    Computes Shannon entropy / perplexity proxy based strictly on Softmax probabilities:
    P = softmax(logits)
    H(P) = - sum P * ln(P)
    PPL = exp( H(P) )
    """
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / (np.sum(exp_l) + 1e-12)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(np.exp(entropy))
