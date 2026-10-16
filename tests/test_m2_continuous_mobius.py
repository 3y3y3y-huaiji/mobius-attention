"""
Unit tests and mathematical verification for Milestone 2 Continuous Möbius RoPE.
Covers:
1. Module instantiation and sample forward execution.
2. 2D planar rotation exact angle addition vs SO(2) rotation matrix (< 10^-6).
3. Euclidean vector norm preservation (||R(theta + phi) v|| == ||v||).
4. Subspace decoupling (anchor vs evolution channels).
5. 1D viscous EMA diffusion (prefill vs token-by-token decode < 10^-5).
6. Non-negative feature map (phi(u) >= 0 and baseline non-negativity).
7. O(1) causal KV-cache invariance (historical key independence).
8. Clean injection into mock Transformer attention modules.
9. Backward compatibility alias (SemanticGaugeMobiusRoPE).
"""

import math
import torch
import torch.nn as nn
import pytest

from src.mobius_attention import (
    ContinuousMobiusRoPEModulator,
    SemanticGaugeConnection,
    SemanticGaugeMobiusRoPE,
    inject_mobius_attention,
    causal_viscous_ema,
    nonnegative_feature_map,
    apply_rotary_angle_addition,
    set_model_holonomy_phase,
    clear_model_holonomy_phase
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def test_instantiation_and_sample_forward():
    """Verify clean instantiation and forward pass on random tensors."""
    hidden_dim = 128
    gauge = SemanticGaugeConnection(hidden_dim=hidden_dim, out_dim=1, omega_max=0.05, nu=0.5)
    modulator = ContinuousMobiusRoPEModulator(head_dim=64, anchor_ratio=0.5)

    x = torch.randn(2, 16, hidden_dim)
    omega, phi = gauge(x)
    assert omega.shape == (2, 16, 1)
    assert phi.shape == (2, 16, 1)
    assert (omega.abs() <= 0.05 + 1e-6).all()

    # Forward through modulator
    q = torch.randn(2, 16, 64)
    pos_ids = torch.arange(16).unsqueeze(0).expand(2, -1)
    modulator.set_holonomy_phase(phi)
    cos, sin = modulator(q, position_ids=pos_ids)

    assert cos.shape == (2, 16, 64)
    assert sin.shape == (2, 16, 64)
    assert not torch.isnan(cos).any()
    assert not torch.isnan(sin).any()


def test_angle_addition_matches_2d_rotation_matrix():
    """Verify angle addition matches true 2D SO(2) rotation matrix to < 10^-6."""
    torch.manual_seed(42)
    dim = 64
    num_subspaces = dim // 2
    x = torch.randn(4, 32, dim, dtype=torch.float64)
    theta = torch.randn(1, 32, num_subspaces, dtype=torch.float64)
    phi = torch.randn(4, 32, num_subspaces, dtype=torch.float64)

    # 1. Full 2D planar rotation matrix per subspace:
    # R(theta + phi) * [x1, x2]^T = [x1 cos(theta+phi) - x2 sin(theta+phi), x2 cos(theta+phi) + x1 sin(theta+phi)]
    total_angle = theta + phi
    cos_tot = total_angle.cos()
    sin_tot = total_angle.sin()

    x1 = x[..., :num_subspaces]
    x2 = x[..., num_subspaces:]
    rot_x1 = x1 * cos_tot - x2 * sin_tot
    rot_x2 = x2 * cos_tot + x1 * sin_tot
    expected_rot = torch.cat([rot_x1, rot_x2], dim=-1)

    # 2. Angle addition from (cos0, sin0) and phi
    cos0 = torch.cat([theta.cos(), theta.cos()], dim=-1)
    sin0 = torch.cat([theta.sin(), theta.sin()], dim=-1)
    phi_cat = torch.cat([phi, phi], dim=-1)

    cos_mod, sin_mod = apply_rotary_angle_addition(cos0, sin0, phi_cat)
    actual_rot = (x * cos_mod) + (rotate_half(x) * sin_mod)

    max_err = (expected_rot - actual_rot).abs().max().item()
    assert max_err < 1e-12, f"Angle addition error {max_err} exceeds 1e-12 tolerance!"


def test_norm_preservation():
    """Verify 2D Euclidean norm is preserved identically for any continuous phase."""
    torch.manual_seed(101)
    dim = 64
    x = torch.randn(4, 20, dim, dtype=torch.float64)
    orig_norm = x.norm(dim=-1)

    modulator = ContinuousMobiusRoPEModulator(head_dim=dim, anchor_ratio=0.5)
    # Test across wide range of phase values including pi/2, pi, 3pi/2, 10pi
    phi_values = torch.tensor([0.0, math.pi/4, math.pi/2, math.pi, 1.5*math.pi, 2*math.pi, 10*math.pi], dtype=torch.float64)

    for phi_val in phi_values:
        phi = torch.full((4, 20, 1), phi_val.item(), dtype=torch.float64)
        cos, sin = modulator(x, phi=phi)
        x_rot = (x * cos) + (rotate_half(x) * sin)
        rot_norm = x_rot.norm(dim=-1)
        max_norm_diff = (orig_norm - rot_norm).abs().max().item()
        assert max_norm_diff < 1e-12, f"Norm changed by {max_norm_diff} at phi={phi_val.item()}"


def test_subspace_decoupling():
    """Verify anchor channels retain exact standard RoPE while evolution channels accumulate phase."""
    torch.manual_seed(202)
    dim = 64
    num_subspaces = dim // 2
    anchor_ratio = 0.5
    num_anchor = int(round(num_subspaces * anchor_ratio))

    modulator = ContinuousMobiusRoPEModulator(
        head_dim=dim,
        anchor_ratio=anchor_ratio,
        anchor_strategy="high_freq"
    )

    pos_ids = torch.arange(10).unsqueeze(0)
    # 1. Base standard RoPE (no phi)
    cos_base, sin_base = modulator(torch.randn(1, 10, dim), position_ids=pos_ids)

    # 2. Modulated with large phase
    phi = torch.full((1, 10, 1), math.pi)
    cos_mod, sin_mod = modulator(torch.randn(1, 10, dim), position_ids=pos_ids, phi=phi)

    # Check anchor subspaces (first num_anchor channels of each half)
    # First half anchor
    assert torch.allclose(cos_base[..., :num_anchor], cos_mod[..., :num_anchor], atol=1e-7)
    assert torch.allclose(sin_base[..., :num_anchor], sin_mod[..., :num_anchor], atol=1e-7)
    # Second half anchor
    assert torch.allclose(cos_base[..., num_subspaces:num_subspaces + num_anchor],
                          cos_mod[..., num_subspaces:num_subspaces + num_anchor], atol=1e-7)
    assert torch.allclose(sin_base[..., num_subspaces:num_subspaces + num_anchor],
                          sin_mod[..., num_subspaces:num_subspaces + num_anchor], atol=1e-7)

    # Check evolution subspaces (should be inverted at phi = pi: cos -> -cos, sin -> -sin)
    # Because cos(theta + pi) = -cos(theta), sin(theta + pi) = -sin(theta)
    assert torch.allclose(cos_mod[..., num_anchor:num_subspaces],
                          -cos_base[..., num_anchor:num_subspaces], atol=1e-6)
    assert torch.allclose(sin_mod[..., num_anchor:num_subspaces],
                          -sin_base[..., num_anchor:num_subspaces], atol=1e-6)


def test_viscous_ema_prefill_decoding_equivalence():
    """Verify parallel sequence prefill and token-by-token decoding match to < 10^-5."""
    torch.manual_seed(303)
    hidden_dim = 64
    seq_len = 128
    nu = 0.35
    omega_max = 0.05

    gauge = SemanticGaugeConnection(hidden_dim=hidden_dim, out_dim=4, omega_max=omega_max, nu=nu)
    x_full = torch.randn(2, seq_len, hidden_dim)

    # 1. Parallel sequence prefill
    omega_prefill, phi_prefill = gauge(x_full)

    # 2. Token-by-token stateful autoregressive decoding
    gauge.reset_state()
    omega_steps = []
    phi_steps = []
    for t in range(seq_len):
        x_t = x_full[:, t:t+1, :]
        o_t, p_t = gauge.step(x_t)
        omega_steps.append(o_t)
        phi_steps.append(p_t)

    omega_decode = torch.cat(omega_steps, dim=1)
    phi_decode = torch.cat(phi_steps, dim=1)

    max_err_omega = (omega_prefill - omega_decode).abs().max().item()
    max_err_phi = (phi_prefill - phi_decode).abs().max().item()

    assert max_err_omega < 1e-6, f"Omega prefill-decode error {max_err_omega} exceeds 1e-6"
    assert max_err_phi < 1e-5, f"Phi prefill-decode error {max_err_phi} exceeds 1e-5"


def test_nonnegative_feature_map():
    """Verify phi(u) = ELU(u) + 1 >= 0 and baseline dot products are non-negative."""
    # Test negative coordinates
    u = torch.linspace(-50, 50, 1000)
    phi_u = nonnegative_feature_map(u)
    assert (phi_u >= 0.0).all(), "phi(u) must be non-negative everywhere"
    assert (phi_u[u > -10.0] > 0.0).all(), "phi(u) must be strictly positive where precision permits"

    # Test baseline dot product between arbitrary vectors
    torch.manual_seed(404)
    q = torch.randn(10, 64)
    k = torch.randn(10, 64)
    q_pos = nonnegative_feature_map(q)
    k_pos = nonnegative_feature_map(k)
    dot_products = (q_pos * k_pos).sum(dim=-1)
    assert (dot_products >= 0.0).all(), "All non-negative feature map dot products must be non-negative!"


def test_causal_kv_cache_invariance():
    """Verify key embeddings at position j are strictly independent of future sequence length."""
    torch.manual_seed(505)
    hidden_dim = 64
    head_dim = 32
    modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5)
    gauge = SemanticGaugeConnection(hidden_dim=hidden_dim, out_dim=1, omega_max=0.05, nu=0.5)

    seq_short = 20
    seq_long = 50
    x_long = torch.randn(1, seq_long, hidden_dim)
    k_long = torch.randn(1, seq_long, head_dim)

    # Prefill short sequence (first 20 tokens)
    x_short = x_long[:, :seq_short]
    k_short = k_long[:, :seq_short]

    _, phi_short = gauge(x_short)
    _, phi_long = gauge(x_long)

    # Rotate keys in short sequence
    cos_short, sin_short = modulator(k_short, phi=phi_short)
    _, k_short_rot = apply_rotary_pos_emb(k_short, k_short, cos_short, sin_short)

    # Rotate keys in long sequence
    cos_long, sin_long = modulator(k_long, phi=phi_long)
    _, k_long_rot = apply_rotary_pos_emb(k_long, k_long, cos_long, sin_long)

    # Keys for positions 0..19 must be identical down to float precision!
    diff = (k_short_rot - k_long_rot[:, :seq_short]).abs().max().item()
    assert diff < 1e-6, f"KV cache invariance violated! Diff: {diff}"


def test_injection_and_backward_compatibility():
    """Verify injection into a mock Transformer model and backward compatibility aliases."""
    class MockRotary(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.dim = dim
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq)

        def forward(self, x, position_ids=None):
            seq_len = x.shape[-2]
            if position_ids is None:
                pos = torch.arange(seq_len, device=x.device).float()
            else:
                pos = position_ids.float().squeeze(0)
            freqs = torch.outer(pos, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            return emb.cos(), emb.sin()

    class MockAttention(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.rotary_emb = MockRotary(dim=dim)

    class MockTransformer(nn.Module):
        def __init__(self, num_layers=4, dim=64):
            super().__init__()
            self.layers = nn.ModuleList([MockAttention(dim=dim) for _ in range(num_layers)])

    model = MockTransformer(num_layers=4, dim=64)
    patched = inject_mobius_attention(model, anchor_ratio=0.5)
    assert patched == 4, f"Expected 4 patched layers, got {patched}"

    # Verify each layer is an instance of ContinuousMobiusRoPEModulator and SemanticGaugeMobiusRoPE
    for layer in model.layers:
        assert isinstance(layer.rotary_emb, ContinuousMobiusRoPEModulator)
        assert isinstance(layer.rotary_emb, SemanticGaugeMobiusRoPE)

    # Verify phase propagation
    phi = torch.randn(1, 10, 1)
    set_model_holonomy_phase(model, phi)
    for layer in model.layers:
        assert layer.rotary_emb.get_holonomy_phase() is not None

    clear_model_holonomy_phase(model)
    for layer in model.layers:
        assert layer.rotary_emb.get_holonomy_phase() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
