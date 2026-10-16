# -*- coding: utf-8 -*-
r"""
Unit Tests for SG-Möbius-RoPE Architectural Refactoring (R3)
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Covers:
1. Subspace mask specification: \mathcal{S}_{\text{anchor}} (0) vs \mathcal{S}_{\text{mob}} (1).
2. Phase-locking attractor dynamics: -\eta \sin(2\Phi) in viscous flow with Lipschitz clamp.
3. Pre-fused base angle table lookup equivalence with eager angle addition.
4. Numerical parity between batch parallel prefill and autoregressive token-by-token decode.
5. Backward compatibility aliases and methods.
"""

import math
import pytest
import torch

from src import (
    ContinuousMobiusRoPEModulator,
    HysteresisSemanticGaugePredictor,
    SGMobiusRotaryEmbedding,
    SemanticGaugeConnection,
    SemanticGaugeMobiusRoPE,
    apply_rotary_angle_addition,
    causal_viscous_ema,
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class TestSubspaceMask:
    """Tests for subspace_mask (0=anchor, 1=mob) specification and decoupling."""

    def test_custom_subspace_mask_tensor_initialization(self):
        head_dim = 16
        num_subspaces = head_dim // 2
        # Explicit mask: 0=anchor, 1=mob (subspaces 0, 2 are anchor; 1, 3 are mob)
        custom_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int64)

        modulator = SGMobiusRotaryEmbedding(
            head_dim=head_dim,
            subspace_mask=custom_mask
        )

        mask_out = modulator.get_subspace_mask(num_subspaces, device=torch.device("cpu"))
        assert mask_out.dtype == torch.bool
        assert mask_out.tolist() == [False, True, False, True, False, True, False, True]

    def test_anchor_zero_modulation_and_mob_pi_inversion(self):
        """Verifies that anchor channels undergo zero modulation (Delta=0),
        while mob channels undergo exact pi rotation when Phi=pi."""
        head_dim = 8
        num_subspaces = head_dim // 2
        # First 2 channels are anchor (0), last 2 channels are mob (1)
        mask = torch.tensor([0, 0, 1, 1], dtype=torch.float32)

        modulator = SGMobiusRotaryEmbedding(
            head_dim=head_dim,
            subspace_mask=mask,
            pre_fuse_angles=True
        )

        x = torch.ones(1, 1, head_dim, dtype=torch.float64)
        pos = torch.zeros(1, 1, dtype=torch.long)

        # Baseline angles with phi = 0
        cos_base, sin_base = modulator(x, position_ids=pos, phi=None)

        # Modulated angles with phi = pi
        phi_pi = torch.tensor([[[math.pi]]], dtype=torch.float64)
        cos_mod, sin_mod = modulator(x, position_ids=pos, phi=phi_pi)

        # Anchor channels (subspaces 0 and 1, appearing at indices 0, 1, 4, 5 in head_dim)
        anchor_indices = [0, 1, 4, 5]
        for idx in anchor_indices:
            assert torch.allclose(cos_mod[..., idx], cos_base[..., idx], atol=1e-12), \
                f"Anchor channel {idx} was unexpectedly modified by phase!"
            assert torch.allclose(sin_mod[..., idx], sin_base[..., idx], atol=1e-12), \
                f"Anchor channel {idx} was unexpectedly modified by phase!"

        # Mob channels (subspaces 2 and 3, appearing at indices 2, 3, 6, 7 in head_dim)
        mob_indices = [2, 3, 6, 7]
        for idx in mob_indices:
            # At phi = pi, cos(th + pi) = -cos(th), sin(th + pi) = -sin(th)
            assert torch.allclose(cos_mod[..., idx], -cos_base[..., idx], atol=1e-12), \
                f"Mob channel {idx} failed to invert sign at phi=pi!"
            assert torch.allclose(sin_mod[..., idx], -sin_base[..., idx], atol=1e-12), \
                f"Mob channel {idx} failed to invert sign at phi=pi!"

    def test_full_head_dim_mask_auto_slice(self):
        """Verifies that a mask provided with length == head_dim is safely truncated to num_subspaces."""
        head_dim = 16
        full_mask = torch.cat([torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4)])
        modulator = SGMobiusRotaryEmbedding(head_dim=head_dim, subspace_mask=full_mask)
        mask_out = modulator.get_subspace_mask(8, device=torch.device("cpu"))
        assert mask_out.numel() == 8


class TestPhaseLockingAttractorDynamics:
    """Tests for phase-locking attractor -eta*sin(2*Phi) in viscous flow."""

    def test_attractor_fixed_points(self):
        """Verifies that Phi in {0, pi, 2pi} are exact fixed points where attractor force is zero."""
        eta = 0.05
        for target_phi in [0.0, math.pi, 2.0 * math.pi, -math.pi]:
            phi_t = torch.tensor([[[target_phi]]], dtype=torch.float64)
            force = -eta * torch.sin(2.0 * phi_t)
            assert torch.allclose(force, torch.zeros_like(force), atol=1e-14), \
                f"Attractor force non-zero at equilibrium phi={target_phi}: {force.item()}"

    def test_attractor_restoring_force_towards_pi(self):
        """Verifies that perturbations near pi experience restoring forces pulling towards pi."""
        eta = 0.05

        # Perturbation to the right of pi (pi + 0.1): force should be negative (pull left towards pi)
        phi_right = torch.tensor([[[math.pi + 0.1]]], dtype=torch.float64)
        force_right = -eta * torch.sin(2.0 * phi_right)
        assert force_right.item() < 0.0, "Force to the right of pi must be negative"

        # Perturbation to the left of pi (pi - 0.1): force should be positive (pull right towards pi)
        phi_left = torch.tensor([[[math.pi - 0.1]]], dtype=torch.float64)
        force_left = -eta * torch.sin(2.0 * phi_left)
        assert force_left.item() > 0.0, "Force to the left of pi must be positive"

    def test_attractor_convergence_in_semantic_gauge_connection(self):
        """Simulates 60 steps of autonomous viscous flow starting near pi and verifies convergence."""
        torch.manual_seed(42)
        hidden_dim = 16
        conn = SemanticGaugeConnection(
            hidden_dim=hidden_dim,
            out_dim=1,
            omega_max=0.10,
            nu=0.25,
            eta=0.08,
            bias=False
        )

        # Zero input tokens (no external semantic driving force)
        zero_x = torch.zeros(1, 60, hidden_dim, dtype=torch.float64)
        initial_phi = torch.tensor([[math.pi + 0.25]], dtype=torch.float64)

        omega, phi = conn(zero_x, initial_phi=initial_phi)
        final_phi = phi[0, -1, 0].item()

        # Should have converged smoothly toward pi
        assert abs(final_phi - math.pi) < 0.01, f"Failed to converge to pi: final_phi={final_phi}"

    def test_lipschitz_clamp_with_large_attractor(self):
        """Verifies that |omega_t| <= omega_max is strictly preserved even under strong attractor coupling."""
        omega_max = 0.05
        conn = SemanticGaugeConnection(
            hidden_dim=8,
            omega_max=omega_max,
            nu=0.5,
            eta=10.0  # Deliberately large eta to stress test Lipschitz clamping
        )
        x = torch.randn(2, 50, 8, dtype=torch.float64)
        omega, _ = conn(x)
        assert (omega.abs() <= omega_max + 1e-12).all(), "Lipschitz envelope clamp was violated!"


class TestPreFusedAngleTableEquivalence:
    """Tests equivalence between pre-fused base angle tables and eager angle addition."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_prefused_vs_eager_modulate_rotary(self, dtype):
        torch.manual_seed(77)
        head_dim = 32
        seq_len = 16
        batch_size = 2

        modulator_fused = SGMobiusRotaryEmbedding(head_dim=head_dim, anchor_ratio=0.5, pre_fuse_angles=True)
        modulator_eager = SGMobiusRotaryEmbedding(head_dim=head_dim, anchor_ratio=0.5, pre_fuse_angles=False)

        x = torch.randn(batch_size, seq_len, head_dim, dtype=dtype)
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        phi = torch.randn(batch_size, seq_len, 1, dtype=dtype)

        cos_fused, sin_fused = modulator_fused(x, position_ids=pos, phi=phi)
        cos_eager, sin_eager = modulator_eager(x, position_ids=pos, phi=phi)

        atol = 1e-5 if dtype == torch.float32 else 1e-12
        assert torch.allclose(cos_fused, cos_eager, atol=atol), "Pre-fused cos diverged from eager cos!"
        assert torch.allclose(sin_fused, sin_eager, atol=atol), "Pre-fused sin diverged from eager sin!"

    def test_rotated_vectors_numerical_identity(self):
        """Verifies that apply_rotary_pos_emb with pre-fused angles gives identical rotated vectors."""
        torch.manual_seed(88)
        head_dim = 64
        seq_len = 32
        batch_size = 2

        mod_fused = SGMobiusRotaryEmbedding(head_dim=head_dim, anchor_ratio=0.25, pre_fuse_angles=True)
        mod_eager = SGMobiusRotaryEmbedding(head_dim=head_dim, anchor_ratio=0.25, pre_fuse_angles=False)

        q = torch.randn(batch_size, 1, seq_len, head_dim, dtype=torch.float64)
        k = torch.randn(batch_size, 1, seq_len, head_dim, dtype=torch.float64)
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        phi = torch.randn(batch_size, seq_len, head_dim // 2, dtype=torch.float64)

        cos_f, sin_f = mod_fused(q, position_ids=pos, phi=phi)
        cos_e, sin_e = mod_eager(q, position_ids=pos, phi=phi)

        q_rot_f = apply_rotary_pos_emb(q, cos_f, sin_f)
        q_rot_e = apply_rotary_pos_emb(q, cos_e, sin_e)
        k_rot_f = apply_rotary_pos_emb(k, cos_f, sin_f)
        k_rot_e = apply_rotary_pos_emb(k, cos_e, sin_e)

        assert torch.allclose(q_rot_f, q_rot_e, atol=1e-12), "Rotated queries differ between pre-fused and eager!"
        assert torch.allclose(k_rot_f, k_rot_e, atol=1e-12), "Rotated keys differ between pre-fused and eager!"


class TestPrefillVsDecodeParityWithAttractor:
    """Tests numerical parity between batch parallel prefill and step-by-step decode under attractor."""

    def test_semantic_gauge_connection_prefill_vs_decode_parity(self):
        torch.manual_seed(99)
        batch_size = 2
        seq_len = 32
        hidden_dim = 16

        conn = SemanticGaugeConnection(
            hidden_dim=hidden_dim,
            out_dim=1,
            omega_max=0.08,
            nu=0.35,
            eta=0.04
        )

        x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float64)

        # 1. Parallel Prefill
        omega_prefill, phi_prefill = conn(x)

        # 2. Token-by-Token Decode
        conn.reset_state()
        omega_decode = torch.empty_like(omega_prefill)
        phi_decode = torch.empty_like(phi_prefill)

        for t in range(seq_len):
            x_t = x[:, t:t+1, :]
            w_t, p_t = conn.step(x_t)
            omega_decode[:, t:t+1, :] = w_t
            phi_decode[:, t:t+1, :] = p_t

        max_err_omega = (omega_prefill - omega_decode).abs().max().item()
        max_err_phi = (phi_prefill - phi_decode).abs().max().item()

        assert max_err_omega < 1e-12, f"Omega prefill vs decode discrepancy: {max_err_omega:.2e}"
        assert max_err_phi < 1e-12, f"Phi prefill vs decode discrepancy: {max_err_phi:.2e}"

    def test_sg_mobius_rotary_prefill_vs_decode_parity(self):
        torch.manual_seed(101)
        batch_size = 2
        seq_len = 32
        head_dim = 32

        modulator = SGMobiusRotaryEmbedding(
            head_dim=head_dim,
            anchor_ratio=0.5,
            pre_fuse_angles=True
        )

        phi = torch.randn(batch_size, seq_len, 1, dtype=torch.float64)
        x = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float64)
        pos_prefill = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        # 1. Prefill
        cos_prefill, sin_prefill = modulator(x, position_ids=pos_prefill, phi=phi)

        # 2. Decode
        cos_decode = torch.empty_like(cos_prefill)
        sin_decode = torch.empty_like(sin_prefill)

        for t in range(seq_len):
            pos_t = torch.tensor([[t]], dtype=torch.long).expand(batch_size, 1)
            phi_t = phi[:, t:t+1, :]
            x_t = x[:, t:t+1, :]
            c_t, s_t = modulator(x_t, position_ids=pos_t, phi=phi_t)
            cos_decode[:, t:t+1, :] = c_t
            sin_decode[:, t:t+1, :] = s_t

        max_err_cos = (cos_prefill - cos_decode).abs().max().item()
        max_err_sin = (sin_prefill - sin_decode).abs().max().item()

        assert max_err_cos < 1e-12, f"Cos prefill vs decode discrepancy: {max_err_cos:.2e}"
        assert max_err_sin < 1e-12, f"Sin prefill vs decode discrepancy: {max_err_sin:.2e}"


class TestBackwardCompatibility:
    """Verifies that all aliases, methods, and predictors preserve backward compatibility."""

    def test_class_aliases(self):
        assert ContinuousMobiusRoPEModulator is SGMobiusRotaryEmbedding
        assert SemanticGaugeMobiusRoPE is SGMobiusRotaryEmbedding
        mod = SGMobiusRotaryEmbedding(head_dim=16)
        assert isinstance(mod, ContinuousMobiusRoPEModulator)
        assert isinstance(mod, SemanticGaugeMobiusRoPE)

    def test_hysteresis_predictor_gating(self):
        predictor = HysteresisSemanticGaugePredictor(hidden_dim=16, refractory_window=4)
        raw_triggers = [2, 3, 5, 10, 11, 15]
        filtered = predictor.filter_with_hysteresis(raw_triggers, seq_len=20)
        # 2 fires -> 3, 5 suppressed (within 2+4=6) -> 10 fires -> 11 suppressed -> 15 fires
        assert filtered == [2, 10, 15]

    def test_quotation_scope_filtering(self):
        predictor = HysteresisSemanticGaugePredictor(hidden_dim=16)
        tokens = ["The", "report", "said", '"', "not", "true", '"', "but", "valid"]
        raw = [4, 7]  # "not" at idx 4 is inside quotes; "but" at idx 7 is outside
        scoped = predictor.filter_with_quotation_scope(tokens, raw)
        assert scoped == [7]
