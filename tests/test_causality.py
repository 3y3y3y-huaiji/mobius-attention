# -*- coding: utf-8 -*-
r"""
Strict Causality & Non-Anticipation Test Suite for SG-Möbius Attention
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Validates that:
1. Strict Pre-Trigger Causality:
   For any causal prompt with a contrastive trigger at position t^*:
       \omega(t) \equiv 0  and  \Phi(t) \equiv 0  \forall t < t^*
2. Zero Future Token Leakage:
   Modifying or perturbing tokens at positions t > t_eval has zero mathematical
   or numerical impact on (\omega(\tau), \Phi(\tau)) for \tau \le t_eval.
3. Strict Causal Gradient Flow (Autograd Lower-Triangularity):
   \frac{\partial \Phi(t)}{\partial h(\tau)} \equiv 0  and  \frac{\partial \omega(t)}{\partial h(\tau)} \equiv 0  \forall \tau > t
4. Two-Stage Parallel Associative Scan Lower-Triangular Structure:
   Prefix scan dependencies are strictly causal (submatrix upper triangle is identically 0).
"""

import pytest
import torch

from src.gauge_predictor import (
    SemanticGaugeConnection,
    causal_viscous_ema,
    parallel_affine_scan,
    two_stage_associative_scan,
)
from src.mobius_attention import SGMobiusRotaryEmbedding


class TestStrictCausality:
    """Verifies strict causal non-anticipation across all gauge and phase modules."""

    def test_strict_pre_trigger_causality(self):
        r"""
        Asserts that before the trigger index t^*, both \omega(t) and \Phi(t)
        are identically zero down to machine precision.
        """
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 100
        trigger_pos = 45
        hidden_dim = 16

        conn = SemanticGaugeConnection(
            hidden_dim=hidden_dim,
            out_dim=1,
            omega_max=0.08,
            nu=0.4,
            eta=0.0,
            bias=False
        )

        # Create input hidden states: strictly zero before trigger, active from trigger onward
        h = torch.zeros(batch_size, seq_len, hidden_dim, dtype=torch.float64)
        h[:, trigger_pos:, :] = torch.randn(batch_size, seq_len - trigger_pos, hidden_dim, dtype=torch.float64)

        omega, phi = conn(h)

        # Assert strict pre-trigger zero state
        pre_omega = omega[:, :trigger_pos, :]
        pre_phi = phi[:, :trigger_pos, :]

        assert (pre_omega == 0.0).all(), f"Pre-trigger omega non-zero! Max: {pre_omega.abs().max().item()}"
        assert (pre_phi == 0.0).all(), f"Pre-trigger phi non-zero! Max: {pre_phi.abs().max().item()}"

        # Assert post-trigger activation is non-zero
        post_omega = omega[:, trigger_pos:, :]
        assert post_omega.abs().max().item() > 0.0, "Post-trigger omega failed to activate!"

    def test_zero_future_leakage(self):
        r"""
        Asserts that mutating future tokens at \tau > t_eval does not change
        past representations at \tau \le t_eval.
        """
        torch.manual_seed(101)
        batch_size = 2
        seq_len = 64
        t_eval = 30
        dim = 8

        g_orig = torch.randn(batch_size, seq_len, dim, dtype=torch.float64) * 0.05
        omega_orig, phi_orig = two_stage_associative_scan(g_orig, nu=0.35, eta=0.05)

        # Mutate all future tokens with extreme adversarial noise
        g_mutated = g_orig.clone()
        g_mutated[:, t_eval + 1:, :] = torch.randn(batch_size, seq_len - t_eval - 1, dim, dtype=torch.float64) * 10.0

        omega_mutated, phi_mutated = two_stage_associative_scan(g_mutated, nu=0.35, eta=0.05)

        # Verify past states [0 .. t_eval] are strictly invariant
        past_omega_err = (omega_orig[:, :t_eval + 1, :] - omega_mutated[:, :t_eval + 1, :]).abs().max().item()
        past_phi_err = (phi_orig[:, :t_eval + 1, :] - phi_mutated[:, :t_eval + 1, :]).abs().max().item()

        assert past_omega_err == 0.0, f"Future leakage in omega! Past max error: {past_omega_err}"
        assert past_phi_err == 0.0, f"Future leakage in phi! Past max error: {past_phi_err}"

        # Verify future states actually changed
        future_omega_diff = (omega_orig[:, t_eval + 1:, :] - omega_mutated[:, t_eval + 1:, :]).abs().max().item()
        assert future_omega_diff > 0.0, "Future states should have changed under mutation"

    def test_causal_gradient_flow_autograd(self):
        r"""
        Verifies via PyTorch Autograd that \frac{\partial \Phi(t_0)}{\partial g(\tau)} \equiv 0 \forall \tau > t_0.
        No backward gradient can propagate from the past into the future.
        """
        torch.manual_seed(202)
        seq_len = 32
        dim = 4
        t_target = 15

        g = torch.randn(1, seq_len, dim, dtype=torch.float64, requires_grad=True)

        omega, phi = two_stage_associative_scan(g, nu=0.4, eta=0.03)

        # Target loss strictly depends on position t_target
        loss = phi[0, t_target, :].sum()
        loss.backward()

        grad = g.grad[0]  # [seq_len, dim]

        # Past and current gradients can be non-zero
        past_grad_norm = grad[:t_target + 1, :].abs().max().item()
        assert past_grad_norm > 0.0, "Gradient with respect to causal past should be non-zero"

        # Future gradients must be identically zero
        future_grad_norm = grad[t_target + 1:, :].abs().max().item()
        assert future_grad_norm == 0.0, f"Acausal gradient leakage! Future grad norm: {future_grad_norm}"

    def test_lower_triangular_jacobian_structure(self):
        r"""
        Explicitly computes the full Jacobian J_{ij} = \frac{\partial \omega(i)}{\partial g(j)}
        and verifies it is strictly lower triangular (J_{ij} = 0 for j > i).
        """
        torch.manual_seed(303)
        seq_len = 16
        g = torch.randn(seq_len, dtype=torch.float64, requires_grad=True)

        # Scalar sequence scan
        a = torch.full_like(g, 0.6)
        b = 0.4 * g

        _, omega = parallel_affine_scan(a, b, dim=0)

        # Build Jacobian row by row
        jacobian = torch.zeros(seq_len, seq_len, dtype=torch.float64)
        for i in range(seq_len):
            grad_out = torch.zeros(seq_len, dtype=torch.float64)
            grad_out[i] = 1.0
            grad_g = torch.autograd.grad(omega, g, grad_outputs=grad_out, retain_graph=True)[0]
            jacobian[i] = grad_g

        # Upper triangular elements (excluding diagonal) must be zero
        upper_triangle = torch.triu(jacobian, diagonal=1)
        max_upper = upper_triangle.abs().max().item()

        assert max_upper == 0.0, f"Non-causal upper triangular elements in Jacobian: {max_upper}"

        # Diagonal elements must equal \nu (0.4)
        diag = torch.diag(jacobian)
        assert torch.allclose(diag, torch.full_like(diag, 0.4), atol=1e-12), "Diagonal elements must equal nu"

    def test_rotary_embedding_causal_preservation(self):
        r"""
        Verifies that SGMobiusRotaryEmbedding preserves strict causality:
        Keys pre-rotated at step j are invariant to future tokens.
        """
        torch.manual_seed(404)
        head_dim = 16
        seq_len = 20
        cutoff = 10

        rotary = SGMobiusRotaryEmbedding(head_dim=head_dim, anchor_ratio=0.5)

        x1 = torch.randn(1, cutoff, head_dim, dtype=torch.float64)
        pos1 = torch.arange(cutoff, dtype=torch.long).unsqueeze(0)
        phi1 = torch.cumsum(torch.full((1, cutoff, 1), 0.02, dtype=torch.float64), dim=1)

        cos1, sin1 = rotary(x1, position_ids=pos1, phi=phi1)

        # Longer sequence containing x1 as prefix
        x2 = torch.cat([x1, torch.randn(1, seq_len - cutoff, head_dim, dtype=torch.float64)], dim=1)
        pos2 = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        phi2 = torch.cat([phi1, phi1[:, -1:, :] + torch.cumsum(torch.full((1, seq_len - cutoff, 1), 0.02, dtype=torch.float64), dim=1)], dim=1)

        cos2, sin2 = rotary(x2, position_ids=pos2, phi=phi2)

        # Compare prefix
        assert torch.allclose(cos1, cos2[:, :cutoff, :], atol=1e-12), "Rotary cos prefix changed under longer sequence!"
        assert torch.allclose(sin1, sin2[:, :cutoff, :], atol=1e-12), "Rotary sin prefix changed under longer sequence!"
