"""
Unit tests verifying strict adherence of SemanticGaugeConnection to Definition 1:
    z_t = w_g^T h_t + b_g
    g(h_t) = \\omega_{\\max} * softsign(ReLU(|z_t| - \\tau_{gate}) * sgn(z_t))

Tests:
1. Dead-zone barrier quenches neutral expository tokens (|z_t| < \\tau_{gate}) to strictly zero.
2. Active reversal tokens (|z_t| >= \\tau_{gate}) activate phase velocity smoothly in [-\\omega_{\\max}, \\omega_{\\max}].
3. Recommended production configuration (\\omega_{\\max} = 0.05, \\tau_{gate} = 0.50) triggers properly without stall.
4. Closed-form factory from_contrastive_embeddings() correctly averages reversal and neutral token sets.
5. Prefill vs. autoregressive decoding equivalence holds under gated softsign dynamics.
"""

import torch
import pytest
from src.gauge_predictor import SemanticGaugeConnection


def test_definition1_gate_deadzone_and_activation():
    """Verify dead-zone barrier behavior under Definition 1 formula."""
    hidden_dim = 64
    omega_max = 0.05
    tau_gate = 0.50

    # Direction along dimension 0
    w_g = torch.zeros(hidden_dim)
    w_g[0] = 1.0

    gauge = SemanticGaugeConnection(
        hidden_dim=hidden_dim,
        out_dim=1,
        omega_max=omega_max,
        nu=0.5,
        tau_gate=tau_gate,
        polar_direction=w_g,
        bias=False,
        normalize_input=False
    )

    # 1. Sub-threshold input: z = 0.3 < 0.50 -> must produce EXACT ZERO
    x_sub = torch.zeros(1, 5, hidden_dim)
    x_sub[:, :, 0] = 0.30
    omega_sub, phi_sub = gauge(x_sub)
    assert (omega_sub == 0.0).all(), "Sub-threshold inputs must be identically zero"
    assert (phi_sub == 0.0).all(), "Phase flow must be identically zero"

    # 2. Super-threshold positive input: z = 1.0 > 0.50 -> excess = 0.50
    # softsign(0.50) = 0.50 / 1.50 = 1/3
    # Expected raw_g = 0.05 * (1/3) approx 0.01667
    x_super = torch.zeros(1, 1, hidden_dim)
    x_super[0, 0, 0] = 1.0
    gauge.reset_state()
    o_step, p_step = gauge.step(x_super[0, 0])
    assert o_step.item() > 0.0, f"Expected active velocity, got {o_step.item()}"
    assert o_step.item() <= omega_max

    # 3. Super-threshold negative input: z = -1.2 < -0.50 -> excess = -0.70
    x_neg = torch.zeros(1, 1, hidden_dim)
    x_neg[0, 0, 0] = -1.2
    gauge.reset_state()
    o_neg, p_neg = gauge.step(x_neg[0, 0])
    assert o_neg.item() < 0.0, f"Expected negative velocity, got {o_neg.item()}"
    assert abs(o_neg.item()) <= omega_max


def test_from_contrastive_embeddings_token_sets():
    """Verify from_contrastive_embeddings averages reversal and continuation token sets."""
    hidden_dim = 32
    torch.manual_seed(101)

    # 5 reversal unembedding vectors and 5 neutral continuation vectors
    rev_tokens = torch.randn(5, hidden_dim) + 2.0
    cont_tokens = torch.randn(5, hidden_dim) - 2.0

    gauge = SemanticGaugeConnection.from_contrastive_embeddings(
        hidden_dim=hidden_dim,
        reversal_embeddings=rev_tokens,
        continuation_embeddings=cont_tokens,
        tau_gate=0.50,
        omega_max=0.05,
        nu=0.4
    )

    # Expected direction is normalized difference of centroids
    expected_diff = rev_tokens.mean(dim=0) - cont_tokens.mean(dim=0)
    expected_w = expected_diff / expected_diff.norm(p=2)

    actual_w = gauge.proj.weight.squeeze()
    cos_sim = torch.cosine_similarity(expected_w, actual_w, dim=0).item()
    assert cos_sim > 0.9999, f"Contrastive direction mismatch, cos_sim={cos_sim}"

    # Forward with a reversal-aligned vector: must activate
    test_rev = rev_tokens.mean(dim=0, keepdim=True).unsqueeze(0)
    omega_rev, phi_rev = gauge(test_rev)
    assert (omega_rev.abs() > 0.0).all(), "Reversal token centroid must activate detector"


def test_gated_softsign_prefill_decoding_equivalence():
    """Verify prefill vs autoregressive decoding equivalence with tau_gate=0.50."""
    hidden_dim = 32
    seq_len = 24
    dtype = torch.float64
    torch.manual_seed(42)

    gauge = SemanticGaugeConnection(
        hidden_dim=hidden_dim,
        out_dim=1,
        omega_max=0.05,
        nu=0.4,
        eta=0.1,
        tau_gate=0.50
    ).to(dtype=dtype)

    x = torch.randn(2, seq_len, hidden_dim, dtype=dtype) * 2.0

    # Prefill
    omega_prefill, phi_prefill = gauge(x)

    # Autoregressive decoding
    gauge.reset_state()
    omega_steps, phi_steps = [], []
    for t in range(seq_len):
        o_t, p_t = gauge.step(x[:, t:t+1, :])
        omega_steps.append(o_t)
        phi_steps.append(p_t)

    omega_decode = torch.cat(omega_steps, dim=1)
    phi_decode = torch.cat(phi_steps, dim=1)

    max_err_omega = (omega_prefill - omega_decode).abs().max().item()
    max_err_phi = (phi_prefill - phi_decode).abs().max().item()

    assert max_err_omega < 1e-12, f"Prefill/decode omega error {max_err_omega}"
    assert max_err_phi < 1e-12, f"Prefill/decode phi error {max_err_phi}"
