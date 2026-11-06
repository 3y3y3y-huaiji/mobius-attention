import math

import pytest
import torch

from src.phase_attention import (
    apply_phase,
    attention_scores,
    causal_attention_weights,
    entity_reversal_phase,
    low_frequency_channel_mask,
    reversal_phase,
)


def _vectors(dim: int = 4):
    q = torch.tensor([1.0, 0.5, -0.25, 0.75])[:dim].view(1, 1, 1, dim)
    k = q.clone()
    return q, k


def _scores(q, k, q_phase, k_phase, mode, mask=None):
    return attention_scores(
        q,
        k,
        q_positions=torch.tensor([0]),
        k_positions=torch.tensor([0]),
        q_phase=q_phase,
        k_phase=k_phase,
        mode=mode,
        channel_mask=mask,
    )


def test_standard_rope_depends_on_relative_position():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, 8, dtype=torch.float64)
    k = torch.randn(1, 1, 1, 8, dtype=torch.float64)
    kwargs = dict(q=q, k=k, mode="standard")
    near = attention_scores(**kwargs, q_positions=torch.tensor([3]), k_positions=torch.tensor([1]))
    shifted = attention_scores(**kwargs, q_positions=torch.tensor([103]), k_positions=torch.tensor([101]))
    torch.testing.assert_close(near, shifted, atol=1e-12, rtol=1e-12)


def test_zero_phase_is_exactly_standard_rope():
    torch.manual_seed(1)
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    positions = torch.arange(3)
    standard = attention_scores(q, k, q_positions=positions, k_positions=positions)
    for mode in ("query", "segment"):
        modified = attention_scores(
            q,
            k,
            q_positions=positions,
            k_positions=positions,
            q_phase=torch.zeros(3),
            k_phase=torch.zeros(3),
            mode=mode,
        )
        torch.testing.assert_close(standard, modified)


def test_same_segment_phase_cancels_in_qk_inner_product():
    q, k = _vectors()
    standard = _scores(q, k, None, None, "standard")
    segment = _scores(q, k, torch.tensor([math.pi]), torch.tensor([math.pi]), "segment")
    torch.testing.assert_close(standard, segment, atol=1e-6, rtol=1e-6)


def test_cross_segment_full_pi_flips_only_the_selected_component():
    q, k = _vectors()
    standard = _scores(q, k, None, None, "standard")
    inverted = _scores(q, k, torch.tensor([math.pi]), torch.tensor([0.0]), "segment")
    torch.testing.assert_close(inverted, -standard, atol=1e-6, rtol=1e-6)


def test_query_only_rotation_is_global_but_segment_rotation_preserves_new_keys():
    q, old_k = _vectors()
    k = torch.cat((old_k, old_k), dim=-2)
    phases_k = torch.tensor([0.0, math.pi])
    query = attention_scores(
        q,
        k,
        q_positions=torch.tensor([0]),
        k_positions=torch.tensor([0, 0]),
        q_phase=torch.tensor([math.pi]),
        k_phase=phases_k,
        mode="query",
    )
    segment = attention_scores(
        q,
        k,
        q_positions=torch.tensor([0]),
        k_positions=torch.tensor([0, 0]),
        q_phase=torch.tensor([math.pi]),
        k_phase=phases_k,
        mode="segment",
    )
    assert query[0, 0, 0, 0].item() == pytest.approx(query[0, 0, 0, 1].item(), abs=1e-6)
    assert segment[0, 0, 0, 0].item() < 0 < segment[0, 0, 0, 1].item()


def test_phase_flip_can_amplify_a_previously_negative_logit():
    q, k = _vectors()
    inverted = _scores(q, -k, torch.tensor([math.pi]), torch.tensor([0.0]), "segment")
    assert inverted.item() > 0


def test_partial_phase_flip_does_not_imply_a_negative_total_logit():
    q, k = _vectors()
    inverted = _scores(
        q,
        k,
        torch.tensor([math.pi]),
        torch.tensor([0.0]),
        "segment",
        mask=torch.tensor([0.0, 1.0]),
    )
    assert inverted.item() > 0


def test_phase_rotation_preserves_vector_norm():
    torch.manual_seed(2)
    x = torch.randn(2, 3, 5, 8)
    phase = torch.randn(2, 3, 5)
    mask = torch.tensor([1.0, 0.0, 0.5, 1.0])
    rotated = apply_phase(x, phase, mask)
    torch.testing.assert_close(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_reversal_phase_is_causal_and_has_pi_steps():
    phase = reversal_phase(6, [2, 4])
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]) * math.pi
    torch.testing.assert_close(phase, expected)


def test_entity_reversal_phase_protects_unrelated_entity():
    # Tokens: 0=ALPHA premise, 1=filler, 2=BETA revision, 3=ALPHA query
    token_entities = [1, 0, 2, 1]
    global_phase = reversal_phase(4, [2])
    entity_phase = entity_reversal_phase(token_entities, [(2, 2)])
    # Under global phase, ALPHA query (token 3) gets pi, clashing with ALPHA premise (token 0, phase 0)
    assert global_phase[3].item() == pytest.approx(math.pi)
    # Under entity phase, only BETA revision (token 2) gets pi; ALPHA premise & query stay at 0
    expected = torch.tensor([0.0, 0.0, math.pi, 0.0])
    torch.testing.assert_close(entity_phase, expected)

    q, k_elem = _vectors()
    k = torch.cat((k_elem, k_elem), dim=-2)  # [ALPHA premise at t=0, BETA revision at t=2]
    scores_entity = attention_scores(
        q,
        k,
        q_positions=torch.tensor([0]),
        k_positions=torch.tensor([0, 0]),
        q_phase=entity_phase[-1:],
        k_phase=torch.stack((entity_phase[0], entity_phase[2])),
        mode="segment",
    )
    # ALPHA premise (index 0) is preserved (> 0), BETA distractor (index 1) is flipped (< 0)
    assert scores_entity[0, 0, 0, 0].item() > 0 > scores_entity[0, 0, 0, 1].item()


def test_low_frequency_channel_mask_and_per_plane_phase():
    mask = low_frequency_channel_mask(8, 0.25)
    torch.testing.assert_close(mask, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    q = torch.tensor([1.0, 0.5, -0.25, 0.75, 0.2, -0.4, 0.8, -0.1]).view(1, 1, 1, 8)
    k = q.clone()
    plane_phase_q = torch.tensor([[0.0, 0.0, 0.0, math.pi]])
    plane_phase_k = torch.zeros(1, 4)
    via_plane = _scores(q, k, plane_phase_q, plane_phase_k, "segment")
    via_mask = _scores(q, k, torch.tensor([math.pi]), torch.tensor([0.0]), "segment", mask=mask)
    torch.testing.assert_close(via_plane, via_mask, atol=1e-6, rtol=1e-6)


def test_causal_attention_masks_future_keys():
    scores = torch.zeros(1, 1, 3, 3)
    positions = torch.arange(3)
    weights = causal_attention_weights(scores, positions, positions)
    torch.testing.assert_close(weights.triu(1), torch.zeros_like(weights.triu(1)))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 1, 3))


def test_fixed_query_rotation_has_finite_norm_preserving_input_jacobian():
    torch.manual_seed(3)
    q = torch.randn(1, 1, 2, 4, requires_grad=True)
    k = torch.randn(1, 1, 2, 4)
    positions = torch.arange(2)
    common = dict(k=k, q_positions=positions, k_positions=positions)
    standard = attention_scores(q=q, **common).sum()
    standard_grad = torch.autograd.grad(standard, q, retain_graph=True)[0]
    modified = attention_scores(q=q, q_phase=torch.tensor([0.3, 1.2]), mode="query", **common).sum()
    modified_grad = torch.autograd.grad(modified, q)[0]
    assert torch.isfinite(modified_grad).all()
    torch.testing.assert_close(standard_grad.norm(), modified_grad.norm(), atol=1e-5, rtol=1e-5)


def test_last_token_decode_scores_match_prefill():
    torch.manual_seed(4)
    q = torch.randn(1, 1, 5, 8)
    k = torch.randn(1, 1, 5, 8)
    positions = torch.arange(5)
    phase = reversal_phase(5, [2])
    for mode in ("query", "segment"):
        full = attention_scores(
            q,
            k,
            q_positions=positions,
            k_positions=positions,
            q_phase=phase,
            k_phase=phase,
            mode=mode,
        )
        decoded = attention_scores(
            q[..., -1:, :],
            k,
            q_positions=positions[-1:],
            k_positions=positions,
            q_phase=phase[-1:],
            k_phase=phase,
            mode=mode,
        )
        torch.testing.assert_close(full[..., -1:, :], decoded, atol=1e-6, rtol=1e-6)
