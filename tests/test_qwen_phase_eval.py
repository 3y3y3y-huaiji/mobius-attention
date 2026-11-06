import math
import types

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from experiments.kaggle_reversal_eval import PhaseAttentionController, paired_answer_logprobs
from src.phase_attention import reversal_phase


def _tiny_model():
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        use_sliding_window=False,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    model.config.use_cache = False
    return model


def test_qwen_patch_matches_original_for_every_zero_phase_mode():
    torch.manual_seed(0)
    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        expected = model(input_ids, use_cache=False).logits
        for mode in ("standard", "query", "segment", "additive"):
            phase = torch.zeros(input_ids.shape[1])
            with PhaseAttentionController(model, mode, phase):
                actual = model(input_ids, use_cache=False).logits
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_qwen_controller_restores_original_forward():
    model = _tiny_model()
    originals = [layer.self_attn.forward for layer in model.model.layers]
    with PhaseAttentionController(model, "segment", reversal_phase(4, [2])):
        assert all(
            layer.self_attn.forward != original
            for layer, original in zip(model.model.layers, originals)
        )
    assert all(
        layer.self_attn.forward == original
        for layer, original in zip(model.model.layers, originals)
    )


def test_qwen_phase_path_has_finite_gradients():
    torch.manual_seed(1)
    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    phase = reversal_phase(4, [2])
    with PhaseAttentionController(model, "segment", phase):
        loss = model(input_ids, use_cache=False).logits.square().mean()
        loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_paired_answer_scoring_uses_one_prompt_forward():
    class AnswerTokenizer:
        answers = {" valid": [10], " invalid": [11]}

        def __call__(self, text, add_special_tokens=False):
            return types.SimpleNamespace(input_ids=self.answers[text])

    model = _tiny_model()
    calls = []
    bound_forward = model.forward

    def counting_forward(*args, **kwargs):
        calls.append(1)
        return bound_forward(*args, **kwargs)

    model.forward = counting_forward
    correct, stale, prompt_tokens, elapsed_ms = paired_answer_logprobs(
        model,
        AnswerTokenizer(),
        [1, 2, 3, 4],
        " valid",
        " invalid",
        [2],
        "standard",
        2.0,
    )

    # Two answers must share a single prompt forward and therefore one softmax normalizer.
    assert len(calls) == 1
    assert prompt_tokens == 4
    assert elapsed_ms >= 0
    assert math.isfinite(correct) and math.isfinite(stale)
    assert correct <= 0.0 and stale <= 0.0
    assert math.exp(correct) + math.exp(stale) <= 1.0 + 1e-6
    assert correct != stale

    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3, 4]]), use_cache=False).logits[0, -1].float()
        reference = torch.log_softmax(logits, dim=-1)
    assert correct == pytest.approx(reference[10].item(), abs=1e-5)
    assert stale == pytest.approx(reference[11].item(), abs=1e-5)
    assert len(calls) == 2


def test_paired_answer_scoring_stays_finite_under_segment_phase():
    class AnswerTokenizer:
        answers = {" valid": [10], " invalid": [11]}

        def __call__(self, text, add_special_tokens=False):
            return types.SimpleNamespace(input_ids=self.answers[text])

    torch.manual_seed(5)
    model = _tiny_model()
    args = (model, AnswerTokenizer(), [1, 2, 3, 4], " valid", " invalid", [2])
    baseline = paired_answer_logprobs(*args, "standard", 2.0)
    phased = paired_answer_logprobs(*args, "segment", 2.0)

    for value in phased[:2]:
        assert math.isfinite(value)
        assert value <= 0.0
    assert math.exp(phased[0]) + math.exp(phased[1]) <= 1.0 + 1e-6
    # The non-zero reversal phase must reach the answer logits instead of being a no-op.
    assert phased[:2] != baseline[:2]


def _tiny_qwen3_model():
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        use_sliding_window=False,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    model.config.use_cache = False
    return model


def _tiny_qwen35_model():
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(config).eval()
    model.config.use_cache = False
    return model


@pytest.mark.parametrize("builder", [_tiny_qwen3_model, _tiny_qwen35_model])
def test_qwen3_and_qwen35_patch_matches_original_for_zero_phase(builder):
    torch.manual_seed(7)
    model = builder()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        expected = model(input_ids, use_cache=False).logits
        for mode in (
            "standard",
            "query",
            "segment",
            "entity_segment",
            "lowfreq_segment",
            "entity_lowfreq",
            "additive",
            "entity_additive",
        ):
            phase = torch.zeros(input_ids.shape[1])
            with PhaseAttentionController(model, mode, phase):
                actual = model(input_ids, use_cache=False).logits
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_entity_segment_matches_segment_on_single_and_protects_unrelated():
    class AnswerTokenizer:
        answers = {" valid": [10], " invalid": [11]}

        def __call__(self, text, add_special_tokens=False):
            return types.SimpleNamespace(input_ids=self.answers[text])

    torch.manual_seed(9)
    model = _tiny_qwen35_model()
    tok = AnswerTokenizer()
    # On single reversal of entity 1, entity_segment and segment have identical phases
    seg_single = paired_answer_logprobs(
        model, tok, [1, 2, 3, 4], " invalid", " valid", [2], "segment", 2.0, [1, 0, 1, 1], [(2, 1)]
    )
    ent_single = paired_answer_logprobs(
        model, tok, [1, 2, 3, 4], " invalid", " valid", [2], "entity_segment", 2.0, [1, 0, 1, 1], [(2, 1)]
    )
    assert ent_single[:2] == pytest.approx(seg_single[:2], abs=1e-5)

    # On unrelated reversal (entity 2 revised at token 2, query at token 3 asks about entity 1),
    # entity_segment differs from blind global segment because token 3 keeps entity 1's 0 phase
    seg_unrel = paired_answer_logprobs(
        model, tok, [1, 2, 3, 4], " valid", " invalid", [2], "segment", 2.0, [1, 0, 2, 1], [(2, 2)]
    )
    ent_unrel = paired_answer_logprobs(
        model, tok, [1, 2, 3, 4], " valid", " invalid", [2], "entity_segment", 2.0, [1, 0, 2, 1], [(2, 2)]
    )
    assert ent_unrel[:2] != seg_unrel[:2]
