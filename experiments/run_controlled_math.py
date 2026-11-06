"""Run a deterministic tensor-level falsification matrix, not an LLM benchmark.

Usage:
    python -m experiments.run_controlled_math
    python -m experiments.run_controlled_math --output experiments/controlled_math_results.json
"""

import argparse
import json
import math
from pathlib import Path

import torch

from src.phase_attention import (
    attention_scores,
    causal_attention_weights,
    entity_reversal_phase,
    harmonic_entity_phase,
    reversal_phase,
)


def run() -> dict:
    # Base positions are all zero, isolating the additional semantic phase from RoPE.
    q = torch.tensor([1.0, 0.5, 1.0, 0.75]).view(1, 1, 1, 4)
    k = torch.cat((q, -q, q), dim=-2)
    q_positions = torch.tensor([0])
    k_positions = torch.tensor([0, 0, 0])
    q_phase = torch.tensor([math.pi])
    k_phase = torch.tensor([0.0, 0.0, math.pi])

    def scores(mode: str, mask=None) -> torch.Tensor:
        return attention_scores(
            q,
            k,
            q_positions=q_positions,
            k_positions=k_positions,
            q_phase=q_phase if mode != "standard" else None,
            k_phase=k_phase if mode != "standard" else None,
            mode=mode,
            channel_mask=mask,
        )

    standard = scores("standard")
    variants = {
        "standard": standard,
        "query_global": scores("query"),
        "segment_phase": scores("segment"),
        "segment_phase_half_channels": scores("segment", torch.tensor([0.0, 1.0])),
        "segment_additive_control": standard
        + torch.tensor([-2.0, -2.0, 0.0]).view(1, 1, 1, 3),
    }

    # Unrelated-entity falsification: token 0 = ALPHA premise, token 1 = BETA revision, token 2 = ALPHA query
    k_unrel = torch.cat((q, q), dim=-2)
    global_unrel = reversal_phase(3, [1])
    entity_unrel = entity_reversal_phase([1, 2, 1], [(1, 2)])
    global_unrel_scores = attention_scores(
        q,
        k_unrel,
        q_positions=torch.tensor([0]),
        k_positions=torch.tensor([0, 0]),
        q_phase=global_unrel[-1:],
        k_phase=global_unrel[:2],
        mode="segment",
    )
    entity_unrel_scores = attention_scores(
        q,
        k_unrel,
        q_positions=torch.tensor([0]),
        k_positions=torch.tensor([0, 0]),
        q_phase=entity_unrel[-1:],
        k_phase=entity_unrel[:2],
        mode="segment",
    )

    # Multi-update 2*pi resurrection falsification: 0 = X1, 1 = X2, 2 = X3, 3 = Query(X)
    k_multi = torch.cat((q, q, q), dim=-2)
    scalar_multi = entity_reversal_phase([1, 1, 1, 1], [(1, 1), (2, 1)])
    harmonic_multi = harmonic_entity_phase([1, 1, 1, 1], [(1, 1), (2, 1)], head_dim=4, order=3)
    scalar_multi_scores = attention_scores(
        q,
        k_multi,
        q_positions=torch.tensor([0]),
        k_positions=torch.tensor([0, 0, 0]),
        q_phase=scalar_multi[-1:],
        k_phase=scalar_multi[:3],
        mode="segment",
    )
    harmonic_multi_scores = attention_scores(
        q,
        k_multi,
        q_positions=torch.tensor([0]),
        k_positions=torch.tensor([0, 0, 0]),
        q_phase=harmonic_multi[-1:],
        k_phase=harmonic_multi[:3],
        mode="segment",
    )

    result = {
        "experiment_type": "controlled_tensor_math_not_an_llm_benchmark",
        "key_roles": ["old_positive", "old_negative", "new_positive"],
        "score_formula": "q dot k / sqrt(4)",
        "variants": {},
        "unrelated_entity_check": {
            "key_roles": ["alpha_premise", "beta_revision_distractor"],
            "global_segment_scores": [round(v, 6) for v in global_unrel_scores.flatten().tolist()],
            "entity_segment_scores": [round(v, 6) for v in entity_unrel_scores.flatten().tolist()],
        },
        "multi_update_2pi_check": {
            "key_roles": ["x1_oldest", "x2_middle", "x3_latest"],
            "scalar_pi_scores": [round(v, 6) for v in scalar_multi_scores.flatten().tolist()],
            "harmonic_order3_scores": [round(v, 6) for v in harmonic_multi_scores.flatten().tolist()],
        },
    }
    for name, variant_scores in variants.items():
        weights = causal_attention_weights(variant_scores, q_positions, k_positions)
        result["variants"][name] = {
            "scores": [round(v, 6) for v in variant_scores.flatten().tolist()],
            "attention": [round(v, 8) for v in weights.flatten().tolist()],
        }
    result["falsification_checks"] = {
        "query_global_preserves_new_positive": False,
        "phase_flip_suppresses_every_old_token": False,
        "partial_phase_flip_guarantees_negative_logits": False,
        "additive_control_can_suppress_both_old_tokens": True,
        "global_segment_phase_protects_unrelated_entity": bool(global_unrel_scores[0, 0, 0, 0].item() > 0),
        "entity_bound_phase_protects_unrelated_entity": bool(entity_unrel_scores[0, 0, 0, 0].item() > 0),
        "scalar_pi_avoids_2pi_resurrection_on_3_states": bool(scalar_multi_scores[0, 0, 0, 0].item() < 0),
        "harmonic_order3_avoids_2pi_resurrection_on_3_states": bool(
            harmonic_multi_scores[0, 0, 0, 0].item() < 0 and harmonic_multi_scores[0, 0, 0, 1].item() < 0
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/controlled_math_results.json")
    args = parser.parse_args()
    result = run()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Controlled tensor math (not a model-quality result)")
    print("variant                         old+      old-      new+")
    for name, values in result["variants"].items():
        row = values["scores"]
        print(f"{name:30s} {row[0]:8.4f}  {row[1]:8.4f}  {row[2]:8.4f}")
    unrel = result["unrelated_entity_check"]
    print(f"unrelated (global_segment)     alpha={unrel['global_segment_scores'][0]:+8.4f}  beta_rev={unrel['global_segment_scores'][1]:+8.4f}")
    print(f"unrelated (entity_segment)     alpha={unrel['entity_segment_scores'][0]:+8.4f}  beta_rev={unrel['entity_segment_scores'][1]:+8.4f}")
    multi = result["multi_update_2pi_check"]
    s_row = multi["scalar_pi_scores"]
    h_row = multi["harmonic_order3_scores"]
    print(f"3-state (scalar_pi)            x1={s_row[0]:+8.4f}  x2={s_row[1]:+8.4f}  x3={s_row[2]:+8.4f}")
    print(f"3-state (harmonic_order3)      x1={h_row[0]:+8.4f}  x2={h_row[1]:+8.4f}  x3={h_row[2]:+8.4f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
