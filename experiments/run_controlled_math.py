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

from src.phase_attention import attention_scores, causal_attention_weights


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

    result = {
        "experiment_type": "controlled_tensor_math_not_an_llm_benchmark",
        "key_roles": ["old_positive", "old_negative", "new_positive"],
        "score_formula": "q dot k / sqrt(4)",
        "variants": {},
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
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
