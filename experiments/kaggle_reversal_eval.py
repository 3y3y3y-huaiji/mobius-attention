"""Qwen2.5 paired-answer evaluation for phase interventions.

This measures answer log-likelihood, not attention-map aesthetics. It uses Oracle
revision boundaries and no KV cache, so it isolates the intervention from detector
learning and decoding-state bugs.

Example (Kaggle or local GPU):
    python -m experiments.kaggle_reversal_eval \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --lengths 512 2048 8192 --cases 8 \
        --output experiments/qwen25_reversal_results.json

Every (length, scenario, case_index) group is a distinct prompt. --shard-file
lets a driver split whole groups across GPUs; the four modes of one group are
never split, so paired mode comparisons stay on a single device.

Validated against the Transformers 5.x Qwen2 forward contract. The tiny-config
unit tests do not download model weights.
"""

import argparse
import inspect
import json
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from src.phase_attention import apply_phase, reversal_phase


class PhaseAttentionController:
    """Temporarily replace Qwen2 attention with a minimal eager intervention."""

    def __init__(self, model, mode: str, phase: torch.Tensor, penalty: float = 2.0):
        if mode not in {"standard", "query", "segment", "additive"}:
            raise ValueError(f"Unknown mode: {mode}")
        self.model = model
        self.mode = mode
        self.phase = phase
        self.penalty = penalty
        self.originals = []

    def install(self) -> None:
        for layer in self.model.model.layers:
            attention = layer.self_attn
            self.originals.append((attention, attention.forward))
            attention.forward = types.MethodType(self._forward, attention)

    def restore(self) -> None:
        for attention, original in self.originals:
            attention.forward = original
        self.originals.clear()

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()

    def _forward(
        self,
        attention,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        **kwargs,
    ):
        past_key_value = kwargs.pop("past_key_value", None)
        if past_key_values is not None or past_key_value is not None:
            raise RuntimeError("This controlled evaluation intentionally runs with use_cache=False.")
        if getattr(attention, "sliding_window", None) is not None:
            raise RuntimeError("The minimal prototype does not implement sliding-window attention.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        q = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            if position_ids is None:
                raise ValueError("position_ids is required when position_embeddings is not supplied.")
            position_embeddings = attention.rotary_emb(hidden_states, position_ids)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        seq_len = q.shape[-2]
        if self.phase.numel() < seq_len:
            raise ValueError("The phase tensor is shorter than the input sequence.")
        phase = self.phase[:seq_len].to(device=q.device, dtype=q.dtype)
        if self.mode == "query":
            q = apply_phase(q, phase)
        elif self.mode == "segment":
            q = apply_phase(q, phase)
            k = apply_phase(k, phase)

        k = repeat_kv(k, attention.num_key_value_groups)
        v = repeat_kv(v, attention.num_key_value_groups)
        scores = torch.matmul(q, k.transpose(-1, -2)) * attention.scaling

        if self.mode == "additive":
            anti_phase = torch.cos(phase.view(1, 1, -1, 1) - phase.view(1, 1, 1, -1)) < 0
            scores = scores + anti_phase.to(scores.dtype) * (-self.penalty)

        if attention_mask is not None:
            scores = scores + attention_mask
        else:
            causal = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal.view(1, 1, seq_len, seq_len), torch.finfo(q.dtype).min)

        weights = F.softmax(scores, dim=-1)
        dropout = attention.attention_dropout if attention.training else 0.0
        weights = F.dropout(weights, p=dropout, training=attention.training)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        output = attention.o_proj(output)

        # Transformers 4.x Qwen2 returned a third cache element; 5.x returns two.
        if int(transformers_major_version()) < 5:
            return output, weights, None
        return output, weights


def transformers_major_version() -> int:
    return int(transformers.__version__.split(".", maxsplit=1)[0])


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


# Sample diversity pools. Every case_index must produce a distinct prompt, so
# repeating a case is never a silent no-op forward pass.
ENTITIES = (
    "ALPHA", "GAMMA", "DELTA", "OMEGA", "SIGMA", "THETA", "KAPPA", "LAMBDA",
    "EPSILON", "ZETA", "IOTA", "RHO", "TAU", "PHI", "CHI", "PSI",
    "OMICRON", "UPSILON", "XI", "NU", "MU", "ETA", "BETA2", "PI",
)
FILLER_TEMPLATES = (
    "Routine telemetry reports normal operation and no action is required. ",
    "The archive contains ordinary maintenance notes unrelated to entity status. ",
    "Workers completed a regular integrity check and continued monitoring. ",
    "A scheduled log entry confirms unchanged operating conditions. ",
    "The daily digest lists only routine sensor readings and timestamps. ",
    "Maintenance staff recorded ambient temperature and left the system as found. ",
    "An automated audit found no pending changes to the entity registry. ",
)
REVISION_TEMPLATES = (
    "REVISION: Entity {entity} is now invalid; its earlier valid status is revoked.\n",
    "REVISION: The earlier valid status of Entity {entity} is revoked; it is now invalid.\n",
    "REVISION: Entity {entity} is no longer valid; the previous status is withdrawn.\n",
)
RESTORATION_TEMPLATES = (
    "RESTORATION: The revision for {entity} was cancelled; valid is the current status again.\n",
    "RESTORATION: The revocation for {entity} has been withdrawn; valid applies again.\n",
)
UNRELATED_TEMPLATES = (
    "REVISION: Entity BETA is now invalid; its earlier valid status is revoked.\n",
    "REVISION: Entity BETA is no longer valid; its previous status is withdrawn.\n",
)
NO_REVISION_TEMPLATES = (
    "No status revision appears in this record.\n",
    "The record contains no change of entity status.\n",
)
QUERY_TEMPLATES = (
    "Question: What is the current status of Entity {entity}? "
    "Reply with exactly one word, valid or invalid.\nAnswer:",
    "Question: State the current status of Entity {entity} using one word, valid or invalid.\nAnswer:",
)


def build_case(tokenizer, scenario: str, target_tokens: int, case_index: int) -> dict:
    """Build one unique prompt whose real token count matches target_tokens."""
    entity = ENTITIES[case_index % len(ENTITIES)]
    variant = (case_index // len(ENTITIES)) % 3
    filler_offset = case_index % len(FILLER_TEMPLATES)
    premise = f"Entity {entity} status: valid.\n"

    revision = REVISION_TEMPLATES[variant].format(entity=entity)
    restoration = RESTORATION_TEMPLATES[variant % len(RESTORATION_TEMPLATES)].format(entity=entity)
    query = QUERY_TEMPLATES[variant % len(QUERY_TEMPLATES)].format(entity=entity)
    if scenario == "single":
        middle = revision
        correct = " invalid"
    elif scenario == "double":
        middle = revision + restoration
        correct = " valid"
    elif scenario == "unrelated":
        middle = UNRELATED_TEMPLATES[variant % len(UNRELATED_TEMPLATES)]
        correct = " valid"
    elif scenario == "none":
        middle = NO_REVISION_TEMPLATES[variant % len(NO_REVISION_TEMPLATES)]
        correct = " valid"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Budget by measured token cost: the original //18 estimate produced prompts
    # ~62% of the advertised length, so "L=8192" was really ~5000 tokens.
    # Concatenated text merges BPE tokens across sentence boundaries, so a
    # short full-prompt calibration loop removes the remaining ~8% shortfall.
    tail = middle + query
    budget = max(1, target_tokens - len(encode(tokenizer, premise)) - len(encode(tokenizer, tail)))
    costs = [len(encode(tokenizer, template)) for template in FILLER_TEMPLATES]
    filler = []
    used = 0.0
    while used < budget * 1.09:
        slot = (len(filler) + filler_offset) % len(FILLER_TEMPLATES)
        filler.append(FILLER_TEMPLATES[slot])
        used += costs[slot]
    for _ in range(6):
        current = len(encode(tokenizer, premise + "".join(filler) + tail))
        gap = target_tokens - current
        if abs(gap) <= 4 or not filler:
            break
        step = max(1, abs(gap) // max(1, min(costs)))
        if gap > 0:
            for _ in range(step):
                slot = (len(filler) + filler_offset) % len(FILLER_TEMPLATES)
                filler.append(FILLER_TEMPLATES[slot])
        else:
            del filler[-min(step, len(filler)):]
    prefix = premise + "".join(filler)
    prefix_tokens = len(encode(tokenizer, prefix))

    if scenario in ("single", "unrelated"):
        reversals = [prefix_tokens]
    elif scenario == "double":
        reversals = [prefix_tokens, len(encode(tokenizer, prefix + revision))]
    else:
        reversals = []

    prompt = prefix + middle + query
    return {
        "scenario": scenario,
        "case_index": case_index,
        "entity": entity,
        "prompt_ids": encode(tokenizer, prompt),
        "reversals": reversals,
        "correct": correct,
        "stale": " invalid" if correct == " valid" else " valid",
    }


def paired_answer_logprobs(
    model,
    tokenizer,
    prompt_ids: list[int],
    correct: str,
    stale: str,
    reversals: list[int],
    mode: str,
    penalty: float,
) -> tuple[float, float, int, float]:
    correct_id = encode(tokenizer, correct)
    stale_id = encode(tokenizer, stale)
    if len(correct_id) != 1 or len(stale_id) != 1:
        raise ValueError(f"Both answers must tokenize to one token, got {correct_id} and {stale_id}.")
    phase = reversal_phase(len(prompt_ids), reversals)
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], device=device)
    # Only the final position is scored. Computing logits for all L positions
    # would materialise an L x vocab tensor (2.5 GB at L=8192) for no reason.
    forward_kwargs = {"use_cache": False}
    parameters = inspect.signature(model.forward).parameters
    for key in ("logits_to_keep", "num_logits_to_keep"):
        if key in parameters:
            forward_kwargs[key] = 1
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with PhaseAttentionController(model, mode, phase, penalty):
        with torch.no_grad():
            logits = model(ids, **forward_kwargs).logits
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    logprobs = F.log_softmax(logits[0, -1].float(), dim=-1)
    return logprobs[correct_id[0]].item(), logprobs[stale_id[0]].item(), len(prompt_ids), elapsed_ms


def summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        key = (record["mode"], record["scenario"], record["target_tokens"])
        groups[key].append(record)
    rows = []
    for (mode, scenario, length), items in sorted(groups.items()):
        rows.append({
            "mode": mode,
            "scenario": scenario,
            "target_tokens": length,
            "count": len(items),
            "accuracy": sum(item["margin"] > 0 for item in items) / len(items),
            "mean_margin": sum(item["margin"] for item in items) / len(items),
            "mean_correct_logprob": sum(item["correct_logprob"] for item in items) / len(items),
            "mean_prompt_tokens": sum(item["prompt_tokens"] for item in items) / len(items),
            "mean_forward_ms": sum(item["elapsed_ms"] for item in items) / len(items),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--cases", type=int, default=8, help="Distinct cases per scenario and target length.")
    parser.add_argument("--modes", nargs="+", default=["standard", "query", "segment", "additive"])
    parser.add_argument("--penalty", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="experiments/qwen25_reversal_results.json")
    parser.add_argument(
        "--shard-file",
        default=None,
        help="Optional JSON list of {length, scenario, case_index} groups assigned to this worker.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if len(encode(tokenizer, " valid")) != 1 or len(encode(tokenizer, " invalid")) != 1:
        raise RuntimeError("This paired evaluation requires one-token ' valid' and ' invalid' candidates.")

    shard = None
    if args.shard_file is not None:
        shard = {
            (item["length"], item["scenario"], item["case_index"])
            for item in json.loads(Path(args.shard_file).read_text(encoding="utf-8"))
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="eager")
    model.to(device).eval()
    model.config.use_cache = False

    scenarios = ["single", "double", "unrelated", "none"]
    # Tokenise every case once, and fail loudly if two cases would be identical.
    cases = {}
    signatures = defaultdict(set)
    for length in args.lengths:
        for scenario in scenarios:
            for case_index in range(args.cases):
                case = build_case(tokenizer, scenario, length, case_index)
                signature = tuple(case["prompt_ids"])
                if signature in signatures[(length, scenario)]:
                    raise RuntimeError(
                        f"Duplicate prompt at length={length} scenario={scenario} case={case_index}; "
                        "repeating identical prompts wastes compute."
                    )
                signatures[(length, scenario)].add(signature)
                cases[(length, scenario, case_index)] = case

    records = []
    try:
        for length in args.lengths:
            for scenario in scenarios:
                for case_index in range(args.cases):
                    if shard is not None and (length, scenario, case_index) not in shard:
                        continue
                    case = cases[(length, scenario, case_index)]
                    for mode in args.modes:
                        correct_logprob, stale_logprob, prompt_tokens, elapsed = paired_answer_logprobs(
                            model,
                            tokenizer,
                            case["prompt_ids"],
                            case["correct"],
                            case["stale"],
                            case["reversals"],
                            mode,
                            args.penalty,
                        )
                        records.append({
                            "scenario": case["scenario"],
                            "case_index": case_index,
                            "entity": case["entity"],
                            "reversals": case["reversals"],
                            "correct": case["correct"],
                            "stale": case["stale"],
                            "mode": mode,
                            "target_tokens": length,
                            "prompt_tokens": prompt_tokens,
                            "correct_logprob": correct_logprob,
                            "stale_logprob": stale_logprob,
                            "margin": correct_logprob - stale_logprob,
                            "elapsed_ms": elapsed,
                        })
                        print(
                            f"L={length} {scenario:9s} case={case_index} {mode:8s} "
                            f"margin={records[-1]['margin']:.4f}"
                        )
    finally:
        model.config.use_cache = True

    result = {
        "model": args.model,
        "transformers_version": transformers.__version__,
        "device": str(device),
        "dtype": str(dtype),
        "penalty": args.penalty,
        "seed": args.seed,
        "cases_are_unique": True,
        "shard_file": args.shard_file,
        "oracle_boundaries": True,
        "uses_kv_cache": False,
        "command": "python -m experiments.kaggle_reversal_eval " + " ".join(sys.argv[1:]),
        "summary": summarize(records),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
