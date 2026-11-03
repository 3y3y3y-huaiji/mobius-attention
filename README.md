# Selective Anti-Phase Attention

A minimal research workspace for testing whether semantic phase rotations can suppress superseded premises in long-context Transformers.

**Status: research reset, not a validated publication claim.** The current path separates algebraic correctness, attention changes, and downstream task improvement. No new LLM benchmark result is claimed before its output JSON and reproduction command exist.

[中文说明](README_zh.md) · [Research plan](RESEARCH_PLAN.md) · [Controlled math result](experiments/controlled_math_results.json)

## The question

Standard softmax attention gives every visible token positive weight. A later correction may therefore compete with an obsolete premise instead of literally deleting it.

The proposed intervention adds a content-driven rotation after standard RoPE. Three different mechanisms must not be conflated:

| Mechanism | What changes | Selectivity | Historical K cache |
| --- | --- | --- | --- |
| Query-only phase | Current Q receives one extra rotation | All past keys are affected together | Unchanged |
| QK segment phase | Q and K retain their own token phases | Same segment unchanged; cross-segment modulation | Written once, then unchanged |
| Pairwise gate | Each query-key pair receives its own gate or bias | Token-pair level | Implementation-dependent |

The main hypothesis is the second one. It is an efficient middle point between global Query rotation and CoPE-style pairwise control, but it can only represent segment/state selectivity—not arbitrary entity-level forgetting.

## What the math does and does not prove

For selected RoPE planes, a relative phase of pi flips their contribution to the attention logit. This is norm-preserving and compatible with a cache written at insertion time.

It does **not** prove that:

- the total logit becomes negative when only some channels rotate;
- an obsolete token is always suppressed—an initially negative component can be amplified;
- softmax weight becomes exactly zero;
- Value vectors cancel or information is erased;
- a model can learn entity-level reversal detection;
- downstream accuracy improves.

See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the exact equations, training risks, engineering constraints, and preregistered stop criteria.

## Minimal implementation

`src/phase_attention.py` contains only:

- standard RoPE tables and rotation;
- Query-only and QK-segment phase interventions;
- causal attention normalization with an optional additive control;
- a causal reversal-phase staircase.

There is no semantic detector, topological terminology, custom cache manager, associative scan, plotting stack, or model monkey-patching in this core module.

`experiments/kaggle_reversal_eval.py` contains the small Qwen2 evaluation adapter. It is deliberately outside the core math module.

## Controlled falsification result

Run:

```bash
python -m experiments.run_controlled_math
```

The command must be run from the repository root: the module entry keeps `src` importable without a `sys.path` patch.

The deterministic example contains an old positive-logit key, an old negative-logit key, and a new positive-logit key:

| Variant | Old positive | Old negative | New positive |
| --- | ---: | ---: | ---: |
| Standard | 1.4062 | -1.4062 | 1.4062 |
| Query global | -1.4062 | 1.4062 | -1.4062 |
| Segment phase | -1.4062 | 1.4062 | 1.4062 |
| Segment phase, half channels | 0.5938 | -0.5938 | 1.4062 |
| Segment additive control | -0.5938 | -3.4062 | 1.4062 |

This is a tensor-level sanity experiment, **not an LLM-quality result**. It exposes two failure modes immediately: global Query rotation harms new information, and phase inversion can amplify an old negative logit.

## Tests

```bash
python -m pytest tests/test_phase_attention.py -q
python -m pytest tests/test_qwen_phase_eval.py -q
```

The first suite covers relative RoPE behavior, same-segment equivalence, negative-logit amplification, partial channels, gradient finiteness, and prefill/decode equivalence.

The second uses a randomly initialized tiny Qwen2—no weight download—to verify that the evaluation adapter exactly matches the original model at zero phase and restores the original forward methods afterward. It also pins the paired scorer to one prompt forward and checks that both candidate log-probabilities are finite, non-positive, and share a single softmax normalizer.

## Kaggle model experiment

The first pretrained-model run uses Qwen2.5-0.5B-Instruct on one GPU. It compares standard, Query global, QK segment, and matched additive control using Oracle revision boundaries.

```bash
python -m experiments.kaggle_reversal_eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --lengths 512 2048 8192 \
  --cases 32 \
  --output experiments/qwen25_reversal_results.json
```

Each case is scored with **one prompt forward**: the correct and stale candidates are read from the same next-token softmax, so the reported margin is a paired log-probability difference rather than two separately normalized scores.

Primary metrics are correct-versus-stale answer log-probability, answer NLL, and accuracy. The suite includes double reversal, no reversal, and an unrelated-entity reversal to measure collateral damage. It does not use attention contrast as a proxy for task success.

The experiment currently has **no recorded 0.5B result in this worktree**. Run the command before drawing model-level conclusions.

## Repository map

```text
src/phase_attention.py                  Minimal differentiable intervention
experiments/run_controlled_math.py      Deterministic falsification matrix
experiments/kaggle_reversal_eval.py     Pretrained Qwen2 evaluation
experiments/controlled_math_results.json
tests/test_phase_attention.py
tests/test_qwen_phase_eval.py
RESEARCH_PLAN.md                       Hypotheses, stages, and stop criteria
legacy/                                Superseded Agent prototypes and notes
paper/                                 Historical manuscript, not current validation
formal_verification/                   Historical algebraic proofs
```

The older SG-Möbius modules and paper-integrity pipeline remain traceable. Their tests and proofs should be interpreted as algebraic regression checks, not evidence of learned premise invalidation or improved downstream accuracy. Duplicate prototypes and narrative handovers were moved to `legacy/` rather than deleted.

`experiments/` also still holds `kaggle_experiment_results.json`, `benchmark_results.json`, `kernel-metadata.json`, `data/`, and the two `fig_*.png` files. They are inputs to the frozen paper tables and remain historical SG-Möbius artifacts, not results of the current path.

## Related work

- [CoPE, arXiv:2405.18719](https://arxiv.org/abs/2405.18719): query-key-dependent contextual positions; the paper does not validate billion-parameter models.
- [Round and Round We Go, arXiv:2410.06205](https://arxiv.org/abs/2410.06205): RoPE frequency roles, semantic drift, and p-RoPE.
- [Differential Transformer, arXiv:2410.05258](https://arxiv.org/abs/2410.05258): subtraction of two softmax attention maps for noise suppression; ICLR 2025 Oral.
- [Selective Attention, arXiv:2411.12892](https://arxiv.org/abs/2411.12892): learned control of attention sparsity and relevance.

## License

Apache-2.0 and MulanPSL-2.0; see the existing license files.
