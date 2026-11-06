# Selective Anti-Phase Attention

A minimal research workspace for testing whether semantic phase rotations can suppress superseded premises in long-context Transformers.

**Status: research reset and empirical stress-test.** The current path separates algebraic correctness, attention changes, and downstream task improvement. No LLM benchmark claim is made without its raw output JSON and exact reproduction command.

[中文说明](README_zh.md) · [Research plan](RESEARCH_PLAN.md) · [Controlled math result](experiments/controlled_math_results.json) · [Qwen2.5 report](experiments/qwen25_reversal_report.md) · [Qwen3 report](experiments/qwen3_06b_reversal_report.md) · [Qwen3.5 report](experiments/qwen35_08b_reversal_report.md)

## The question

Standard softmax attention gives every visible token positive weight. A later correction may therefore compete with an obsolete premise instead of literally deleting it.

The proposed intervention adds a content-driven rotation after standard RoPE. Three different mechanisms must not be conflated:

| Mechanism | What changes | Selectivity | Historical K cache |
| --- | --- | --- | --- |
| Query-only phase | Current Q receives one extra rotation | All past keys are affected together | Unchanged |
| QK segment phase | Q and K retain their own token phases | Same segment unchanged; cross-segment modulation | Written once, then unchanged |
| Pairwise / entity-bound gate | Each query-key pair or entity subspace receives its own gate | Entity or token-pair level | Implementation-dependent |

The main hypothesis under test is the second one (`segment`). It is an O(Ld) middle point between global Query rotation and CoPE-style pairwise control, but a 1D global time staircase can only represent segment-level selectivity—not entity-specific forgetting.

## What the math does and does not prove

For selected RoPE planes, a relative phase of pi flips their contribution to the attention logit. This is norm-preserving and compatible with a cache written at insertion time.

It does **not** prove that:

- the total logit becomes negative when only some channels rotate;
- an obsolete token is always suppressed—an initially negative component can be amplified;
- softmax weight becomes exactly zero;
- Value vectors cancel or information is erased;
- a 1D time-segment phase can avoid suppressing unrelated entities defined in earlier segments;
- a model can learn entity-level reversal detection;
- downstream accuracy improves over a matched additive bias.

See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the exact equations, training risks, engineering constraints, and preregistered stop criteria.

## Minimal implementation

`src/phase_attention.py` contains only:

- standard RoPE tables and rotation;
- Query-only and QK-segment phase interventions;
- causal attention normalization with an optional additive control;
- a causal reversal-phase staircase.

There is no semantic detector, topological terminology, custom cache manager, associative scan, plotting stack, or model monkey-patching in this core module.

`experiments/kaggle_reversal_eval.py` contains the evaluation adapter for `Qwen2`, `Qwen3`, and `Qwen3.5` (including QK-Norm, `partial_rotary_factor`, and hybrid `full_attention` / `linear_attention` layers). It is deliberately outside the core math module.

## Controlled falsification result

Run from the repository root:

```bash
python -m experiments.run_controlled_math
```

The deterministic example contains an old positive-logit key, an old negative-logit key, and a new positive-logit key:

| Variant | Old positive | Old negative | New positive |
| --- | ---: | ---: | ---: |
| Standard | 1.4062 | -1.4062 | 1.4062 |
| Query global | -1.4062 | 1.4062 | -1.4062 |
| Segment phase | -1.4062 | 1.4062 | 1.4062 |
| Segment phase, half channels | 0.5938 | -0.5938 | 1.4062 |
| Segment additive control | -0.5938 | -3.4062 | 1.4062 |

This is a tensor-level sanity check, **not an LLM-quality result**. It exposes two failure modes immediately: global Query rotation harms new information, and phase inversion can amplify an old negative logit.

## Tests

```bash
python -m pytest tests/test_phase_attention.py -q
python -m pytest tests/test_qwen_phase_eval.py -q
```

The first suite covers relative RoPE behavior, same-segment equivalence, negative-logit amplification, partial channels, gradient finiteness, and prefill/decode equivalence.

The second uses randomly initialized tiny `Qwen2`, `Qwen3`, and `Qwen3.5` models—no weight download—to verify that the evaluation adapter exactly matches the original model at zero phase and restores the original forward methods afterward.

## Kaggle pretrained-model experiments (Qwen2.5 vs Qwen3 vs Qwen3.5)

We evaluated three generations of Qwen architectures under Oracle revision boundaries across target lengths 512, 2048, and 8192 (8 distinct prompts per length and scenario, 4 modes = **384 paired forwards per model** on 2× Tesla T4):

```bash
python -m experiments.kaggle_reversal_eval \
  --model Qwen/Qwen3.5-0.8B \
  --lengths 512 2048 8192 \
  --cases 8 \
  --output experiments/qwen35_08b_reversal_results.json
```

Each case is scored with **one prompt forward**: the correct and stale candidates are read from the same next-token softmax, so the reported margin is a paired log-probability difference. All four modes of a `(length, scenario, case_index)` group run on the same GPU.

### Cross-architecture summary (Accuracy / Mean Margin across L = 512, 2048, 8192)

| Model & Rotary Architecture | Scenario | `standard` | `query` | `segment` | `additive` |
| --- | --- | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B-Instruct**<br>*(24 full-attn layers, 100% rotary, no QK-Norm)*<br>[JSON](experiments/qwen25_reversal_results.json) · [Report](experiments/qwen25_reversal_report.md) | `single` | 1.000 / +3.49 | 0.792 / +1.37 | 0.833 / +1.75 | 0.000 / -2.90 |
| | `double` | 1.000 / +1.85 | 1.000 / +1.20 | 1.000 / +1.69 | 1.000 / +2.71 |
| | `unrelated` | 0.208 / -0.86 | 0.250 / -1.21 | 0.167 / -1.56 | 1.000 / +6.35 |
| | `none` | 1.000 / +4.41 | 1.000 / +4.41 | 1.000 / +4.41 | 1.000 / +4.41 |
| **Qwen3-0.6B**<br>*(28 full-attn layers, 100% rotary, with QK-Norm)*<br>[JSON](experiments/qwen3_06b_reversal_results.json) · [Report](experiments/qwen3_06b_reversal_report.md) | `single` | 1.000 / +3.44 | 0.000 / -1.01 | 0.292 / -0.80 | 0.750 / +2.10 |
| | `double` | 0.000 / -1.72 | 0.625 / +0.50 | 0.458 / -0.57 | 0.792 / +0.55 |
| | `unrelated` | 0.000 / -1.40 | 1.000 / +1.13 | 0.708 / +0.70 | 0.458 / -0.33 |
| | `none` | 0.917 / +1.35 | 0.917 / +1.35 | 0.917 / +1.35 | 0.917 / +1.35 |
| **Qwen3.5-0.8B**<br>*(6 full-attn + 18 linear-attn layers, 25% partial rotary)*<br>[JSON](experiments/qwen35_08b_reversal_results.json) · [Report](experiments/qwen35_08b_reversal_report.md) | `single` | 1.000 / +1.37 | 0.000 / -1.05 | **1.000 / +1.82** | 1.000 / +2.17 |
| | `double` | 0.000 / -0.48 | 0.625 / +0.04 | **1.000 / +0.65** | 1.000 / +0.30 |
| | `unrelated` | 0.042 / -0.39 | 1.000 / +1.04 | 0.000 / -1.42 | 0.000 / -1.29 |
| | `none` | 1.000 / +0.48 | 1.000 / +0.48 | **1.000 / +0.48** | 1.000 / +0.48 |

### Three empirical takeaways

1. **Why 25% partial rotary (`Qwen3.5-0.8B`) works zero-shot while 100% rotary (`Qwen2.5-0.5B`, `Qwen3-0.6B`) degrades**:
   In `Qwen2.5` and `Qwen3`, `partial_rotary_factor = 1.0` rotates 100% of head dimensions. Flipping all 100% of channels by pi without fine-tuning disrupts syntactic and positional anchor heads (`Qwen3-0.6B` drops from `1.000` to `0.292` on `single`). By contrast, `Qwen3.5-0.8B` natively uses `partial_rotary_factor = 0.25` (only 64 of 256 head dimensions use RoPE; the remaining 192 dimensions are unrotated semantic anchors) across 6 full-attention layers. On `Qwen3.5-0.8B`, `segment` raises reversal accuracy from **0.500 to 1.000 (+50.00 pp)**: it restores `double` reversal (`valid -> invalid -> valid`) from `0.000` to `1.000` at all lengths and increases the positive margin on `single` (`+1.37 -> +1.82`).
2. **Both structural critiques from theoretical review are quantitatively confirmed**:
   - **Query-only rotation (`query`) fails on `single` (`0.000` on Qwen3 and Qwen3.5)**: rotating only the Query flips its dot product with all prior Keys simultaneously, penalizing both the obsolete premise and the valid revision.
   - **1D global time-segment phase hurts unrelated entities (`unrelated`)**: when `Entity ALPHA` is defined in segment 0 and an unrelated `Entity BETA` reverses near the end of the prompt (advancing the global phase to pi), querying `ALPHA` at phase pi assigns a relative phase of pi to `ALPHA`'s definition (`0.000` accuracy on `Qwen3.5-0.8B`, margin dropping from `-0.39` to `-1.42`).
3. **Why the stop criteria still hold before learned gating (E3)**:
   Even on `Qwen3.5-0.8B`, the matched additive control (`additive`) also achieves `1.000` accuracy on the reversal scenarios (`segment - additive = 0.00 pp`, though `segment` achieves a larger positive margin on `double`: `+0.65` vs `+0.30`). Because a 1D global phase staircase acts as a time-segment filter rather than an entity-bound operator, the next step is **entity-bound (key-dependent) phase gating** using the unrotated 75% semantic subspace before training any learned detector.

## Repository map

```text
src/phase_attention.py                      Minimal differentiable phase intervention
experiments/run_controlled_math.py          Deterministic tensor falsification matrix
experiments/kaggle_reversal_eval.py         Qwen2 / Qwen3 / Qwen3.5 evaluation harness
experiments/controlled_math_results.json    Output of run_controlled_math.py
experiments/qwen25_reversal_results.json    Qwen2.5-0.5B-Instruct Kaggle E2 results
experiments/qwen25_reversal_report.md       Qwen2.5-0.5B-Instruct preregistered report
experiments/qwen3_06b_reversal_results.json Qwen3-0.6B Kaggle E2 results
experiments/qwen3_06b_reversal_report.md    Qwen3-0.6B preregistered report
experiments/qwen35_08b_reversal_results.json Qwen3.5-0.8B Kaggle E2 results
experiments/qwen35_08b_reversal_report.md   Qwen3.5-0.8B preregistered report
scripts/build_kaggle_kernel.py              Self-contained dual-T4 Kaggle kernel builder
scripts/monitor_kaggle_e2.py                Kaggle kernel polling and downloader
scripts/summarize_kaggle_e2.py              Preregistered criteria evaluator
tests/test_phase_attention.py               Unit tests for core phase operators
tests/test_qwen_phase_eval.py               Zero-phase equivalence tests (Qwen2/3/3.5)
RESEARCH_PLAN.md                            Hypotheses, stages, and stop criteria
legacy/                                     Superseded Agent prototypes and handovers
paper/                                      Historical manuscript, not current validation
formal_verification/                        Lean 4 algebraic proofs (17 theorems)
```

The older SG-Möbius modules (`src/mobius_attention.py`, `src/gauge_predictor.py`, `src/kv_cache.py`), `paper/`, and `formal_verification/` remain traceable because the regression and paper-integrity pipelines verify them. Their tests and proofs are algebraic regression checks, not evidence of learned premise invalidation or downstream LLM accuracy.

## Related work

- [CoPE, arXiv:2405.18719](https://arxiv.org/abs/2405.18719): query-key-dependent contextual positions; the paper does not validate billion-parameter models.
- [Round and Round We Go, arXiv:2410.06205](https://arxiv.org/abs/2410.06205): RoPE frequency roles, semantic drift, and p-RoPE.
- [Differential Transformer, arXiv:2410.05258](https://arxiv.org/abs/2410.05258): subtraction of two softmax attention maps for noise suppression; ICLR 2025 Oral.
- [Selective Attention, arXiv:2411.12892](https://arxiv.org/abs/2411.12892): learned control of attention sparsity and relevance.

## License

Apache-2.0 and MulanPSL-2.0; see the existing license files.
