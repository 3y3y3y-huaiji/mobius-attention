# E2E Test Infra: Semantic Gauge Möbius Attention (SG-Möbius-RoPE)

## Test Philosophy
- Opaque-box, requirement-driven derived strictly from `ORIGINAL_REQUEST.md`.
- Independent of internal implementation details.
- 4-Tier test methodology:
  * Tier 1: Feature Coverage (>=5 per feature)
  * Tier 2: Boundary & Corner Cases (>=5 per feature)
  * Tier 3: Cross-Feature Combinations (pairwise interaction coverage)
  * Tier 4: Real-World Application Scenarios (end-to-end full context tasks)

## Feature Inventory & Test Mapping
| # | Feature | Requirement Source | Tier 1 (Count) | Tier 2 (Count) | Tier 3 (Pairs) |
|---|---------|-------------------|:--------------:|:--------------:|:--------------:|
| F1 | Gauge Connection & Discrete Holonomy | R1 | 5 | 5 | ✓ |
| F2 | Lean 4 Theorem Clean Compilation | R1 | 5 | 5 | ✓ |
| F3 | Property Fuzzing Invariant Preservation | R1 | 5 | 5 | ✓ |
| F4 | Antagonistic Sequences (Negations/Quotes) | R2.1 | 5 | 5 | ✓ |
| F5 | Cross-Architecture Generalization (Qwen/Llama) | R2.2 | 5 | 5 | ✓ |
| F6 | Long-Context Scaling (8k - 32k) | R2.3 | 5 | 5 | ✓ |
| F7 | Neutral Text Perplexity Preservation (<0.1%) | R2.3 | 5 | 5 | ✓ |
| F8 | Linear KV-Cache Decoding Scaling | R2.3 | 5 | 5 | ✓ |
| F9 | Inverted Needle In A Haystack (NIAH) | R2.4 | 5 | 5 | ✓ |
| F10 | ArXiv Preprint Paper Drafting & Compilability | R3 | 5 | 5 | ✓ |
| F11 | High-Resolution Vector Figures (Assets) | R3 | 5 | 5 | ✓ |
| F12 | Complete Academic Bibliography | R3 | 5 | 5 | ✓ |

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Legal Contract Clause Invalidation (32k context) | F1, F4, F6, F7, F8, F9 | High |
| 2 | Medical Diagnostic Conflict & Reversal Analysis | F1, F4, F5, F7, F9 | High |
| 3 | Software Vulnerability Revocation Trace (16k context) | F1, F6, F8, F9 | High |
| 4 | Multi-Turn Conversational Negotiation with Retractions | F1, F4, F5, F8 | Medium |
| 5 | Cross-Lingual Knowledge Inversion (Qwen2.5 vs Llama-3) | F4, F5, F7, F9 | High |

## Test Architecture
- Test Runner: Python pytest test suite invoking CLI / module test entrypoints.
- Verification semantics: Exit code 0, 100% assertions satisfied, zero unproven axioms in Lean 4.
- Directory Layout:
  `tests/` contains all unit and end-to-end test cases.
