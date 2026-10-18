# Project: Semantic Gauge Möbius Attention (SG-Möbius-RoPE)
## Continuous Subspace Anti-Phase Rotary Modulation for LLMs

> **Status**: Completed, Formally Verified, and Publication-Ready (NeurIPS / ICLR Format)  
> **Target Model**: Qwen3.5-9B  
> **Evaluation Span**: $L \in [545, 18,030]$ tokens  
> **Test Suite**: 407 / 407 passed (100%)  
> **Formal Verification**: Lean 4 (9 theorems, 0 sorry, 0 custom axioms, 1946 jobs)

---

## Architecture Summary

SG-Möbius-RoPE augments Rotary Position Embedding (RoPE) with continuous orthogonal subspace phase modulation to resolve strictly positive Softmax attention leakage under premise invalidation:

1. **Subspace Orthogonal Decomposition**:
   $$\mathbb{R}^d = \mathcal{S}_{\text{anchor}} \oplus \mathcal{S}_{\text{mob}}$$
   - $\mathcal{S}_{\text{anchor}}$: High-frequency channels ($m \in \mathcal{I}_{\text{anchor}}$) retain standard static RoPE rotation ($\omega \equiv 0$), guaranteeing strict syntax, punctuation, and local grammar stability ($0.0\%$ WikiText-103 perplexity degradation, 7.48).
   - $\mathcal{S}_{\text{mob}}$: Low-frequency semantic channels ($m \in \mathcal{I}_{\text{mob}}$) undergo continuous causal phase modulation $\Phi_t = \sum_{\tau \le t} \omega(\tau)$.

2. **Continuous Rotary Dual Modulation**:
   $$\tilde{q}_i^{(m)} = \mathbf{R}(i\theta_m + \Phi_m(i)) q_i^{(m)}, \qquad \tilde{k}_j^{(m)} = \mathbf{R}(j\theta_m + \Phi_m(j)) k_j^{(m)}$$
   Preserves exact relative translation invariance:
   $$\langle \tilde{q}_i^{(m)}, \tilde{k}_j^{(m)} \rangle = (q_i^{(m)})^\top \mathbf{R}((j - i)\theta_m + \Delta\Phi_{j \to i}^{(m)}) k_j^{(m)}$$

3. **Discourse Transition Detector (Definition 1)**:
   $$z_t = w_g^\top h_t + b_g, \qquad g(h_t) = \omega_{\max} \cdot \mathrm{softsign}\Big(\mathrm{ReLU}\big(|z_t| - \tau_{\text{gate}}\big)\mathrm{sgn}(z_t)\Big)$$
   where $w_g \propto \mu_{\text{rev}} - \mu_{\text{cont}}$ is derived from pretrained unembedding token centroids. Dead-zone barrier $\tau_{\text{gate}} > 0$ quenches false activations on standard text.

4. **1D Viscous Flow & Phase-Locking Attractor**:
   $$\omega_t = (1 - \nu)\omega_{t-1} + \nu g(h_t), \qquad \Phi_t = \bar{\Phi}_t - \eta\sin(2\bar{\Phi}_t)$$
   Quenches numerical floating-point drift and smoothly snaps phase accumulation to $\pi$ (destructive cancellation) or $2\pi$ (cyclic negation-of-negation recovery).

5. **Prefix Partition Contraction & Probability Mass Siphoning (Corollary 4)**:
   Collective phase cancellation across superseded prefix $\mathcal{P}$ ($|\mathcal{P}| = 576$ tokens including sentence-initial Attention Sink $l=0$) contracts partition function by $Z^{\mathrm{std}}/\tilde{Z} \approx 3.48\times$. This siphons probability mass into operative targets ($P(\text{target})$ amplified $3.49\times$) while cushioning apparent decoy probability drop ($1.62\times$), driving multi-head ACR $+465.1\%$ ($3.62 \to 20.44$).

6. **KV-Cache Immutability (Proposition 5)**:
   Session-level anchor cutoff $\alpha(L_{\max})$ is fixed at session initialization. Past keys $j < t$ remain byte-for-byte immutable in cache memory, ensuring strict $\mathcal{O}(1)$ autoregressive decoding.

7. **Higham FP16 Forward Stability Bound (Proposition 6)**:
   Under operational benchmark regime of up to $K \le 10$ discourse turns ($M_\Phi \le 10\pi$), tree-depth prefix scan in FP16 ($u = 2^{-11}$) yields $|\Delta\bar{\Phi}_L| \le 0.230\,\text{rad} < \pi/6$, residing strictly inside the phase-locking attractor basin.

---

## Codebase Organization

```
.
├── src/                                  # Core PyTorch Implementation
│   ├── gauge_predictor.py                # Definition 1 softsign detector, centroids, viscous EMA
│   ├── mobius_attention.py               # Subspace rotary modulation layer
│   └── kv_cache.py                       # Linear incremental KV-Cache manager
├── formal_verification/                  # Lean 4 & Z3 Verification
│   ├── MobiusAttention.lean              # 9 core theorems (0 sorry, 0 custom axioms)
│   ├── lakefile.toml                     # Lake build configuration (Lean 4.33.1)
│   └── verify_properties.py              # Z3 SMT UNSAT proof & 2000-trial property fuzzing
├── tests/                                # 407 End-to-End Tests across 4 Tiers
│   ├── test_gauge_gate_definition1.py    # Definition 1 softsign dead-zone tests
│   ├── test_prefill_decoding_equivalence.py # Machine-precision prefill vs decode (< 10^-12)
│   ├── test_tier1_features.py            # Feature unit tests (60 tests)
│   ├── test_tier2_boundaries.py          # Boundary tests (60 tests)
│   ├── test_tier3_combinations.py        # Complex combinations (12 tests)
│   ├── test_tier4_scenarios.py           # Real-world long-context scenarios (5 tests)
│   └── ...                               # Robustness, adversarial, NIAH tests (270 tests)
├── paper/                                # Publication Manuscripts (NeurIPS / ICLR)
│   ├── main.tex                          # English master paper (521 lines)
│   ├── main_zh.tex                       # Chinese master paper (506 lines)
│   ├── references.bib                    # Verified bibliography (15 citations, zero-hallucination)
│   ├── main.pdf                          # Compiled English preprint PDF
│   └── main_zh.pdf                       # Compiled Chinese preprint PDF
├── scripts/                              # Zero-Trust Audit Tools & Figure Generators
│   ├── verify_paper_claims.py            # SSOT claim, terminology, and arithmetic auditor
│   ├── check_paper_integrity.py          # LaTeX citation & label integrity linter
│   ├── check_markdown_math.py            # Markdown math syntax linter
│   ├── render_scaling_figures.py         # Figure 1(a), 4(a) scaling data renderer
│   └── plot_english_figures.py           # 300 DPI publication heatmap renderer
├── experiments/                          # Ground Truth Benchmark Data
│   └── kaggle_experiment_results.json    # Verified Qwen3.5-9B benchmark metrics
├── AI_HANDOVER.md                        # Master Handover Guide for AI Assistants
├── AGENTS.md                             # Autonomous Agent Operational Directives
├── .cursorrules                          # Cursor AI Editor Configuration
└── .github/workflows/ci.yml              # 8-job Parallel GitHub Actions CI Pipeline
```

---

## Milestone Execution Matrix (All Completed)

| Milestone | Scope | Deliverables | Verification | Status |
| :--- | :--- | :--- | :--- | :---: |
| **M1: Mathematical Rigor & Lean 4** | Theorems 1-3, Corollary 4, Subspace Decoupling | `MobiusAttention.lean`, Lake configuration | `lake build` (1946 jobs, 0 sorry) | ✅ COMPLETED |
| **M2: Detector Implementation & KV-Cache** | Definition 1 softsign gate, $\tau_{\text{gate}}$ dead-zone, centroid constructor, Proposition 5 | `src/gauge_predictor.py`, `src/kv_cache.py` | Prefill/decode equivalence ($< 10^{-12}$) | ✅ COMPLETED |
| **M3: Adversarial Benchmarks & Kaggle Run** | Inverted NIAH, $L \in [545, 18,030]$, Table 1-3, Figure 1-6 | `experiments/kaggle_experiment_results.json`, `assets/` | Qwen3.5-9B on Dual T4 (FP16) | ✅ COMPLETED |
| **M4: Publication Preprint & Bilingual Sync** | 18-page English & Chinese preprints, BibTeX anti-hallucination | `paper/main.tex`, `paper/main_zh.tex`, `paper/*.pdf` | GitHub Actions CI LaTeX compile | ✅ COMPLETED |
| **M5: Zero-Trust Auditing & Linter Gates** | Terminology linter, SSOT caption sync, Higham bound qualification | `scripts/verify_paper_claims.py`, `scripts/check_paper_integrity.py` | 5 audit stages passed | ✅ COMPLETED |
| **M6: Full Test Suite & CI/CD Hardening** | 407 tests, 8 parallel CI jobs, ORCID integration | `.github/workflows/ci.yml`, `tests/` | 407 passed in 25s, CI green | ✅ COMPLETED |

---

## Maintenance Directives

For maintenance instructions, forbidden anti-patterns, and step-by-step verification commands, refer to:
- [`AI_HANDOVER.md`](file:///AI_HANDOVER.md) (Master Handover Guide)
- [`AGENTS.md`](file:///AGENTS.md) (Operational Directives for Coding Agents)
