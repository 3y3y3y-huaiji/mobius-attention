# Project: Semantic Gauge Möbius Attention (SG-Möbius-RoPE)
Academic Peer Review, Adversarial Stress-Testing & ArXiv Technical Report

## Architecture
SG-Möbius-RoPE augments Rotary Position Embedding (RoPE) with a non-orientable fiber bundle connection. The semantic gauge connection defines a discrete $\mathbb{Z}_2$ holonomy phase $\Phi_t \in \{0, \pi\}$ along causal sequence paths. 
- **Discrete Gauge Factorization**: Because $\omega_t \in \{0, \pi\}$, the relative phase modulates attention query-key inner products via $\cos(\Phi_i - \Phi_j) = \cos(\Phi_i)\cos(\Phi_j) = s_i s_j$, where $s_t = \cos(\Phi_t) \in \{-1, +1\}$.
- **Dual Rotary Modulation**: Applying sign $s_t$ to both $\cos$ and $\sin$ tensors (or equivalently rotating the 2D vector by $\theta_t + \Phi_t$) preserves relative translation invariance $\langle \tilde{q}_i, \tilde{k}_j \rangle = s_i s_j \langle q_i, k_j \rangle_{\text{RoPE}}$ with 0.00 mathematical error.
- **KV-Cache Compatibility**: Since modulation is a local rank-1 sign factor $s_t$, past key-value caches do not require back-propagation or global $O(N^2)$ recomputation upon semantic inversion.
- **Refractory Semantic Predictor**: Semantic inversion detection employs refractory hysteresis to prevent spurious multi-word parity oscillations across long prompts.

## Code Layout
```
c:\Users\安卓人\Documents\antigravity\算法研究\
├── formal_verification/
│   ├── MobiusAttention.lean        # Lean 4 formal mathematical theorems (0 sorry, 0 axioms)
│   ├── lakefile.toml               # Lake package specification
│   ├── lean-toolchain              # Lean 4 toolchain pin (4.33.1)
│   ├── verify_properties.py        # 1000+ randomized property fuzzing & Z3 SMT verification
│   └── test_rigor_verification.py  # 2000-trial extreme-length fuzzing (up to 32k)
├── src/
│   ├── mobius_attention.py         # PyTorch SG-Möbius-RoPE attention layer (dual cos/sin modulation)
│   ├── gauge_predictor.py          # Hysteresis-gated semantic gauge connection predictor
│   └── kv_cache.py                 # Linear KV-cache incremental decoding handler
├── mobius_rope.py                  # Standalone reference rotary embedding module
├── tests/
│   ├── test_antagonistic.py        # Antagonistic edge cases (nested negations, quotes, contrastive)
│   ├── test_cross_arch.py          # Cross-architecture benchmarks (Qwen2.5 and Llama-3)
│   ├── test_long_context_niah.py   # 8k-32k scalability, ACR, perplexity preservation, Inverted NIAH
│   └── test_kv_linearity.py        # O(N) memory scaling & KV-cache linear decoding tests
├── paper/
│   ├── main.tex                    # ArXiv preprint LaTeX master file (NeurIPS 2024 style)
│   ├── neurips_2024.sty            # Conference style package
│   ├── references.bib              # Complete academic bibliography (27 foundational references)
│   ├── sections/                   # Modular sections (01_intro to 09_conclusion)
│   ├── figures/                    # High-res English vector figures (fig1-fig6 PDF/SVG)
│   ├── compile_paper.bat           # Local compilation script
│   └── package_paper_zip.py        # Overleaf / ArXiv source bundle generator
├── scripts/
│   └── plot_paper_figures_en.py    # English vector figure generation from raw npz tensors
└── assets/                         # Legacy & reference assets
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Discrete $\mathbb{Z}_2$ Holonomy Formulation | Continuous/discrete fiber bundle theory, causal Wilson line $\Phi(j, i)$, rank-1 sign factorization | M1 | Survey R1 |
| 2 | Dual Rotary Injection Fix | Scaling both $\cos$ and $\sin$ by $s_t = \cos(\Phi_t)$ to restore exact relative position translation invariance | M1 | Survey R1/R2 |
| 3 | Lean 4 Proof Verification | Lake workspace setup and compilation of `formal_verification/MobiusAttention.lean` (zero `sorry`, zero unproven axioms) | M1 | Survey R1 |
| 4 | Property Fuzzing & Z3 SMT Suite | 2,000+ randomized trials covering extreme lengths (up to 32k) and arbitrary reversal points, plus Z3 proofs | M1 | Survey R1 |
| 5 | Hysteresis Semantic Predictor | Refractory window and stateful gating to eliminate multi-trigger parity oscillation | M2 | Survey R2 |
| 6 | Linear KV-Cache Incremental Decoding | Correct autoregressive token-by-token slicing for $O(1)$ decoding step | M2 | Survey R2 |
| 7 | Antagonistic Stress Suite | Stress testing on nested negations, negations in quotes, and consecutive contrastive tokens | M2 | Survey R2 |
| 8 | Cross-Architecture Generalization | Benchmark across Qwen2.5 and Llama-3 base architectures | M2 | Survey R2 |
| 9 | Long-Context Scalability (8k - 32k) | Verify linear memory scaling and zero KV degradation across 8k to 32k sequence lengths | M2 | Survey R2 |
| 10 | Attention Contrast Ratio (ACR) | Quantitative ACR improvement of SG-Möbius over standard RoPE | M2 | Survey R2 |
| 11 | Neutral Text Perplexity Preservation | Confirm < 0.1% perplexity degradation on neutral texts (zero false-suppression) | M2 | Survey R2 |
| 12 | Inverted Needle In A Haystack (NIAH) | Benchmark on conflicting premises and mid-context reversals | M2 | Survey R2 |
| 13 | English Publication-Grade Vector Figures | Generate Figures 1-6 in high-resolution PDF/SVG format with English typography | M3 | Survey R3 |
| 14 | LaTeX Master & Modular Sections | Scaffold compilable preprint (`paper/main.tex` and 9 sections) adhering to NeurIPS/ICLR formatting | M3 | Survey R3 |
| 15 | Complete Bibliography | 27-entry bibtex file (`paper/references.bib`) covering RoPE, gauge theory, context cancellation, and non-Euclidean attention | M3 | Survey R3 |
| 16 | Compilation & Packaging Pipeline | Overleaf bundle creator and compilation automation scripts | M3 | Survey R3 |
| 17 | E2E Regression & Adversarial Hardening | Full requirement-driven 4-tier E2E testing followed by Tier 5 adversarial coverage hardening | M4 | Survey R1-R3 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Mathematical Rigor & Lean 4 Verification | Features 1, 2, 3, 4: Rotary dual-modulation fix, Lean 4 Lake setup, 2000+ trial fuzzing & Z3 proofs | none | PLANNED |
| M2 | Adversarial Stress-Testing & Benchmarks | Features 5, 6, 7, 8, 9, 10, 11, 12: Hysteresis predictor, KV decoding, antagonistic suite, Qwen/Llama-3, 8k-32k ACR/PPL, Inverted NIAH | M1 | PLANNED |
| M3 | ArXiv Preprint Paper Drafting & Assets | Features 13, 14, 15, 16: English vector figures (fig1-fig6), LaTeX template, 9 modular sections, complete bibliography, packaging pipeline | M1 | PLANNED |
| M4 | Final Milestone: E2E Pass & Hardening | Feature 17: Phase 1 pass 100% of E2E test suite (Tiers 1-4); Phase 2 adversarial coverage hardening (Tier 5) | M1, M2, M3 | PLANNED |

## Interface Contracts
### Gauge Connection $\leftrightarrow$ Attention Layer
- Function: `apply_mobius_rope(q, k, cos, sin, phi)`
- Input: `q, k`: `[batch, heads, seq_len, head_dim]`; `cos, sin`: `[seq_len, head_dim]`; `phi`: `[batch, seq_len]`
- Sign factor: $s = \cos(\phi) \in \{-1, +1\}$
- Output: `(q_rot, k_rot)` where $\tilde{q} = s_i \mathcal{R}_{\theta, i} q$, $\tilde{k} = s_j \mathcal{R}_{\theta, j} k$
- Mathematical Invariant: $\langle \tilde{q}_i, \tilde{k}_j \rangle = s_i s_j \langle q_i, k_j \rangle_{\text{RoPE}} = \cos(\Phi_i - \Phi_j) \langle q_i, k_j \rangle_{\text{RoPE}}$

### Gauge Predictor $\leftrightarrow$ KV-Cache
- Method: `step(token_emb, past_phi)`
- Invariant: $O(1)$ time complexity per token; returns incremental phase $\phi_t = \phi_{t-1} + \omega_t$ with hysteresis suppression.
