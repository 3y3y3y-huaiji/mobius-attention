# AGENTS.md - Autonomous AI Agent Operational Protocol

> **CRITICAL DIRECTIVE FOR ALL AI AGENTS**:  
> Read [AI_HANDOVER.md](file:///AI_HANDOVER.md) before making any modifications to this repository.  
> This repository implements **Subspace Anti-Phase Attention (SG-Möbius-RoPE)** on Qwen3.5-9B. It is under strict Zero-Trust regression audit for top-tier publication (NeurIPS / ICLR).

---

## 1. Prime Directives (Non-Negotiable)

1. **Zero Hallucination Policy**:
   - Never invent line numbers, functions, or file paths.
   - Actual file lengths:
     - `src/gauge_predictor.py`: 612 lines
     - `paper/main.tex`: 521 lines
     - `tests/test_gauge_gate_definition1.py`: 97 lines
     - `formal_verification/MobiusAttention.lean`: 9 theorems
   - Never cite line numbers beyond the actual length of the file.

2. **Definition 1 Gating Integrity**:
   - The discourse transition detector must evaluate the dead-zone threshold on the **unscaled scalar projection** $z_t = w_g^\top h_t + b_g$:
     $$g(h_t) = \omega_{\max} \cdot \mathrm{softsign}\Big(\mathrm{ReLU}\big(|z_t| - \tau_{\text{gate}}\big) \cdot \mathrm{sgn}(z_t)\Big)$$
   - NEVER place the gate thresholding after $\omega_{\max} \tanh(z)$, as this causes a fatal zero-lock bug when $\omega_{\max} = 0.05 < 0.50 = \tau_{\text{gate}}$.

3. **Distinguish Detector Gate vs Phase Velocity**:
   - With recommended parameters ($\omega_{\max} = 0.05, \tau_{\text{gate}} = 0.50, \nu = 0.5$):
     - For super-threshold projection $z = 1.0$: raw gate output $g(h_t) = 0.01667$ rad/step.
     - Smoothed phase velocity at initial step: $\omega_t = \nu g(h_t) = 0.00833$ rad/step.
   - Do NOT conflate raw detector output with smoothed velocity.

4. **Single Source of Truth (SSOT)**:
   - All claims across `README.md`, `README_zh.md`, `paper/main.tex`, and `paper/main_zh.tex` must strictly match:
     - Context evaluation span: $L \in [545, 18,030]$
     - ACR amplification: $+465.1\%$ ($3.62 \to 20.44$)
     - Decoy suppression ratio: $1.62\times$
     - Target amplification ratio: $3.49\times$
     - WikiText-103 perplexity: $7.48$ ($0.0\%$ degradation)
     - Test suite: **407 passed**

5. **Formal Verification Guardrails**:
   - `formal_verification/MobiusAttention.lean` must maintain:
     - 0 `sorry`
     - 0 unproven custom axioms
     - 100% build pass via `lake build` (1946 jobs)
   - Lean 4 proves real-number algebraic theorems and 1D Lyapunov stability. Floating-point error is bounded analytically by Proposition 6.

6. **Proposition Numbering**:
   - Proposition 5: Prefill vs Autoregressive KV-Cache Decoding Equivalence.
   - Proposition 6: Two-Stage Parallel Associative Scan and Higham FP16 Stability Bound ($M_\Phi \le 10\pi$ for up to 10 narrative turns; FP32 mixed precision bounds drift to $< 10^{-5}$ rad for $L \le 10^6$).

---

## 2. Mandatory Verification Pipeline

Before committing or completing any turn, run the following verification pipeline in powershell:

```powershell
# 1. Full 407-test suite
pytest tests/ -n auto -q

# 2. Zero-trust paper claim & terminology linter
python scripts/verify_paper_claims.py

# 3. Paper LaTeX integrity & citation anti-hallucination check
python scripts/check_paper_integrity.py paper/main.tex
python scripts/check_paper_integrity.py paper/main_zh.tex

# 4. Markdown LaTeX syntax check
python scripts/check_markdown_math.py README.md
python scripts/check_markdown_math.py README_zh.md
pytest tests/test_readme_latex.py

# 5. Lean 4 formal proof build
cd formal_verification; lake build; cd ..

# 6. Python code quality
ruff check src/ tests/ scripts/
```

All 6 commands must pass with 0 errors.
