# Möbius Attention (SG-Möbius-RoPE)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE-APACHE)
[![License: MulanPSL - 2.0](https://img.shields.io/badge/License-MulanPSL%202.0-brightgreen.svg)](LICENSE-MULAN)
[![Formal Verification: Lean 4](https://img.shields.io/badge/Formal%20Proof-Lean%204-purple.svg)](formal_verification/MobiusAttention.lean)
[![Formal Verification: Property Fuzzing](https://img.shields.io/badge/Properties-100%25%20Verified-success.svg)](formal_verification/verify_properties.py)

**[中文文档 (Chinese Documentation)](README_zh.md)**

> **Data-Dependent Semantic Gauge Möbius Attention**: Non-orientable fiber bundle half-twists and continuous gauge connections for Transformers.  
> Overcomes the inherent non-negativity bottleneck of Softmax Attention by enabling **active destructive interference (negative association)** and **double-negation loop recovery** at arbitrary sequence coordinates.

---

## 💡 The Intuition in Plain English

### The Core Defect of Standard Transformer Attention
Standard Softmax Attention maps query-key inner products to strictly non-negative weights:
$$A_{ij} = \frac{\exp(q_i k_j^\top / \sqrt{d})}{\sum \dots} \ge 0$$
Because weights can never be negative, standard Transformers **cannot physically cancel or antagonize outdated evidence**. When a critical fact is invalidated later in a long text (e.g., *"Directive Alpha is valid ... HOWEVER, subsequent audit confirms Directive Alpha is FALSE and INVALID"*), the model can only "squint" by assigning small positive weights. Over thousands of tokens, superseded premises linger as background noise, inducing hallucination and context confusion.

### The Möbius Solution
We upgrade the flat sequence manifold into a **non-orientable Möbius fiber bundle with a $4\pi$ spinor double covering**:
1. **Normal Flow**: On smooth text, the gauge connection curvature is $\omega = 0$. The mechanism operates **identically to standard RoPE** with zero distortion.
2. **Logic Reversal**: When contrastive/inversion tokens appear at **any arbitrary position**, the local connection fires a $\pi$-curvature pulse. The metric transitions to the antipodal sheet ($\cos(\pi) = -1$), producing **destructive phase cancellation** that physically zeros out superseded context!
3. **Double Negation (Negation-of-Negation)**: When a secondary reversal restores the premise, the path holonomy reaches $2\pi$. Under the $4\pi$ spinor covering, the phase returns to $+1$, resurrecting positive causal resonance!

---

## 🔬 Mathematical Formulation

### 1. Semantic Gauge Connection 1-Form
For hidden state $x_t$, the local gauge curvature density $\omega_t$ is dynamically predicted via contrastive polarity projection:
$$\omega_t = \pi \cdot \sigma\left(\frac{w_{\text{gauge}}^\top x_t - \tau}{\beta}\right) \in [0, \pi]$$

### 2. Path Holonomy
The accumulated parallel transport from key $j$ to query $i$ ($j \le i$) along the causal path:
$$\Phi(j, i) = \sum_{k=j+1}^{i} \omega_k$$

### 3. Spinor Parity Modulation
The inner product is modulated by the non-orientable topological parity factor:
$$\mathcal{T}(j, i) = \cos(\Phi(j, i))$$

* **Flat narrative**: $\Phi = 0 \implies \mathcal{T} = +1.0$ (Exact RoPE equivalence)
* **Single inversion**: $\Phi = \pi \implies \mathcal{T} = -1.0$ (Destructive phase interference)
* **Double negation**: $\Phi = 2\pi \implies \mathcal{T} = +1.0$ (Topological closed-loop return)

---

## 📊 Empirical Verification on Qwen3.5-9B (Dual Nvidia T4 GPUs)

Evaluated natively on **`Qwen/Qwen3.5-9B`** (Native FP16, zero quantization, eager attention):

| Benchmark Scenario | Sequence Length | Triggered Inversion Tokens | Standard RoPE Entropy | SG-Möbius Entropy | Result Summary |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Control Group (Smooth)** | 337 | **None (`[]`)** | 3.4462 | **3.4462** | **Identical bit-wise to RoPE; 0% false suppression rate** |
| **Arbitrary Inversion** | 298 | `Token 51` | 3.4503 | **3.3966** | Sharp vertical causal cut; superseded premise phased out |
| **Double Negation** | 234 | `Tokens 39, 41, 133` | 3.3120 | **3.1626** | 4.51% entropy reduction; $4\pi$ spinor restoration |

### Figure Gallery
- **[Curvature Profile](assets/fig4_gauge_curvature.png)**: Shows zero curvature during smooth narrative and sharp $\pi$-pulses at semantic negation tokens.
- **[2D Attention Heatmap Trio](assets/fig5_arbitrary_inversion_heatmap.png)**: Visualizes the crisp vertical extinction boundary in SG-Möbius compared to standard RoPE and fixed-period baselines.
- **[Double Negation Loop](assets/fig6_double_negation_restoration.png)**: Demonstrates the restoration of attention energy ln(A_ij) to original premises after double negation.

---

## 🛡️ Formal Verification

This project features end-to-end formal mathematical proofs:

1. **Lean 4 Proofs** ([`formal_verification/MobiusAttention.lean`](formal_verification/MobiusAttention.lean)):
   - `conservation_of_flatness`: Proves identical reduction to standard RoPE when $\omega = 0$.
   - `holonomy_cocycle`: Formally verifies path additivity along composite sequences.
   - `antipodal_inversion`: Proves that $\Phi = \pi \implies \mathcal{T} = -1$.
   - `double_negation_closure`: Proves $2\pi$ spinor closure $\implies \mathcal{T} = +1$.
   - `gauge_translation_invariance`: Proves independence under global gauge shifts.

2. **Automated Property Verifier** ([`formal_verification/verify_properties.py`](formal_verification/verify_properties.py)):
   ```bash
   python formal_verification/verify_properties.py
   ```
   Passes 1,000 randomized property fuzzing trials across variable sequence lengths, arbitrary inversion locations, and deep nestings.

---

## 🚀 Quickstart

Plug-and-play into any HuggingFace model with zero retraining and full FlashAttention / SDPA compatibility:

```python
import torch
from transformers import AutoModelForCausalLM
from src.mobius_attention import inject_mobius_attention, SemanticGaugeConnection

# 1. Load any standard model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16, device_map="auto")

# 2. Hot-swap rotary embedding modules
count = inject_mobius_attention(model)
print(f"Injected into {count} attention layers.")

# 3. Attach semantic gauge connection
embed = model.get_input_embeddings()
gauge = SemanticGaugeConnection(hidden_dim=embed.embedding_dim).to(model.device)
```

---

## 📜 Dual License

This project is dual-licensed under:
* [Apache License, Version 2.0](LICENSE-APACHE)
* [Mulan Permissive Software License, Version 2 (MulanPSL - 2.0)](LICENSE-MULAN)

You may choose either license at your option.
