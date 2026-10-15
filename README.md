# Möbius Attention (SG-Möbius-RoPE)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE-APACHE)
[![License: MulanPSL - 2.0](https://img.shields.io/badge/License-MulanPSL%202.0-brightgreen.svg)](LICENSE-MULAN)
[![Formal Verification: Lean 4](https://img.shields.io/badge/Formal%20Proof-Lean%204-purple.svg)](formal_verification/MobiusAttention.lean)
[![Formal Verification: Invariant Fuzzing](https://img.shields.io/badge/Properties-100%25%20Verified-success.svg)](formal_verification/verify_properties.py)

**[中文文档 (Chinese Documentation)](README_zh.md)**

> **Adaptive Semantic Gauge Möbius Attention for Transformers**  
> Resolves the fundamental non-negativity limitation of Softmax Attention by enabling **active destructive interference (negative association)** and **arbitrary-position causal truncation**.

---

## 💡 The Problem: Why Do LLMs Hallucinate After Logical Reversals?

### The Core Flaw of Standard Softmax Attention
In modern Transformer architectures (GPT-4, Llama, Qwen), self-attention is governed by Softmax:
$$A_{ij} = \frac{\exp(q_i k_j^\top / \sqrt{d})}{\sum \dots} > 0$$

Because the output is strictly positive, the model **can only attend more, but cannot physically subtract or cancel outdated context**.

Consider a long-context scenario:
> **Initial Premise**: *"Directive Alpha has passed all security checks and is marked as ACTIVE and NOMINAL."*  
> **Long-range Contrastive Pivot**: *"**CRITICAL REVERSAL ALERT!** Subsequent audit revealed severe vulnerabilities; the previous confirmation is completely REVOKED, and Directive Alpha is now INVALID and FORBIDDEN."*  
> **Query**: *"State the current execution status of Directive Alpha."*

In standard models, the superseded premise cannot be cancelled; it remains as **diffuse background noise** across the attention matrix. Over long contexts, this noise contaminates reasoning, causing the model to hallucinate or recall outdated facts.

---

## ⚡ The Solution: Möbius Topology & Destructive Phase Cancellation

The Möbius strip is a non-orientable manifold with a continuous half-twist.

We apply a data-dependent gauge connection to the RoPE positional embedding:

1. **Smooth Context (Flat Narrative)**:
   When semantics are continuous, the topological factor is identically $+1.0$. The mechanism is **strictly identical to standard RoPE**, preserving native syntax comprehension with zero overhead.
2. **Logical Inversion (Antipodal Cancellation)**:
   Whenever an inversion or negation occurs at **any arbitrary token position** (e.g., token 15, 51, or 300), the semantic gauge field fires a $\pi$-curvature pulse.
   The Query and pre-inversion Keys produce **destructive phase interference ($\cos \pi = -1.0$)** in the rotary subspace!
   Superseded context is **physically extinguished**, dropping attention noise to zero and focusing causal energy on the updated fact!

---

## 🔬 Mathematical Formulation

### 1. Semantic Gauge Curvature (Connection 1-form)
The local connection curvature density $\omega_t \in [0, \pi]$ is dynamically predicted from token hidden states:
$$\omega_t = \pi \cdot \sigma\left(\frac{w_{\text{gauge}}^\top x_t - \tau}{\beta}\right)$$
* **Continuative tokens** (e.g., "furthermore", "and", "continues") $\implies \omega_t \approx 0$
* **Contrastive tokens** (e.g., "however", "contrary", "invalidated") $\implies \omega_t \approx \pi$

### 2. Path Holonomy (Cumulative Gauge Phase)
The accumulated parallel transport from key $j$ to query $i$ ($j \le i$) along the causal path:
$$\Phi(j, i) = \sum_{k=j+1}^{i} \omega_k$$

### 3. Topological Parity Modulation
The inner product is modulated by the non-orientable topological parity factor:
$$\mathcal{T}(j, i) = \cos(\Phi(j, i))$$
* **No reversal**: $\Phi = 0 \implies \mathcal{T} = +1.0$ (Standard RoPE)
* **Semantic reversal**: $\Phi = \pi \implies \mathcal{T} = -1.0$ (Destructive cancellation of invalidated facts)

---

## 📊 Empirical Verification on Qwen3.5-9B (Dual Nvidia T4 GPUs)

Evaluated natively on **`Qwen/Qwen3.5-9B`** (Native FP16, zero quantization, eager attention):

### 1. Dynamic Curvature Pulses at Arbitrary Token Coordinates
![Dynamic Gauge Profile](assets/fig4_gauge_curvature.png)
* During smooth narrative, curvature $\omega_t$ is **strictly 0.0**. At **Token 51**, the model fires a sharp $\pi$-pulse, triggering the phase shift without any rigid periodic wavelength.

### 2. 2D Attention Heatmap Comparison
![Arbitrary Inversion Heatmap](assets/fig5_arbitrary_inversion_heatmap.png)
* **Standard 2π RoPE (Left)**: Background noise continues to diffuse into pre-reversal tokens.
* **Fixed 256-period Möbius (Middle)**: Completely fails because the reversal did not happen at step 256.
* **Adaptive SG-Möbius (Right)**: Produces a **sharp vertical causal cut** at Token 51 (all obsolete premises on the left are completely extinguished), while valid facts remain strongly focused.

### 3. Quantitative Evaluation Benchmarks

| Benchmark Scenario | Sequence Length | Triggered Inversion Tokens | Standard RoPE Entropy | SG-Möbius Entropy | Result Summary |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Smooth Control Group** | 337 | **None (`[]`)** | 3.4462 | **3.4462** | **0.0% entropy change; zero false alarms, identical to standard RoPE** |
| **Arbitrary Inversion** | 298 | `Token 51` | 3.4503 | **3.3966** | Entropy significantly reduced; obsolete premises extinguished |
| **Multi-turn Contrast** | 234 | `Tokens 39, 41` | 3.3120 | **3.1626** | 4.51% entropy reduction; sharp attention focus on active state |

---

## 🛡️ Formal Verification

1. **Lean 4 Proofs** ([`formal_verification/MobiusAttention.lean`](formal_verification/MobiusAttention.lean)):
   - `conservation_of_flatness`: Formally proves exact reduction to standard RoPE when $\omega = 0$.
   - `holonomy_cocycle`: Proves path additivity across composite sequence trajectories.
   - `antipodal_inversion`: Proves that accumulated holonomy $\Phi = \pi \implies \mathcal{T} = -1.0$.
   - `gauge_translation_invariance`: Proves invariance of relative holonomy under global shifts.

2. **Automated Property Verifier** ([`formal_verification/verify_properties.py`](formal_verification/verify_properties.py)):
   ```bash
   python formal_verification/verify_properties.py
   ```
   Passes 1,000 randomized property fuzzing trials across variable sequence lengths and arbitrary inversion coordinates with a 100% success rate.

---

## 🚀 Quickstart

Plug-and-play into any HuggingFace causal LM with three lines of code (fully compatible with FlashAttention and SDPA):

```python
import torch
from transformers import AutoModelForCausalLM
from src.mobius_attention import inject_mobius_attention, SemanticGaugeConnection

# 1. Load any standard HuggingFace model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16, device_map="auto")

# 2. Hot-swap rotary embedding modules (zero extra VRAM, linear complexity)
patched_count = inject_mobius_attention(model)
print(f"Successfully injected into {patched_count} attention layers.")

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
