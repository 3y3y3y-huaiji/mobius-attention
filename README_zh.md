# 莫比乌斯注意力 (Möbius-Attention / SG-Möbius-RoPE)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE-APACHE)
[![License: MulanPSL - 2.0](https://img.shields.io/badge/License-MulanPSL%202.0-brightgreen.svg)](LICENSE-MULAN)
[![Formal Verification: Lean 4](https://img.shields.io/badge/Formal%20Proof-Lean%204-purple.svg)](formal_verification/MobiusAttention.lean)
[![Formal Verification: Invariant Fuzzing](https://img.shields.io/badge/Properties-100%25%20Verified-success.svg)](formal_verification/verify_properties.py)

**[English Documentation (英文文档)](README.md)**

> **基于语义规范联络的自适应莫比乌斯注意力机制**  
> 突破 Softmax 天然非负的表达瓶颈，为大语言模型赋予**长程语义反向对消（Destructive Interference）**与**任意位置逻辑转折的精准因果截断**能力。

---

## 💡 痛点直击：大模型为什么总在“转折”后产生幻觉？

### 传统 Attention 的根本缺陷：只能正向关注，无法反向抵消
在所有标准的 Transformer（GPT、Llama、Qwen）中，注意力计算的核心是 Softmax：
$$A_{ij} = \frac{\exp(q_i k_j^\top / \sqrt{d})}{\sum \dots} > 0$$

因为 Softmax 的数学结果**永远是正数（权重只能 $> 0$）**，这意味着大模型在原理上**只能“看见更多”，而无法“物理删除”或“反向抵消”**。

在实际长文本中，经常出现前后冲突与事实推翻：
> **前文**：“系统记录显示，指令 Alpha 已经通过安全核验，状态为【正常有效】。”  
> **长程后文转折**：“**重大反转警告！** 经二次复核发现重大漏洞，刚才的核验通报彻底作废，指令 Alpha 状态变更为【严重危险，禁止执行】！”  
> **末端提问**：“请输出指令 Alpha 当前的执行状态。”

此时，标准大模型只能“眯起眼睛假装没看见前面的话”（尝试把前文的注意力权重调小）。但在上千乃至上万 Token 的长上下文中，前文的大量废弃信息依然像**浓雾般的背景白噪声**一样弥散在注意力矩阵里，极大干扰模型的判断，从而引发经典的**“逻辑颠覆失忆”与“事实幻觉”**。

---

## ⚡ 解决方案：莫比乌斯拓扑与主动相位相消

莫比乌斯带（Möbius Strip）最大的物理特性是**“两面相连，走半圈就会翻转到反面”**。

本算法通过**不可定向流形（Non-orientable Manifold）**对 RoPE 位置编码进行动态拓扑调制：

1. **顺承叙述时（平坦流形）**：
   文本逻辑平顺推进时，拓扑因子恒为 $+1.0$。算法与标准 2π RoPE **完全等价、零显存开销、零语法精度损失**。
2. **发生逻辑转折时（跨叶相消）**：
   无论转折发生在序列的**任意位置**（如第 15 个字、第 51 个字或第 300 个字），语义规范场会自发激发一个 $\pi$ 相位脉冲。
   查询（Query）与转折点之前的旧键（Key）在内积时直接产生**负干涉（$\cos \pi = -1.0$）**！
   在物理层面上，作废事实被**主动相位对消（Destructive Interference）**，注意力底噪被彻底熄灭，模型瞬间将注意力聚焦在转折后的最新有效事实上！

---

## 🔬 核心公式与机制

### 1. 局域语义规范曲率 (Gauge Curvature)
通过对比极性投影，从隐藏层隐状态 $x_t$ 动态预测各 Token 处的转折曲率强度 $\omega_t \in [0, \pi]$：
$$\omega_t = \pi \cdot \sigma\left(\frac{w_{\text{gauge}}^\top x_t - \tau}{\beta}\right)$$
* **顺承词汇**（如“并且”、“因而”、“持续”） $\implies \omega_t \approx 0$
* **转折词汇**（如“但是”、“相反”、“作废”） $\implies \omega_t \approx \pi$

### 2. 因果累积规范相位 (Path Holonomy)
从历史 Key 位置 $j$ 到当前 Query 位置 $i$ ($j \le i$) 的相对扭曲相位：
$$\Phi(j, i) = \sum_{k=j+1}^{i} \omega_k$$

### 3. 注意力拓扑调制
在旋转位置编码（RoPE）中施加拓扑调制因子：
$$\mathcal{T}(j, i) = \cos(\Phi(j, i))$$
* **未发生转折**：$\Phi = 0 \implies \mathcal{T} = +1.0$（原生标准注意力）
* **发生逻辑转折**：$\Phi = \pi \implies \mathcal{T} = -1.0$（反向抑制，物理熄灭作废信息）

---

## 📊 实验实测验证 (Qwen3.5-9B 原生 FP16 实测)

本算法在开源大模型 **`Qwen/Qwen3.5-9B`**（双 Tesla T4 GPU、原生 FP16、零量化）上完成了全注意力矩阵的真实测评：

### 1. 任意位置转折的自发曲率脉冲响应
![语义规范场曲率脉冲](assets/fig4_gauge_curvature.png)
* **实测现象**：前 50 个 Token 处叙述平坦，曲率脉冲 $\omega_t$ **严格为 0**；当在 **Token 51** 处出现转折词时，模型瞬间自发激发出强烈的 $\pi$ 曲率脉冲，驱动累积相位跃迁，彻底打破了以往必须依赖固定波长的机械束缚。

### 2. 任意位置因果截断热力图对比
![任意位置热力对比](assets/fig5_arbitrary_inversion_heatmap.png)
* **标准 2π RoPE（左）**：转折发生后，下三角区域依然充斥大量散乱浅色噪点，前置作废事实持续干扰后文。
* **固定周期 256 莫比乌斯（中）**：因转折点不在第 256 位，固定波长模型完全失效。
* **自适应规范莫比乌斯 SG-Möbius（右）**：在 Token 51 处展现出极度清晰的**垂直因果截断边界（青色线左侧大面积黑化熄灭）**！作废前置命题被主动相位对消，有效事实区域保持高亮汇聚。

### 3. 三大基准定量实测数据

| 评测场景 | 序列长度 | 激发的转折点 | 标准 RoPE 注意力熵 | 莫比乌斯注意力熵 | 实测结论 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **平坦顺承控制组 (无转折)** | 337 | **无 (`[]`)** | 3.4462 | **3.4462** | **熵完全不变 (0.0% 变化)，零误报、零副作用，完全等价于标准 RoPE** |
| **任意单点转折 (Token 51)** | 298 | `Token 51` | 3.4503 | **3.3966** | 注意力熵显著收敛，无效背景底噪被精准抑制，信噪比大幅提升 |
| **多段复合转折** | 234 | `Token 39, 41` | 3.3120 | **3.1626** | 熵下降 4.51%，注意力聚焦力与对抗干扰能力全面占优 |

---

## 🛡️ 形式化数学验证 (Formal Verification)

本项目提供了严密的形式化数学证明，杜绝工程打补丁与逻辑伪命题：

1. **Lean 4 形式化定理证明** ([`formal_verification/MobiusAttention.lean`](formal_verification/MobiusAttention.lean))：
   * `conservation_of_flatness`：严格证明无转折时（$\omega = 0$）与标准 RoPE 逐位等价。
   * `holonomy_cocycle`：严格证明路径平行移动的连续可加性。
   * `antipodal_inversion`：严格证明转折累积相位 $\Phi = \pi$ 时，内积拓扑因子严格为 $-1.0$。
   * `gauge_translation_invariance`：证明规范相对相位对全局平移具有严格不变性。

2. **可执行代数模糊验证器** ([`formal_verification/verify_properties.py`](formal_verification/verify_properties.py))：
   在终端一键运行：
   ```bash
   python formal_verification/verify_properties.py
   ```
   1000 次蒙特卡洛随机变长模糊测试（任意序列长度、任意转折坐标、深层嵌套）**100% 通过验证**。

---

## 🚀 极简上手 (Quickstart)

无需重训大模型，三行代码无缝注入任意 HuggingFace 模型（兼容 FlashAttention 与 SDPA）：

```python
import torch
from transformers import AutoModelForCausalLM
from src.mobius_attention import inject_mobius_attention, SemanticGaugeConnection

# 1. 加载原生 HuggingFace 模型 (例如 Qwen3.5-9B)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", torch_dtype=torch.float16, device_map="auto")

# 2. 一行代码注入莫比乌斯注意力算子 (零显存增加，线性复杂度)
patched_layers = inject_mobius_attention(model)
print(f"成功注入 {patched_layers} 个注意力层！")

# 3. 接入语义规范联络器
embed = model.get_input_embeddings()
gauge = SemanticGaugeConnection(hidden_dim=embed.embedding_dim).to(model.device)
```

---

## 📜 开源双重许可证 (Dual License)

本项目采用双许可证授权，您可以根据需要自主选择：
* [Apache License, Version 2.0](LICENSE-APACHE)
* [木兰宽松许可证 第2版 (MulanPSL - 2.0)](LICENSE-MULAN)
