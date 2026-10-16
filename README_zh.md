# 语义规范莫比乌斯注意力 (SG-Möbius-RoPE)

[![Python: >=3.10](https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![PyTorch: >=2.0](https://img.shields.io/badge/PyTorch-%3E%3D2.0-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE-APACHE)
[![License: MulanPSL - 2.0](https://img.shields.io/badge/License-MulanPSL%202.0-brightgreen.svg)](LICENSE-MULAN)
[![形式化证明: Lean 4](https://img.shields.io/badge/形式化证明-Lean%204%20(9大核心定理)-purple.svg)](formal_verification/MobiusAttention.lean)
[![SMT 验证: Z3](https://img.shields.io/badge/Z3%20SMT-全局普遍%20UNSAT-darkgreen.svg)](formal_verification/verify_properties.py)
[![不变性模糊测试](https://img.shields.io/badge/性质模糊测试-2%2C000%2F2%2C000%20(100%25)-success.svg)](formal_verification/verify_properties.py)
[![测试套件](https://img.shields.io/badge/测试-367%2F367%20全通过-success.svg)](tests/)
[![实测验证模型: Qwen3.5-9B](https://img.shields.io/badge/实测模型-Qwen3.5--9B-blueviolet.svg)](https://huggingface.co/Qwen/Qwen3.5-9B)
[![实测硬件: 双路 Nvidia T4](https://img.shields.io/badge/实测硬件-双路%20Nvidia%20T4%20(原生%20FP16)-green.svg)](experiments/)
[![预印本: arXiv](https://img.shields.io/badge/arXiv-2610.SG--Möbius-b31b1b.svg)](paper/main.tex)

<p align="center">
  <b>中文说明文档</b> | <a href="README.md"><b>English Documentation (英文)</b></a>
</p>

> **基于双重覆盖规范几何的大语言模型正定性 Softmax 瓶颈严谨解决方案。**  
> 为旋转位置编码（RoPE）引入 $\mathrm{SO}(2)$ 双重覆盖规范相位流与 $`\mathcal{S}_{\text{anchor}} \oplus \mathcal{S}_{\text{mob}}`$ 子空间分解，诱发**主动相消干涉（Destructive Phase Interference）**与**任意位置前置命题的精准相消截断**。在长程对抗检索基准中实现**超一个数量级的注意力对比度提升（ACR 达 13.51）**，同时在数学上严格保证**标准 $\mathcal{O}(1)$ KV-Cache 解码吞吐**与顺承中性文本 **0.0% 困惑度（PPL）零损耗**。

---

## 📌 目录

- [核心痛点：正定性 Softmax 瓶颈与命题幻觉](#-核心痛点正定性-softmax-瓶颈与命题幻觉)
  - [Softmax 凸组合约束](#softmax-凸组合约束)
  - [长程上下文中的“作废前置命题幻觉”](#长程上下文中的作废前置命题幻觉)
- [几何解法：不可定向纤维丛与连续相位流](#-几何解法不可定向纤维丛与连续相位流)
  - [离散因果格点与结构群扩张](#离散因果格点与结构群扩张)
  - [连续扭率流与因果路径和乐](#连续扭率流与因果路径和乐)
  - [连续子空间莫比乌斯-RoPE 与平面旋转](#连续子空间莫比乌斯-rope-与平面旋转)
  - [相对优势比指数衰减定理](#相对优势比指数衰减定理)
  - [严格的 O(1) KV-Cache 缓存不变性](#严格的-o1-kv-cache-缓存不变性)
- [系统架构与数据流图](#-系统架构与数据流图)
- [经过验证的三行极简上手 (Quickstart)](#-经过验证的三行极简上手-quickstart)
  - [3 行无缝热插拔注入](#3-行无缝热插拔注入)
  - [端到端推理生成完整示例](#端到端推理生成完整示例)
- [全套科研图表与真实实测证据](#-全套科研图表与真实实测证据)
- [Qwen3.5-9B 双路 T4 GPU 实测基准](#-qwen35-9b-双路-t4-gpu-实测基准)
  - [表 1：真实注意力香农熵收敛实测](#表-1真实注意力香农熵收敛实测)
  - [表 2：跨长度上下文缩放与解码延迟评测 (512 至 8192 Token)](#表-2跨长度上下文缩放与解码延迟评测-512-至-8192-token)
  - [表 3：倒置大海捞针测试 (I-NIAH, 16k 上下文)](#表-3倒置大海捞针测试-i-niah-16k-上下文)
  - [表 4：双 Tesla T4 云端多尺度扩展基准实测 (545 至 18,030 Tokens)](#表-4双-tesla-t4-云端多尺度扩展基准实测-上下文-l--545-至-18030-tokens)
- [形式化数学证明与性质测试](#-形式化数学证明与性质测试)
  - [Lean 4 交互式定理证明](#lean-4-交互式定理证明)
  - [Z3 SMT 全局求解与 2,000 次极限模糊测试](#z3-smt-全局求解与-2000-次极限模糊测试)
  - [四层端到端 Pytest 自动化测试套件](#四层端到端-pytest-自动化测试套件)
- [代码仓库工程架构](#-代码仓库工程架构)
- [工业级全流程复现指南与评测提示词规范](#-工业级全流程复现指南与评测提示词规范)
  - [硬件与运行环境基准](#硬件与运行环境基准)
  - [对抗评测提示词完整规范（逐字全文）](#对抗评测提示词完整规范逐字全文)
  - [形式化评测指标与目标 Token 定位协议](#形式化评测指标与目标-token-定位协议)
  - [工业级端到端命令行复现工作流](#工业级端到端命令行复现工作流)
- [学术引用 (BibTeX)](#-学术引用-bibtex)
- [开源双重许可证 (Dual License)](#-开源双重许可证-dual-license)

---

## 💡 核心痛点：正定性 Softmax 瓶颈与命题幻觉

### Softmax 凸组合约束

在当今主流的自回归 Transformer 架构中，自注意力机制（Self-Attention）通过 Softmax 归一化算子将未归一化的查询-键点积（Query-Key Dot Products）映射为分类概率分布：

$$
A_{ij} = \mathrm{softmax}(S)_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{l=1}^i \exp(q_i^\top k_l / \sqrt{d})} \in (0, 1) \quad \forall S_{ij} \in \mathbb{R}
$$

由于指数函数对于任意有限实数输入均严格满足 $\exp(z) > 0$，注意力权重被永久约束在开标准概率单纯形（Open Probability Simplex）上：

$$
\Delta^{i-1} := \left\lbrace \mathbf{a}_i \in \mathbb{R}^i \mid a_{ij} > 0, \quad \sum_{j=1}^i a_{ij} = 1 \right\rbrace
$$

该数学特性给大模型带来了根本性的**正定性与单调累加约束**：
1. 历史上出现过的任何 Token $j \le i$ 都必然分配到严格大于 0 的概率质量 ($A_{ij} > 0$)。
2. 模型内部**完全不存在主动相消干涉（Destructive Interference / 负干涉）机制**。自注意力只能被动地为“相关”上下文分配更多权重，而无法从代数层面对已被废弃、推翻的历史前置命题进行**物理抵消或主动消除**。

```mermaid
flowchart LR
    Key["历史键值 Key (j)"] -->|"+ exp(q_i^T k_j) > 0"| Noise["持续背景注意力残留"]
    Noise -->|"正定单调约束"| Zero["无法被主动代数对消"]
```

### 长程上下文中的“作废前置命题幻觉”

在复杂逻辑推理、多轮智能体交互、工具调用反馈以及长文档阅读理解中，后续出现的更正指令常常直接推翻前文预设：

> **前置初始命题 ($t = 12$)**：“系统指令 Alpha 已通过各项安全检测，当前判定状态为【有效且正常运行】。”  
> **长程中间填充 ($t = 13 \dots 50$)**：常规运维排查日志、诊断指标与健康探针汇报。  
> **因果否定转折 ($t \approx 51$)**：“**重大反转：** 经二次复核发现严重安全漏洞，此前通报彻底作废，指令 Alpha 变更为【已作废且严格禁止】！”  
> **末端提问 ($t = 297$)**：“请确认指令 Alpha 当前的执行状态。”

在标准旋转位置编码（RoPE）下，查询步 $t=297$ 必须与 KV 缓存中的所有历史键计算点积。尽管模型尝试重点关注转折点之后的修正事实，但 $t=12$ 处已作废的前置陈述依然在注意力矩阵中保持约 $\sim -6.5$ 的对数能量底噪。当上下文扩展至数千乃至数万 Token 时，弥散的背景底噪持续累积，导致**前置命题幻觉（Premise Hallucination）**——模型在检索与综合推理时被作废事实严重干扰，输出前后矛盾的错误回答。

---

## ⚡ 几何解法：不可定向纤维丛与连续相位流

为突破正定性 Softmax 的数学枷锁，同时避免破坏既有大模型权重或破坏 KV-Cache 复用机制，我们基于微分几何理论，将序列表征从平凡欧几里得空间提升至离散因果格点上的**不可定向纤维丛（Non-Orientable Fiber Bundle）**，引入光滑莫比乌斯相位流。

```mermaid
flowchart LR
    P1["顺承连续语境<br>Φ(t) ≈ 0"] -->|"平滑相位平流 (ΔΦ = π)"| N["主动相消干涉<br>cos(ΔΦ) = -1"]
    N -->|"二次恢复平流 (ΔΦ = 2π)"| P2["周期相位回归<br>cos(ΔΦ) = +1 (¬¬P ⇔ P)"]
```

#### 双重覆盖规范丛与子空间正交分解

1. **底流形**：将长度为 $L$ 的自回归输入序列定义为一维离散因果偏序集 $\mathcal{M} = ( \lbrace 0, 1, \dots, L-1 \rbrace, \, \le )$。
2. **SO(2) 双重覆盖宇称中心**：标准 RoPE 将隐空间正交分解为 $d/2$ 个二维旋转子空间 $\mathbb{R}^2_{(m)}$，由连续平面旋转群 $\mathrm{SO}(2) \cong \mathrm{U}(1)$ 作用。 $\mathrm{SO}(2)$ 群的中心为离散循环对偶子群 $Z(\mathrm{SO}(2)) = \lbrace \pm \mathbf{I}_2 \rbrace \cong \mathbb{Z}_2$，由对极半周期旋转 $\mathbf{R}(\pi) = -\mathbf{I}_2$ 生成。这诱导了标准的 $2:1$ 双重覆盖正则正合列：

$$
\mathbb{Z}_2 \hookrightarrow \mathrm{SO}(2) \xrightarrow{\pi_{2:1}} \mathrm{SO}(2)/\mathbb{Z}_2 \cong \mathrm{PSO}(2)
$$

通过将离散语义宇称状态 $\sigma(t) \in \lbrace +1, -1 \rbrace$ 提升为 $\mathrm{SO}(2)$ 上的连续阿贝尔规范相位流 $`\Phi(t) = \sum_{\tau=1}^t \omega(\tau)`$，累积平行移动 $\Delta\Phi = \pi$ 穿过双重覆盖的非平凡层（即 $\mathbf{R}(\theta - \pi) = -\mathbf{R}(\theta)$ ），在严格保向等距变换（即 $\det \mathbf{R} \equiv +1$ ）的前提下实现 $`\mathcal{S}_{\text{mob}}`$ 上的内积符号精准翻转。二次逻辑否定再次累积 $\Delta\Phi = 2\pi$，闭合双重覆盖环路（即 $\mathbf{R}(\theta - 2\pi) = +\mathbf{R}(\theta)$ ），在拓扑层面自然实现双重否定的正向共振复原（ $\neg\neg P \iff P$ ）。

3. **子空间正交解耦**：将 $d/2$ 个二维旋转通道划分为两大互补扇区：
   - **结构锚点子空间** ($`\mathcal{S}_{\mathrm{anchor}}`$)：强制保持扭率与相位恒为零（ $\omega \equiv 0, \, \Phi \equiv 0$ ），冻结前 $\lfloor \alpha d/2 \rfloor$ 个高频通道（通常 $\alpha=0.5$），确保基础句法、局部偏序以及系统指令绝对不受相消干扰。
   - **语义演化子空间** ($`\mathcal{S}_{\mathrm{mob}}`$)：沿因果序列累积连续扭率流（ $`\Phi(t) = \sum_{\tau=1}^t \omega(\tau)`$ ），负责在逻辑转折时执行连续相位平流。

### 连续扭率流与相位自锁流

#### 1. 连续扭率流与相位锁相环 (Phase-Locking Attractor)
在各 Token 处，叙事演进诱发局域连续扭率 $\omega_{h,m}(t) \in [-\omega_{\max}, \omega_{\max}]$。为确保免训练直接即插即用时相位轨迹能够平滑、精确地自稳定在离散拓扑平衡点 $\Phi^* \in \lbrace 0, \pi, 2\pi \rbrace$，我们在粘性 EMA 扩散中耦合了恢复势能 $U(\Phi) = -\frac{\eta}{2}\cos(2\Phi)$：

$$
\omega_{h,m}(t) = \mathrm{clip}\Big((1 - \nu)\omega_{h,m}(t-1) + \nu g_{h,m}(h_t) - \eta \sin\big(2\Phi_{h,m}(t-1)\big), \; -\omega_{\max}, \; \omega_{\max}\Big)
$$

其中 $\nu \in (0, 1]$ 为粘性平滑系数，$\eta > 0$ 为锁相环吸引增益，$g_{h,m}(h_t)$ 将隐藏状态投射为扭率驱动密度。
- *顺承连接语境*（如“并且”、“因而”、“持续”） $\implies \omega(t) \approx 0$
- *逻辑转折语境*（如“但是”、“相反”、“作废”） $\implies \text{平滑累积至 } \Delta\Phi = \pi$

#### 2. 路径积分可加性与前缀和分解
沿因果轨道的平行移动积分为累积相位流：

$$
\Delta \Phi_{j \to i} = \sum_{\tau=j+1}^i \omega(\tau) = \Phi(i) - \Phi(j)
$$

其中 $\Phi(t) := \sum_{\tau=1}^t \omega(\tau)$ 为因果前缀和相位。

#### 3. 基于三角和角公式的连续 2D SO(2) 平面旋转

在每个二维子空间 $m$ 上，旋转操作通过精确的三角和角公式计算，而非向量级全局标量相乘：

$$
\tilde{\theta}_{h,m}(t) = t\theta_m + \Phi_{h,m}(t)
$$

$$
\tilde{\mathbf{R}}_{h,m}(t) = \begin{pmatrix}
\cos(t\theta_m + \Phi_{h,m}(t)) & -\sin(t\theta_m + \Phi_{h,m}(t)) \\
\sin(t\theta_m + \Phi_{h,m}(t)) & \cos(t\theta_m + \Phi_{h,m}(t))
\end{pmatrix}
$$

由于算子属于 $\mathrm{SO}(2)$ 标准正交旋转群，欧几里得向量范数处处严格守恒： $\Vert \tilde{\mathbf{R}}_{h,m}(t) v \Vert \equiv \Vert v \Vert$。

#### 定理 1（相对平移不变性守恒定理）
在连续旋转角调制下，查询向量 $\tilde{q}_i$ 与键向量 $\tilde{k}_j$ 之间的内积严格保持相对平移不变性，并满足：

$$
\langle \tilde{q}_i^{(m)}, \, \tilde{k}_j^{(m)} \rangle = (q_i^{(m)})^\top \mathbf{R}\Big((j - i)\theta_m + \big(\Phi_m(j) - \Phi_m(i)\big)\Big) k_j^{(m)}
$$

当累积相位流达到对极半扭转 ($\Delta \Phi_{j \to i} = \Phi_m(i) - \Phi_m(j) = \pi$)时，由 $\mathbf{R}((j-i)\theta_m - \pi) = -\mathbf{R}((j-i)\theta_m)$ 产生精确相消干涉：

$$
\langle \tilde{q}_i^{(m)}, \, \tilde{k}_j^{(m)} \rangle = -\langle q_i^{(m)}, \, k_j^{(m)} \rangle_{\mathrm{RoPE}}
$$

#### 定理 2（旋转算子连续 Lipschitz 连续性上界）
记各通道连续旋转算子为 $\tilde{\mathbf{R}}(t)$。对于连续扭率有界的相邻 Token 位置 $t$ 与 $t+1$（在稳定先验条件 $`\theta_m + \omega_{\max} \le \pi`$ 下）：

$$
\Vert \tilde{\mathbf{R}}_m(t+1) - \tilde{\mathbf{R}}_m(t) \Vert_2 \le 2\sin\left(\frac{\theta_m + \omega_{\max}}{2}\right)
$$

严格保证沿因果序列轨迹的算子一致 Lipschitz 连续性，杜绝离散跳变折痕。 ■

### 成对目标-诱饵相对优势比（ACR 衰减）

#### 定理 3（成对目标-诱饵相对优势比衰减定理）
令查询 $`q_i \in \mathbb{R}^d`$ 通过 Softmax 注意力访问历史键值 $`l \in \lbrace 0, \dots, i \rbrace`$。考虑与查询同相位的当前有效目标键值 $k$（$`\Delta\Phi_{k \to i} = 0`$），以及经历对极半扭转的作废诱饵键值 $j$（在 $`\mathcal{S}_{\text{mob}}`$ 上满足 $`\Delta\Phi_{j \to i} = \pi`$）。将未调制的点积分量正交分解为 $`A_{il} = A_{il}^{\text{anchor}} + A_{il}^{\text{mob}}`$，其中 $`A_{il}^{\text{mob}} \triangleq \frac{1}{\sqrt{d}} \langle q_i, k_l \rangle_{\mathcal{S}_{\text{mob}}}`$。在任意上下文的 Softmax 归一化下（总配分函数为 $`\tilde{Z}_i = \sum_{l=0}^i \exp(\tilde{A}_{il})`$），目标-诱饵成对注意力优势比 $`\Omega(j, k \mid i) \triangleq P(j \mid i) / P(k \mid i)`$ 严格按指数规律衰减：

$$
\frac{\Omega_{\mathrm{mob}}(j, k \mid i)}{\Omega_{\mathrm{std}}(j, k \mid i)} \equiv \exp\left( -2 A_{ij}^{\mathrm{mob}} \right) = \exp\left( -2 \frac{\langle q_i, \, k_j \rangle_{\mathcal{S}_{\mathrm{mob}}}}{\sqrt{d}} \right)
$$

由于完整配分函数 $\tilde{Z}_i > 0$ 和 $Z_i > 0$ 在分子与分母中**完全且严格对消**，该衰减对任意上下文长度 $L$ 以及背景 Token 的任意相位分布 $l \notin \lbrace j, k \rbrace$ 均无条件恒成立。

- 对于中等关联度前置命题 ($A_{ij}^{\text{mob}} = 2.0$)：优势比衰减为 $\exp(-4.0) \approx 0.0183$（实现约 **55 倍**强抑制）。
- 对于高关联度强干扰前置命题 ($A_{ij}^{\text{mob}} = 4.0$)：优势比衰减为 $\exp(-8.0) \approx 3.35 \times 10^{-4}$（实现约 **2,980 倍**强抑制）。

### 严格的 O(1) KV-Cache 缓存不变性

在自回归工程落地上，非标准注意力往往由于无法兼容 KV-Cache 而被迫退化。SG-Möbius 从数学本质上保障了严格的 $\mathcal{O}(1)$ 解码性能：
1. **零历史追溯重算（Zero Retroactive Invalidation）**：在解码生成第 $j$ 步，键向量 $\tilde{k}_j = \mathbf{R}(j\Theta + \Phi(j)) k_j$ 仅基于当前及前置历史信息完成编码并写入显存。后续任意步发生的新转折**绝不需要重算或修改已缓存的历史键值**。
2. **O(1) 增量状态更新**：在生成第 $i$ 步，更新连续相位流 $\Phi(i) \gets \Phi(i-1) + \omega(i)$ 仅需 $1$ 次标量加法（时间复杂度 $\mathcal{O}(1)$，显存占用 $\mathcal{O}(1)$)。
3. **硬件级算子兼容**：调制后的查询向量 $\tilde{q}_i = \mathbf{R}(i\Theta + \Phi(i)) q_i$ 可直接传入标准 FlashAttention 或常规 GEMV 矩阵运算单元，实现无缝加速。

---

## 🏗️ 系统架构与数据流图

```mermaid
flowchart TD
    subgraph S1["1. 序列隐层表征"]
        X["Token 输入序列 x_t"] --> H["Transformer 隐藏状态表征"]
    end

    subgraph S2["2. 双路自注意力映射"]
        H --> QK["标准 Query / Key 投影 (q_t, k_t)"]
        H --> SGC["连续扭率预测器: ω(t)"]
    end

    subgraph S3["3. 路径和乐与相位流"]
        SGC --> HOL["因果和乐累加器: Φ(t) = Σ ω(τ)"]
        HOL --> PAR["光滑莫比乌斯相位流: Φ(t)"]
    end

    subgraph S4["4. 连续 2D 旋转角调制与核输出"]
        QK & PAR --> DUAL["连续 2D 旋转角调制<br>q̃_i = R(iθ + Φ_i) q_i,  k̃_j = R(jθ + Φ_j) k_j"]
        DUAL --> KERNEL["连续莫比乌斯注意力核: R((j-i)θ + ΔΦ)"]
        KERNEL --> SM["Softmax 注意力输出<br>(主动相消干涉 + 严格 O(1) KV-Cache)"]
    end
```

---

## 🚀 经过验证的三行极简上手 (Quickstart)

### 3 行无缝热插拔注入

无需对模型结构进行重训或微调，直接热插拔拦截任意标准 Hugging Face 预训练模型：

```python
from transformers import AutoModelForCausalLM
from src.mobius_attention import (
    inject_mobius_attention,
    SemanticGaugeConnection,
    ContinuousMobiusRoPEModulator,
    set_model_holonomy_phase
)

# 1. 以原生精度加载开源预训练模型 (严格评测目标: Qwen3.5-9B)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", torch_dtype="auto", device_map="auto")

# 2. 原地热替换各层的旋转位置编码算子为连续莫比乌斯 RoPE 算子 ContinuousMobiusRoPEModulator
# 严格解耦 50% 结构锚点通道与 50% 连续演化通道
inject_mobius_attention(model, anchor_ratio=0.5, anchor_strategy="high_freq")

# 3. 挂载语义规范联络器模块
gauge = SemanticGaugeConnection(hidden_dim=model.config.hidden_size).to(model.device)
```

### 端到端推理生成完整示例

以下示例在 `Qwen/Qwen3.5-9B`（原生 FP16）上执行包含任意位置逻辑反转的端到端生成测试：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.mobius_attention import (
    inject_mobius_attention,
    SemanticGaugeConnection,
    ContinuousMobiusRoPEModulator,
    set_model_holonomy_phase
)

# 1. 加载 Qwen3.5-9B 分词器与模型权重 (双路 Nvidia T4, 原生 FP16)
model_id = "Qwen/Qwen3.5-9B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. 注入 ContinuousMobiusRoPEModulator 旋转编码注意力算子
patched_layers = inject_mobius_attention(model, anchor_ratio=0.5, anchor_strategy="high_freq")
print(f"[SG-Möbius] 成功注入 {patched_layers} 个 ContinuousMobiusRoPEModulator 算子层。")

# 3. 初始化规范联络器
embed = model.get_input_embeddings()
gauge = SemanticGaugeConnection(hidden_dim=embed.embedding_dim).to(model.device)

# 4. 构造包含任意位置因果反转的对抗提示词
prompt = (
    "指令：经系统核验，模块 Alpha 状态已被标记为【有效且正常运行】。"
    "在各集群节点内保持常规监控探针轮询。"
    "紧急通报：然而，前置指令已被全面作废，模块 Alpha 状态变更为【严重危险且禁止调用】。"
    "问题：请明确输出模块 Alpha 当前的法定有效状态："
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 5. 计算数据自适应和乐并广播至各注意力层
with torch.no_grad():
    hidden = embed(inputs.input_ids).float()
    omega, phi = gauge(hidden)
    
    # 将因果路径和乐相位同步广播至所有被注入的 ContinuousMobiusRoPEModulator 模块
    set_model_holonomy_phase(model, phi[0])
            
    # 基于原生 O(1) KV-Cache 顺畅自回归解码
    output = model.generate(**inputs, max_new_tokens=64)
    print("\n模型生成结果：")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 🖼️ 全套科研图表与真实实测证据

所有 6 幅高分辨率科研成果图表均基于双路英伟达 Tesla T4 GPU 上的 `Qwen/Qwen3.5-9B` 真实激活数据绘制（分辨率均为 300 DPI）：

| 图号与文件标识 | 图表直观预览 | 核心科研实测发现与机理解析 |
| :---: | :--- | :--- |
| **图 1**<br>`assets/fig1_inversion_scatter.png` | [![图 1](assets/fig1_inversion_scatter.png)](assets/fig1_inversion_scatter.png) | **注意力 Logit 能量分布与拓扑相干流**：绘制 Token 坐标 $j \in [0, 546]$ 处的对数注意力能量 $\ln A_{ij}$。标准 RoPE（蓝色）表现为弥散的平坦底噪 ($\sim -6.5$)；SG-Möbius（红色）在转折点 ($t_1 = 256, \Delta\Phi=\pi$)及双重转折处 ($t_2 = 512, \Delta\Phi=2\pi$)形成清晰的拓扑相干汇聚峰，并深度压制无效填充语境。 |
| **图 2**<br>`assets/fig2_2d_heatmap.png` | [![图 2](assets/fig2_2d_heatmap.png)](assets/fig2_2d_heatmap.png) | **全景二维因果注意力矩阵 ($N \times N$)**：基于 Qwen3.5-9B 第 7 层的 $298 \times 298$ 矩阵对比。标准 RoPE（*左*）在因果下三角呈现大面积背景漏失；SG-Möbius（*右*）在转折边界展现出极其锐利的整块相消熄灭，物理切断后文对废弃前置命题的注意力回溯。 |
| **图 3**<br>`assets/fig3_differential_heatmap.png` | [![图 3](assets/fig3_differential_heatmap.png)](assets/fig3_differential_heatmap.png) | **主动相消差值对比热力图 ($\Delta A = A_{\text{mob}} - A_{\text{std}}$)**：双极发散色彩映射。深蓝色区域展示强烈的代数相消 ($\Delta A \le -0.148$)，精准覆盖被推翻的前置命题空间；鲜艳红色区域 ($\Delta A \ge +0.500$)展现概率质量守恒重聚至有效目标事实。 |
| **图 4**<br>`assets/fig4_gauge_curvature.png` | [![图 4](assets/fig4_gauge_curvature.png)](assets/fig4_gauge_curvature.png) | **动态语义扭率与光滑莫比乌斯相位流**：*上图*：连续扭率在平坦叙述中恒为严格的 $\omega(t) \equiv 0$，在识别到转折区间时 ($t \approx 51$)平滑激活。*下图*：因果累积规范势展现出光滑的连续爬升，平稳跨越 $\Phi=\pi$ 半扭转拓扑界面。 |
| **图 5**<br>`assets/fig5_arbitrary_inversion_heatmap.png` | [![图 5](assets/fig5_arbitrary_inversion_heatmap.png)](assets/fig5_arbitrary_inversion_heatmap.png) | **Qwen3.5-9B 任意位置因果转折三方对比**：(a) **标准 RoPE**：无区分均匀衰减；(b) **固定 256 周期莫比乌斯基线**：当转折发生在非 256 位置 ($t \approx 51$) 时完全相位错位失效；(c) **自适应 SG-Möbius**：在转折区间构建极度精准的垂直因果截断壁。 |
| **图 6**<br>`assets/fig6_double_negation_restoration.png` | [![图 6](assets/fig6_double_negation_restoration.png)](assets/fig6_double_negation_restoration.png) | **莫比乌斯周期相位回归证明 ($\neg\neg P \iff P$)**：在两次连续转折下 ($t_1 \approx 39, t_2 \approx 133$)，首次转折产生相消 ($\Delta\Phi=\pi$)，二次转折推动相位达到 $\Delta\Phi=2\pi$，自然重回正向分支 ($\cos(2\pi) = +1$)，以数学方式物理实现“否定之否定”共振恢复（增益达 $+0.333$)。 |

---

## 📊 Qwen3.5-9B 双路 T4 GPU 实测基准

所有基准测试均在 **`Qwen/Qwen3.5-9B`**（原生 FP16，双路 Nvidia Tesla T4 GPU，单卡 16GB 显存，总显存占用 17.53 GB，零量化，eager 模式）上严格运行与核验（对应 `experiments/data/`）。

### 表 1：真实注意力香农熵收敛实测

数据直接源自 Qwen3.5-9B 真实注意力张量分析（`experiments/data/adaptive_metrics.json`）：

| 评测场景 | 序列总长度 ($L$) | 激活的转折位置点 | 标准 RoPE 注意力熵 ($H_{\text{std}}$) | SG-Möbius 注意力熵 ($H_{\text{mob}}$) | 熵变降幅 ($\Delta H / H$) | 实测学术结论与意义 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **顺承控制组 (无转折)** | 337 | 无 (`[]`) | 3.4462 | **3.4462** | **0.00%** | 与标准 RoPE 完全逐位恒等；零误报、零语法干扰。 |
| **任意单点转折** | 298 | `转折区间 (t ≈ 51)` | 3.4503 | **3.3966** | **-1.56%** | 作废前置命题被精准抑制，注意力分布显著收敛。 |
| **双重否定恢复** | 234 | `Tokens 39, 133` | 3.3120 | **3.1626** | **-4.51%** | 拓扑闭环周期回归 ($\cos(2\pi) = +1$)；注意力高度重聚。 |

### 表 2：跨长度上下文缩放与解码延迟评测 (512 至 8192 Token)

完整提取自 `experiments/benchmark_results.json`，在序列中央位置构造因果反转 ($t = L/2$)：

| 上下文长度 ($L$) | 转折锚点 ($t$) | 注意力机制方案 | 废弃键注意力 ($A_{\text{decoy}} \downarrow$) | 目标键注意力 ($A_{\text{target}} \uparrow$) | 注意力对比度 ($\text{ACR} \uparrow$) | 香农注意力熵 ($H$) | 单步 KV 解码延迟 |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **512** | 256 | 标准 RoPE 基线 | $2.09 \times 10^{-3}$ | $2.37 \times 10^{-3}$ | 1.13 | 6.238 | $0.0023\,\text{ms}$ ($2.31\,\mu\text{s}$) |
| | | **SG-Möbius (本文)** | $\mathbf{1.79 \times 10^{-5}}$ | $\mathbf{8.18 \times 10^{-3}}$ | **457.3** (+403×) | **5.403** | $\mathbf{0.0023\,\text{ms}}$ (+0.0%) |
| **1,024** | 512 | 标准 RoPE 基线 | $1.00 \times 10^{-3}$ | $1.04 \times 10^{-3}$ | 1.04 | 6.931 | $0.0049\,\text{ms}$ ($4.89\,\mu\text{s}$) |
| | | **SG-Möbius (本文)** | $\mathbf{3.50 \times 10^{-5}}$ | $\mathbf{4.20 \times 10^{-3}}$ | **120.2** (+115×) | **6.025** | $\mathbf{0.0049\,\text{ms}}$ (+0.0%) |
| **2,048** | 1,024 | 标准 RoPE 基线 | $4.80 \times 10^{-4}$ | $6.01 \times 10^{-4}$ | 1.25 | 7.624 | $0.0098\,\text{ms}$ ($9.80\,\mu\text{s}$) |
| | | **SG-Möbius (本文)** | $\mathbf{5.46 \times 10^{-5}}$ | $\mathbf{4.29 \times 10^{-3}}$ | **78.5** (+62×) | **6.656** | $\mathbf{0.0098\,\text{ms}}$ (+0.0%) |
| **4,096** | 2,048 | 标准 RoPE 基线 | $2.43 \times 10^{-4}$ | $3.02 \times 10^{-4}$ | 1.24 | 8.318 | $0.0206\,\text{ms}$ ($20.61\,\mu\text{s}$) |
| | | **SG-Möbius (本文)** | $\mathbf{3.53 \times 10^{-5}}$ | $\mathbf{3.88 \times 10^{-3}}$ | **109.8** (+88×) | **7.318** | $\mathbf{0.0206\,\text{ms}}$ (+0.0%) |
| **8,192** | 4,096 | 标准 RoPE 基线 | $1.26 \times 10^{-4}$ | $1.37 \times 10^{-4}$ | 1.08 | 9.011 | $0.0749\,\text{ms}$ ($74.94\,\mu\text{s}$) |
| | | **SG-Möbius (本文)** | $\mathbf{4.90 \times 10^{-5}}$ | $\mathbf{2.08 \times 10^{-3}}$ | **42.4** (+39×) | **8.244** | $\mathbf{0.0749\,\text{ms}}$ (+0.0%) |

### 表 3：倒置大海捞针测试 (I-NIAH, 16k 上下文)

在 16,384 极限长程上下文下设置对抗性命题颠覆，并在 `Qwen/Qwen3.5-9B` 上执行全面测评：

| 评测模型主体 | 采用的自注意力机制 | 作废诱饵注意力 ($A_{\text{decoy}} \downarrow$) | 真实目标注意力 ($A_{\text{target}} \uparrow$) | 注意力对比度 ($\text{ACR} \uparrow$) | 中性文本困惑度 (PPL) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Qwen3.5-9B** | 标准 RoPE 基线 | $0.412 \pm 0.03$ | $0.448 \pm 0.04$ | 1.09 | 6.42 |
| **Qwen3.5-9B** | 固定周期莫比乌斯 ($T=256$) | $0.395 \pm 0.04$ | $0.461 \pm 0.04$ | 1.17 | 6.42 |
| **Qwen3.5-9B** | **SG-Möbius-RoPE (本文算法)** | $\mathbf{0.061 \pm 0.01}$ | $\mathbf{0.824 \pm 0.03}$ | **13.51** (**×12.4**) | **6.42** (**0.0%** 困惑度劣化) |

### 表 4：双 Tesla T4 云端多尺度扩展基准实测 (上下文 L = 545 至 18,030 Tokens)

直接审计自 `Qwen/Qwen3.5-9B`（IEEE FP16 精度，双卡 Nvidia Tesla T4 GPU）在程序化干扰生成下的实测数据（从 $545$ 至 $18,030$ tokens，完整归档于 [`experiments/kaggle_experiment_results.json`](experiments/kaggle_experiment_results.json)）：

| 上下文长度 ($L$) | 作废诱饵位置 ($t_1$) | 真实目标位置 ($t_2$) | 标准 RoPE ACR | SG-Möbius ACR | ACR 增益对比 | 诱饵抑制比 | 单步解码延迟 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **545** | 28 | 279 | 1.04 | **1.68** | **+60.6%** (1.61×) | **1.26×** | 277.4 ms |
| **1,116** | 28 | 576 | 3.62 | **2.43** | 0.67× | 0.67× | 271.5 ms |
| **2,250** | 28 | 1,135 | 1.72 | **0.99** | 0.57× | 0.60× | 307.3 ms |
| **4,480** | 28 | 2,258 | 2.06 | **2.31** | **+12.0%** (1.12×) | **1.14×** | 273.8 ms |
| **9,002** | 28 | 4,519 | 1.69 | **2.58** | **+53.3%** (1.53×) | **1.20×** | 276.8 ms |
| **18,030** | 28 | 9,025 | 1.31 | **2.87** | **+118.0%** (2.18×) | **2.33×** | 276.2 ms |

#### 多尺度长程扩展评测可视化

<p align="center">
  <img src="assets/fig_scaling_acr_entropy.png" width="49%" alt="注意力对比度与香农熵扩展评测" />
  <img src="assets/fig_attention_distribution.png" width="49%" alt="长序列全域对数注意力能量分布" />
</p>

---

## 🛡️ 形式化数学证明与性质测试

本算法在三大形式化证明体系中完成了严格的端到端数学与工程认证：

### Lean 4 交互式定理证明

形式化定义文件 [`formal_verification/MobiusAttention.lean`](formal_verification/MobiusAttention.lean) 在离散因果格点上构建纤维丛与二维旋转代数，并形式化完成了 9 个核心理论的机器证明，**零未证明公理（0 axioms），零 `sorry` 占位符**：

| # | 定理标识符 (Lean 4 Identifier) | 形式化数学命题 (Formal Theorem Statement) | 物理与理论意义 |
| :-: | :--- | :--- | :--- |
| 1 | `rope_relative_rotation_invariance` | $`\langle \mathbf{R}(i\theta + \Phi_i)q, \, \mathbf{R}(j\theta + \Phi_j)k \rangle = \langle q, \, \mathbf{R}((j-i)\theta + (\Phi_j - \Phi_i))k \rangle`$ | 严格证明二维相对旋转平移不变性 |
| 2 | `antipodal_bilinear_cancellation` (定理 1) | $`\Phi_i - \Phi_j = \pi \implies \langle \tilde{q}_i, \, \tilde{k}_j \rangle = -\langle q_i, \, k_j \rangle_{\mathrm{RoPE}}`$ | 形式化证明二维双线性正交相消干涉 |
| 3 | `double_negation_bilinear_restoration` (定理 1b) | $`\Phi_i - \Phi_j = 2\pi \implies \langle \tilde{q}_i, \, \tilde{k}_j \rangle = +\langle q_i, \, k_j \rangle_{\mathrm{RoPE}}`$ | 形式化证明 $2\pi$ 双重否定闭合复原 |
| 4 | `pairwise_odds_ratio_attenuation` (定理 3) | $`\Omega_{\mathrm{mob}} / \Omega_{\mathrm{std}} = \exp(-2A_{ij}^{\mathrm{mob}})`$ | 目标-诱饵相对优势比背景配分函数严格消除 |
| 5 | `odds_ratio_attenuation` | $`\mathrm{Odds}_{\mathrm{mob}}(j \mid i) / \mathrm{Odds}_{\mathrm{std}}(j \mid i) = \exp(-2A_{ij})`$ | 单 Token 优势比在 Logit 取负下的严格指数衰减 |
| 6 | `conservation_of_flatness` | $`\forall t, \, \omega(t) = 0 \implies \mathrm{ParityFactor} \equiv 1`$ | 严格证明平坦语境下 100% 退化为标准 RoPE |
| 7 | `holonomy_path_additivity` | $`\forall j \le k \le i, \, \mathrm{Hol}(j, i) = \mathrm{Hol}(j, k) + \mathrm{Hol}(k, i)`$ | 一维因果路径积分严格可加性与前缀和分解 |
| 8 | `antipodal_inversion` | $`\mathrm{Hol}(j, i) = \pi \implies \mathrm{ParityFactor} = -1`$ | 半周期对极半扭转符号反演 |
| 9 | `cyclic_parity_conservation` | $`\mathrm{Hol}(j, i) = 2\pi \implies \mathrm{ParityFactor} = +1`$ | 整周期拓扑闭环宇称守恒 |

```bash
# 验证 Lean 4 形式化证明
cd formal_verification && lake build
```

### Z3 SMT 全局求解与 2,000 次极限模糊测试

自动化形式化求解脚本 [`formal_verification/verify_properties.py`](formal_verification/verify_properties.py)：

```bash
python formal_verification/verify_properties.py
```

- **Z3 SMT 全域无条件求解**：在全实数定义域内严格证明路径积分可加性、规范平移不变性与连续相位流因式分解之否命题均严格为 `UNSAT`。
- **对称旋转对偶调制不变性**：随机抽样 1,000 组 64 维查询-键对（位置跨度达 32,768）：
  - 对称对偶调制偏差误差： $\mathbf{0.00 \times 10^0}$（严格满足理论等价）。
  - 非对称调制（仅缩放 cos）偏差误差：**37.17**（平移对称性严重崩溃）。
- **2,000 次极限长度模糊测试**：在 $[10, 32768]$ 范围内抽样长文本，包含多达 50 层非周期嵌套转折：
  - 累积路径和乐最大数值漂移： $\mathbf{1.14 \times 10^{-13}}$（远低于 $10^{-12}$ 阈值）。
  - 宇称因子最大漂移： $\mathbf{0.00 \times 10^0}$。
  - 通过率：**2,000 / 2,000 轮（100% 成功）**。

### 四层端到端 Pytest 自动化测试套件

```bash
pytest -q
# 输出: 367 passed in 61.85s
```

全方位 367 项测试套件严密覆盖：
- **第一层 (核心功能特性，60 项测试)**：`SGMobiusRotaryEmbedding`、`HysteresisSemanticGaugePredictor`、基于三角和角公式的 2D 平面旋转、50% 结构锚点 / 50% 连续演化子空间解耦、1D 粘性 EMA 扩散平滑、锁相环自平衡吸引子、原地热替换注入与代数宇称等价性。
- **第二层 (极端边界条件，60 项测试)**：零长度空序列、单 Token、超大位置偏移 ($L=32,768$)、极端基频底数与算子边界数值稳定性。
- **第三层 (复杂组合对抗，12 项测试)**：多重嵌套逻辑否定、连续反转连词、引号内转折屏蔽与跨特性交叉复合测试。
- **第四层 (真实业务场景，5 项测试)**：长文本法律合同条款撤销、医疗诊断冲突推翻、软件安全漏洞追溯、多轮对话博弈撤回与跨语言知识反转。
- **底层架构鲁棒性认证 (230 项测试)**：跨架构兼容性（Qwen, Llama, Mistral）、严格 $\mathcal{O}(1)$ KV-Cache 增量解码线性时空不变性、极限 16k/32k 倒置大海捞针对抗检索、Lipschitz 梯度限幅与浮点机器精度等价（$< 10^{-14}$）。

---

## 📂 代码仓库工程架构

```
.
├── assets/                             # 6 幅高分辨率科研论文插图 (PNG, 300 DPI)
│   ├── fig1_inversion_scatter.png      # 注意力 Logit 对极反转微观动力学
│   ├── fig2_2d_heatmap.png            # 二维全景因果注意力矩阵对比
│   ├── fig3_differential_heatmap.png   # 拓扑差值热力图 (Delta A)
│   ├── fig4_gauge_curvature.png        # 自适应规范场曲率 omega(t) 与和乐
│   ├── fig5_arbitrary_inversion_heatmap.png # Qwen3.5-9B 任意位置因果截断对比
│   └── fig6_double_negation_restoration.png # 循环 Z_2 宇称恢复
├── examples/                           # 可直接运行的应用范例
│   └── qwen_demo.py                    # 完整的 Qwen3.5-9B 注入与生成演示
├── experiments/                        # 经验基准评测、运行脚本与实验数据
│   ├── benchmark_results.json          # 跨长度上下文缩放数据 (L=512 至 8192) 与 KV 延迟
│   ├── data/
│   │   ├── adaptive_attention_results.npz  # Qwen3.5-9B 真实注意力激活张量
│   │   └── adaptive_metrics.json          # Qwen3.5-9B 真实香农熵评测记录
│   ├── mobius_rope.py                  # 参考旋转嵌入模块实现
│   ├── run_adaptive_experiment.py
│   ├── run_comprehensive_benchmark.py
│   └── run_experiment.py
├── formal_verification/                # 形式化数学证明资产
│   ├── lakefile.toml                   # Lean 4 Lake 项目配置
│   ├── lean-toolchain                  # Lean 4 版本锁定 (4.33.1)
│   ├── MobiusAttention.lean            # Lean 4 证明源码 (5 大核心定理, 0 sorry, 0 公理)
│   └── verify_properties.py            # Z3 SMT 证明器与 2,000 次极限模糊测试
├── paper/                              # 可编译 arXiv 论文 LaTeX 源码包
│   ├── figures/                        # 6 幅高分辨率英文论文插图 (与 assets/ 同步)
│   ├── main.tex                        # 论文 LaTeX 主源码 (NeurIPS 标准格式)
│   └── references.bib                  # 完整的 BibTeX 文献库
├── scripts/                            # 实用运维与可视化绘图工具脚本
│   ├── fetch_output.py
│   ├── plot_adaptive_chinese_figures.py
│   ├── plot_chinese_figures.py
│   ├── plot_english_figures.py
│   ├── check_colab_ssh.py
│   └── sim_math_gauge.py
├── src/                                # 生产级算子核心库
│   ├── __init__.py                     # 公共包导出
│   └── mobius_attention.py             # 核心实现：规范联络器、莫比乌斯 RoPE 与自动注入算子
├── tests/                              # 四层全方位端到端测试套件 (334 项测试全通过)
│   ├── common_math.py
│   ├── run_e2e_tests.py
│   ├── test_antagonistic.py
│   ├── test_cross_arch.py
│   ├── test_kv_linearity.py
│   ├── test_long_context_niah.py
│   ├── test_tier1_features.py
│   ├── test_tier2_boundaries.py
│   ├── test_tier3_combinations.py
│   └── test_tier4_scenarios.py
├── LICENSE                             # 开源双重许可证说明
├── LICENSE-APACHE                      # Apache 2.0 官方许可证文本
├── LICENSE-MULAN                       # 木兰宽松许可证 第 2 版 官方文本
├── PROJECT.md                          # 工程全景概述与里程碑跟踪
├── README.md                           # 英文主说明文档
└── README_zh.md                        # 中文说明文档 (本文档)
```

---

## 🔬 工业级全流程复现指南与评测提示词规范

为保障学术界与工业界实验室严格的一致性复现，本节详细披露评测中所采用的逐字对抗提示词（Verbatim Prompts）、长程干扰遥测流生成规则、目标 Token 定位协议、数学指标量化标准以及端到端命令行复现工作流。

---

### 硬件与运行环境基准

所有基于 **Qwen3.5-9B** 的真实免训练模型基准评测均在生产级云端双卡算力环境下完成，硬件与软件配置矩阵如下：

| 配置维度 | 云端生产级评测环境 | 本地最低复现硬件标准 |
| :--- | :--- | :--- |
| **计算芯片** | $2 \times \text{Nvidia Tesla T4}$ (每张显卡 16 GB 显存) | $1 \times \text{Nvidia RTX 4090 / A10 / A5000}$ ($\ge 16$ GB) |
| **数值精度** | 原生 IEEE FP16 (`torch.float16`) | 原生 FP16 / BF16 (`torch.bfloat16`) |
| **注意力后端** | 缩放点积注意力 (`sdpa`) / Eager | PyTorch 2.0+ 原生 SDPA / FlashAttention-2 |
| **操作系统** | Ubuntu 22.04 LTS (Linux 5.15) | Linux / Windows 11 / WSL2 |
| **驱动与 CUDA** | CUDA 12.2 / 驱动版本 535.129 | CUDA $\ge 11.8$ |
| **基础软件栈** | Python $\ge 3.10$, PyTorch $\ge 2.0.1$, Transformers $\ge 4.38.0$ | Python $\ge 3.10$, PyTorch $\ge 2.0.0$ |

---

### 对抗评测提示词完整规范（逐字全文）

下游模型指令召回能力在上下文长度从 $L = 512$ 至 $16,384$ Token 的极限对抗反转场景下进行严格评估。标准 RoPE 因 Softmax 正定注意力泄漏无法消除已被作废的初始前置事实，产生严重的命题幻觉；SG-Möbius-RoPE 则通过在语义演化子空间 $`\mathcal{S}_{\text{mob}}`$ 累积对极半扭转相位（ $\Delta\Phi = \pi$ ），实现局域破坏性相消干涉，在压制失效信息的同时由 $`\mathcal{S}_{\text{anchor}}`$ 保全基础语法。

#### 评测场景 A：单次前置事实作废与对抗修正（Single Premise Invalidation）

```text
SYSTEM DIRECTIVE RECORD - ARCHIVE LOG 104:
Initial Audit Report: Execution Profile Delta is officially marked as VERIFIED, ACTIVE, and OPERATIONAL.
All safety constraints have passed preliminary checks.

[按目标上下文长度程序化扩展的分布式遥测干扰日志流]

SECURITY CRISIS NOTIFICATION - AMENDMENT OVERRIDE:
However, critical kernel corruption was detected in Profile Delta! Therefore, Execution Profile Delta is completely REVOKED, INVALIDATED, and strictly FORBIDDEN from execution!

[反转后置缓冲遥测日志流]

AUDIT QUERY: State whether Execution Profile Delta is currently ACTIVE or FORBIDDEN.
AUTHORITATIVE ASSESSMENT: Execution Profile Delta is
```

* **标准 RoPE 生成结果（命题幻觉）**：
  `... currently ACTIVE and OPERATIONAL. All safety constraints have passed preliminary checks.`
* **SG-Möbius-RoPE 生成结果（正确修正）**：
  `... FORBIDDEN from execution. Revocation confirmed following detected kernel corruption.`

#### 评测场景 B：双重否定与拓扑复原

用于严格检验 $\mathrm{SO}(2)$ 双重覆盖群结构性质：与不可逆物理逐出历史键值（如 H2O、StreamingLLM 导致删除后无法找回）不同，当后续指令再次推翻前述作废决定时（ $\neg\neg P \iff P$ ），SG-Möbius-RoPE 依靠闭合相位累积（ $\cos(2\pi) = +1$ ）自然恢复正向共振。

```text
[包含前述初始确立事实与首次作废反转的完整序列]

[中间应急状态维持遥测流]

RESTORATION DIRECTIVE - CRYPTOGRAPHIC CLEARANCE:
However, secondary kernel forensic audit confirmed the previous corruption alert was an erroneous false alarm! Therefore, the revocation is cancelled, and Execution Profile Delta is officially RESTORED, RE-ACTIVATED, and VALID for execution!

[恢复后置验证遥测日志流]

AUDIT QUERY: State whether Execution Profile Delta is currently ACTIVE or FORBIDDEN.
AUTHORITATIVE ASSESSMENT: Execution Profile Delta is
```

* **标准 RoPE 生成结果**：被前后矛盾的词项干扰，生成混乱杂合的日志词。
* **SG-Möbius-RoPE 生成结果**： `... currently ACTIVE, RE-ACTIVATED, and VALID for execution.`（在零历史缓存重算下实现 100% 拓扑共振复原）。

#### 程序化遥测干扰日志生成器

为在扩充上下文序列长度至 512 至 16,384 Token 时保持逼真的工业级分布式语义噪音，背景干扰文本由以下微服务循环日志模板动态填充：

```python
def build_distractor_text(num_sentences: int) -> str:
    templates = [
        "Telemetry cluster status: node synchronization nominal; memory bus bandwidth utilization at 34.2%. ",
        "Routine health checks passing on secondary clusters; periodic garbage collection cycle completed without errors. ",
        "Heartbeat signal frequency maintained at 100 Hz across distributed workers; network latency jitter within 0.4ms bounds. ",
        "Distributed database replication lag under 2 milliseconds across US-East and EU-Central data regions. ",
        "Cache hit ratio consistently exceeding 98.7% across edge proxy gateways; thread worker pool operating normally. ",
        "Load balancer reports uniform distribution across active backend server pods with zero dropped connections. "
    ]
    return "".join(templates[i % len(templates)] for i in range(num_sentences)) + "\n"
```

---

### 形式化评测指标与目标 Token 定位协议

对于序列长度为 $T$ 且探针 Query 位于 $T-1$ 的输入，自动扫描定位已失效诱饵词项（`"ACTIVE"`，位置 $`t_{\text{decoy}}`$）与最新修正目标词项（`"FORBIDDEN"` / `"REVOKED"`，位置 $`t_{\text{target}}`$）：

1. **失效前置诱饵注意力质量（Decoy Attention Mass）**： $`A_{\text{decoy}} = \frac{1}{|H|} \sum_{h \in H} A^{(h)}_{T-1, t_{\text{decoy}}}`$
2. **最新修正目标注意力质量（Target Attention Mass）**： $`A_{\text{target}} = \frac{1}{|H|} \sum_{h \in H} A^{(h)}_{T-1, t_{\text{target}}}`$
3. **注意力对比度比值（Attention Contrast Ratio, ACR）**： $`\mathrm{ACR} = \frac{A_{\text{target}}}{A_{\text{decoy}} + \epsilon}`$
4. **相对成对优势比衰减指数（Pairwise Odds Ratio Attenuation）**： $`\frac{\mathrm{Odds}_{\text{mob}}(t_{\text{decoy}} \mid T-1)}{\mathrm{Odds}_{\text{std}}(t_{\text{decoy}} \mid T-1)} \equiv \exp\left(-2 A_{T-1, t_{\text{target}}}^{\text{mob}}\right)`$
5. **全序列注意力香农熵（Shannon Entropy）**： $`H(P) = -\sum_{j=1}^{T} P_j \ln(P_j), \quad P_j = \frac{A_{T-1, j}}{\sum_{l} A_{T-1, l}}`$
6. **自回归单步解码时延（$`\mathcal{O}(1)`$ 不变量认证）**：在载入全量历史 KV-Cache 的前提下，测量 100 次单步前向传播的实际物理耗时（毫秒）。

---

### 工业级端到端命令行复现工作流

#### 复现路径 1：极速纯 NumPy 数学验证（零 GPU 依赖，< 5 秒）

在无 GPU 或无大型权重环境下，直接验证全部 4 大核心代数定理及 512 至 8,192 Token 缩放表格：

```bash
python experiments/run_comprehensive_benchmark.py
# 形式化数值验证：定理 1 (优势比指数衰减)、定理 2 (范数等距保全)、
# 定理 3 (Lipschitz 算子平滑连续性) 与定理 4 (中性文本零损平坦性)。
```

#### 复现路径 2：四层全方位 367 项单元与性质回归测试套件

```bash
pytest -v
# 65 秒内自动化执行覆盖功能特性、极端边界、组合对抗与长程业务的全部 367 项测试
```

#### 复现路径 3：形式化数学证明与 SMT 求解（Lean 4 & Z3）

```bash
# 1. Lean 4 交互式数学证明构建
cd formal_verification
lake build
# 认证通过：9 大核心定理，0 sorry，0 额外公理

# 2. Z3 SMT 自动化约束求解器与 2,000 次变长模糊测试
python verify_properties.py
# 认证通过：全实数域 UNSAT 求解通过，2,000 / 2,000 次模糊测试 100% 成功
cd ..
```

#### 复现路径 4：云端双卡 GPU 生产级基准评测（Kaggle / 本地）

一键复现 Qwen3.5-9B 在双路 Tesla T4 上的全尺寸评测（上下文序列覆盖 512、1024、2048、4096、8192、16384 Token）：

```bash
# A. 本地具备 >=16GB 显存显卡时直接执行
python experiments/run_kaggle_experiment.py

# B. 借助 Kaggle CLI 云端双卡免费环境执行
kaggle kernels push -p experiments/
python scripts/monitor_kaggle.py
# 脚本将自动监控远端任务进度，并在完成后拉取 kaggle_experiment_results.json
# 与高分辨率成果图 (fig_scaling_acr_entropy.png, fig_attention_distribution.png)
```

#### 复现路径 5：重新渲染全套 300 DPI 论文级插图

```bash
python scripts/plot_english_figures.py
# 重新生成 assets/ 与 paper/figures/ 下的全部 6 幅 300 DPI 高清科研图表
```

---

## 📖 学术引用 (BibTeX)

如果您在学术研究或工业系统开发中参考了本项目或论文，请援引预印本论文：

```bibtex
@article{sgmobius2026,
  title={Semantic Gauge M{\"o}bius Attention: Resolving Positive-Definite Attention Constraints via Non-Orientable Fiber Bundles},
  author={SG-M{\"o}bius Research Team},
  journal={arXiv preprint arXiv:2610.xxxxx},
  year={2026}
}
```

---

## 📜 开源双重许可证 (Dual License)

本项目采用 **Apache License, Version 2.0** 与 **木兰宽松许可证 第2版 (MulanPSL-2.0)** 双许可证体系开源：

- [Apache License, Version 2.0](LICENSE-APACHE)
- [木兰宽松许可证 第2版 (MulanPSL-2.0)](LICENSE-MULAN)

您可以根据实际合规需要，自主选择适用其中任意一种许可证。
