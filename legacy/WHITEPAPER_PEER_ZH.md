# 📐 子空间反相注意力（SG-Möbius-RoPE）同行技术白皮书

> **面向读者**：机器学习研究员、大语言模型（LLM）算法工程师、Transformer 架构与形式化验证同行  
> **基准模型与验证标准**：Qwen3.5-9B（FP16，Dual Tesla T4）、Lean 4（Mathlib4，17 条实代数与稳定性定理，0 `sorry`）、453 项单元与性质测试全绿。

---

## 1. 核心问题定义：RoPE 的单调距离归纳偏置与“过时前提泄漏”

标准旋转位置编码（RoPE, Su et al., 2024）通过正交块对角变换将词元相对距离 $i-j$ 编码为二维子空间 $\mathcal{S}_k$ 上的平面旋转角 $\theta_k(i-j)$（其中 $\theta_k = \theta_{\mathrm{base}}^{-2k/d}$ ）：

$$
f_{\mathrm{RoPE}}(x, m)_k = R\big(m\theta_k\big) x_k, \quad R(\varphi) = \begin{pmatrix} \cos\varphi & -\sin\varphi \\ \sin\varphi & \cos\varphi \end{pmatrix} \in \mathrm{SO}(2)
$$

在长上下文因果注意力中，当位置 $j$ 的早期前提（如 `Profile Delta is ACTIVE`）在位置 $t_{\mathrm{rev}} \in (j, i)$ 被显式推翻（如 `However, Profile Delta is REVOKED`）时，标准 RoPE 暴露出两个结构性缺陷：
1. **缺少布尔/拓扑极性维度**：注意力对数几率（Logit）$s_{i,j} = \frac{1}{\sqrt{d}} \sum_{k=0}^{d/2-1} q_{i,k}^\top R(\theta_k(i-j)) k_{j,k}$ 仅依赖静态内容相似度与相对距离 $i-j$ ，无法表征跨越语篇转折点的**逻辑极性反转（Premise Invalidation）**。
2. **低频语义通道的同相顽固性**：在负责长程语义聚合的低频子空间（ $\theta_k(i-j) \approx 0$ ）中， $R(\theta_k(i-j)) \approx I_2$ 。只要过时诱饵 $k_j$ 与当前查询 $q_i$ 共享高实体/主题重叠度，就会产生持续的正内积贡献，形成 **幽灵注意力泄漏（Ghost Attention Leakage）**。

---

## 2. 数学架构：子空间正交解耦与 $\mathrm{SO}(2)$ 语义联络流

我们在不破坏预训练权重、不改变 FlashAttention / SDPA 算子签名的前提下，将每个注意力头的旋转空间 $\mathbb{R}^{d_{\mathrm{rot}}}$ 正交分解为两个互补子空间：

$$
\mathbb{R}^{d_{\mathrm{rot}}} = \mathcal{S}_{\mathrm{anchor}} \oplus \mathcal{S}_{\mathrm{mob}}
$$

### 2.1 动态无混叠频段划分 $\alpha(L)$（Proposition 2）
- **高频语法锚定子空间 $\mathcal{S}_{\mathrm{anchor}}$**（前 $d_{\mathrm{anchor}} = 2\lfloor \alpha(L)\frac{d}{2}\rfloor$ 维）：保持标准 RoPE 旋转角不变（ $\Phi \equiv 0$ ），严格锁定局部词序与句法结构。
- **超低频莫比乌斯语义子空间 $\mathcal{S}_{\mathrm{mob}}$**（后 $d_{\mathrm{mob}} = d_{\mathrm{rot}} - d_{\mathrm{anchor}}$ 维）：叠加累积语义全纯相位（Holonomy Phase）$\Phi_t \in \mathbb{R}$ 。
- 为消除基波载波在上下文跨度 $L$ 内进入负半周期（ $\theta_k(i-j) > \pi/2$ ）引发的“负负得正”相位混叠（Carrier Aliasing），自适应截断率定义为：

$$
\alpha(L) = \min\left(0.875,\; \max\left(0.50,\; \frac{\ln(2L/\pi)}{\ln \theta_{\mathrm{base}}}\right)\right)
$$

  该下界保证对一切 $k \ge \alpha(L)\frac{d}{2}$ 及相对距离 $i-j \le L$ ，恒有 $\theta_k(i-j) \le \pi/2$ ，从而 $\cos(\theta_k(i-j)) \ge 0$ 。

### 2.2 死区门控与离散双稳态吸引子动力学（Definition 1 & Theorem 1）
1. **未缩放标量投影死区门控（Definition 1）**：  
   在词嵌入/隐状态投影 $z_t = w_g^\top h_t + b_g$ 上直接施加阈值 $\tau_{\mathrm{gate}} > 0$（严禁在 $\omega_{\max}\tanh(z_t)$ 之后截断，从而彻底规避 $\omega_{\max} < \tau_{\mathrm{gate}}$ 时的零锁死漏洞）：

$$
g(h_t) = \omega_{\max} \cdot \mathrm{softsign}\Big(\mathrm{ReLU}\big(|z_t| - \tau_{\mathrm{gate}}\big)\cdot \mathrm{sgn}(z_t)\Big)
$$

   - 在无转折的自然语料（ $|z_t| \le \tau_{\mathrm{gate}}$ ）上， $g(h_t) \equiv 0$ ，若初始 $\Phi_0 = 0, \omega_0 = 0$ ，则全序列 $\Phi_t \equiv 0$ ，**100% 精确退化为标准 RoPE**。
2. **双曲正切自锁吸引子与精确 $\pi$ 量子化**：  
   平滑角速度与相位递推遵循：

$$
\omega_t = (1-\nu)\omega_{t-1} + \nu g(h_t) - \eta \sin(2\Phi_{t-1}), \qquad \Phi_t = \Phi_{t-1} + \omega_t
$$

   其中势函数 $V(\Phi) = \frac{\eta}{2}\sin^2(\Phi)$ 产生的恢复力 $-\eta\sin(2\Phi)$ 在 $0 < \eta < \frac{1}{2}$ 时，使离散映射 $\Phi_{t+1} = \Phi_t - \eta\sin(2\Phi_t)$ 在 $\Phi^* = n\pi$ ($n \in \mathbb{Z}$) 处指数级渐近稳定（吸引盆覆盖 $(n\pi - \pi/2, n\pi + \pi/2)$ ，收敛率 $1 - 2\eta\cos(2\delta_0) < 1$ ）。
   - **单次否定（ $n=1$ ）**： $\Delta\Phi = \pi \implies R(\pi) = -I_2$ ，莫比乌斯子空间精确反相；
   - **双重否定（ $n=2$ ）**： $\Delta\Phi = 2\pi \implies R(2\pi) = +I_2$ ，精确恢复原始肯定极性（ $\mathbb{Z}_2$ 同余回归）。

---

## 3. 核心理论机制：配分函数收缩与多轮奇偶交替定律

为什么仅翻转低频子空间的符号，就能在实测中产生高达 $+465.1\%$ 的注意力对比度（ACR）增益？我们在 **Corollary 4**（单轮）与 **Theorem 7 / Corollary 8**（任意 $K$ 轮反转）中给出了严格的微观代数分解：

### 3.1 单轮全前缀配分函数收缩（Corollary 4）
设查询 $i$ 的注意力对数几率分解为不变部分 $a_{i,l}$（含 $\mathcal{S}_{\mathrm{anchor}}$ 与非旋转通道）与莫比乌斯子空间基波同相分量 $m_{i,l}$ 、正交分量 $q_{i,l}^{\perp}$ 。当在 $t_{\mathrm{rev}}$ 处发生首次语篇反转（ $\Delta\Phi_{i,l} = \pi + \delta_{i,l}$ 对于 $l < t_{\mathrm{rev}}$ ，而 $\Delta\Phi_{i,l} = 0$ 对于 $l \ge t_{\mathrm{rev}}$ ）：
- **过时前缀区 $l \in [0, t_{\mathrm{rev}}-1]$**：调制后对数几率发生对称破缺下降：

$$
\tilde{s}_{i,l} = s_{i,l} - (1 + \cos\delta_{i,l})m_{i,l} - \sin\delta_{i,l}  q_{i,l}^{\perp}
$$

定义每个过时词元的收缩因子 $\gamma_{i,l} = \exp\big(-(1+\cos\delta_{i,l})m_{i,l} - \sin\delta_{i,l} q_{i,l}^{\perp}\big)$ 。当 $\gamma_{i,l} < 1$ 时，整个过时前缀区的未归一化 Softmax 权重之和严格收缩：

$$
\Delta Z_{\mathrm{pre}} = \sum_{l=0}^{t_{\mathrm{rev}}-1} e^{s_{i,l}}(1 - \gamma_{i,l}) > 0 \implies \tilde{Z}_i = Z_i^{\mathrm{std}} - \Delta Z_{\mathrm{pre}} < Z_i^{\mathrm{std}}
$$

- **后文有效指令区 $l \ge t_{\mathrm{rev}}$（含目标词元 $t_{\mathrm{target}}$ ）**：由于 $\Delta\Phi_{i,l} = 0$ ，其分子 $\exp(\tilde{s}_{i,l}) = \exp(s_{i,l})$ 保持不变，但因全局分母 $\tilde{Z}_i$ 收缩，其归一化注意力概率被**全局虹吸放大**了 $Z_i^{\mathrm{std}} / \tilde{Z}_i > 1$ 倍！

### 3.2 多轮（ $K$-Turn）奇偶交替配分函数收缩定理（Theorem 7 & Corollary 8）
将上下文划分为 $K$ 次反转边界 $1 \le t_1 < t_2 < \dots < t_K \le i$ 生成的 $K+1$ 个极大同相语篇段 $\mathcal{I}_0, \mathcal{I}_1, \dots, \mathcal{I}_K$（查询 $i \in \mathcal{I}_K$ ）。定义**相对奇偶差** $\pi_r = (K - r) \bmod 2 \in \lbrace 0, 1 \rbrace$ ：
- **同相活跃段（ $\pi_r = 0$ ，即与当前查询相差偶数次反转的段，包括已通过双重否定恢复的早期段）**： $\Delta\Phi_{i,l} \equiv 0 \pmod{2\pi}$ ，未归一化质量完全守恒 $\tilde{Z}^{(r)}_i = Z^{\mathrm{std},(r)}_i$ ；
- **反相作废段（ $\pi_r = 1$ ，即与当前查询相差奇数次反转的所有历史作废段）**： $\Delta\Phi_{i,l} \equiv \pi + \delta_{i,l} \pmod{2\pi}$ ，段内配分函数严格收缩 $\Delta Z^{(r)} = \sum_{l \in \mathcal{I}_r} w_{i,l}(1-\gamma_{i,l}) > 0$ 。

由此导出三条严格的代数推论（已在 Lean 4 中完成机器形式化证明）：
1. **多轮累积配分函数收缩**：

$$
\tilde{Z}_i = Z_i^{\mathrm{std}} - \Delta Z_{\mathrm{tot}}, \qquad \Delta Z_{\mathrm{tot}} = \sum_{r: \pi_r=1} \Delta Z^{(r)} > 0
$$

2. **跨段均匀同相放大率（Uniform In-Phase Amplification）**：  
   一切同相段 $\mathcal{I}_r$（ $\pi_r = 0$ ）中的任意词元 $k$（无论是最新生效指令 $t_{\mathrm{target}} \in \mathcal{I}_K$ ，还是经双重否定恢复的早期词元 $t_{\mathrm{restored}} \in \mathcal{I}_{K-2}$ ），均获得完全一致的同相放大因子：

$$
T \triangleq \frac{\tilde{\alpha}_{i,k}}{\alpha^{\mathrm{std}}_{i,k}} = \frac{Z_i^{\mathrm{std}}}{Z_i^{\mathrm{std}} - \Delta Z_{\mathrm{tot}}} > 1
$$

3. **诱饵缓冲恒等式与 ACR 精确分解（Corollary 8）**：  
   任意反相段（ $\pi_r = 1$ ）中诱饵词元 $j$ 的净抑制比受到分母收缩的缓冲：

$$
D_j^{\mathrm{cushion}} \triangleq \frac{\alpha^{\mathrm{std}}_{i,j}}{\tilde{\alpha}_{i,j}} = \frac{1}{\gamma_{i,j}\cdot T} = \frac{Z_i^{\mathrm{std}} - \Delta Z_{\mathrm{tot}}}{\gamma_{i,j} Z_i^{\mathrm{std}}}
$$

而注意力对比度（ACR）的净增益倍数则满足**分母精确相消**的复合法则：

$$
\frac{\mathrm{ACR}_{\mathrm{M\ddot{o}bius}}(k, j)}{\mathrm{ACR}_{\mathrm{std}}(k, j)} = T \cdot D_j^{\mathrm{cushion}} = \frac{1}{\gamma_{i,j}}
$$

   这完美解释了表 1 实测现象：在 $L=1,116$ 时，尽管分母大幅收缩将目标词元放大了 $T = 3.49\times$ 、同时将诱饵的净抑制比缓冲至 $D_j^{\mathrm{cushion}} = 1.62\times$ ，两者相乘仍精确还原微观惩罚因子 $\gamma_{i,j}^{-1} = 3.49 \times 1.62 = 5.65\times$（ $+465.1\%$ ）！

---

## 4. 工程实现与数值稳定性保障（Propositions 5 & 6）

### 4.1 预填充与自回归 KV-Cache 严格等价（Proposition 5）
- **预融合旋转表（Kernel-Aligned Pre-Fusion）**：通过三角积化和差恒等式将语义相位 $\Phi_t$ 预融合进 $(\cos_{\mathrm{mod}}, \sin_{\mathrm{mod}})$ 张量，再送入原生 PyTorch SDPA / FlashAttention-2 内核，**零自定义 CUDA 内核修改**。
- **静态会话预算 $\alpha(L_{\max})$**：在自回归生成阶段，固定会话预算 $L_{\max}$ 使子空间掩码在整个 KV-Cache 生命周期内保持时不变，且 $\Phi_t$ 完全由因果前向递推生成，从而保证 **Prefill 阶段与逐词元 Decoding 阶段注意力输出逐位严格等价**，单步解码仅需 $O(1)$ 状态更新。

### 4.2 两阶段并行关联扫描与 Higham FP16 误差界（Proposition 6）
- 在非线性吸引子激活（ $\eta > 0$ ）时，递推为非仿射；在分段锁定区间（ $|\sin(2\Phi)| \le \varepsilon_{\mathrm{lock}}$ ）退化为二阶线性时不变递推，可通过 $2\times 2$ 矩阵关联扫描（Blelloch Scan）在 $O(\log L)$ 并行步内完成。
- **浮点稳定性界**：由 Higham (2002, Theorem 3.1)，在最大累积相位满足 $M_\Phi \le 10\pi$（对应 $\le 10$ 次语篇反转）且使用平衡二叉树归约时，FP16（ $u_{16} = 2^{-11}$ ）的相对旋转算子误差严格受控；而在生产级推荐配置——**算子内核保持 FP16、1D 标量相位前缀和采用 FP32 累加器（ $u_{32} = 2^{-24}$ ）** 下，即使在 $L = 10^6$ 词元超长上下文中，相位漂移仍严格小于 $10^{-5}\text{ rad}$ ，显存与延迟开销为 $0.0\%$ 。

---

## 5. 实证基准与形式化验证总览

### 5.1 Qwen3.5-9B 多尺度长上下文基准（Table 1 SSOT）
在 $L \in [545, 18,030]$ 跨度上对比 Standard RoPE、固定 $\alpha=0.50$ 与动态自适应 $\alpha(L)$ ：
- **峰值对比度增益（ $L=1,116$ ）**：Standard RoPE 的 $\mathrm{ACR} = 3.62$ ；SG-Möbius（自适应 $\alpha(L)=0.713$ ）将过时诱饵注意力从 $1.09 \times 10^{-3}$ 压制至 $6.76 \times 10^{-4}$（**$1.62\times$ 净抑制**），将有效指令注意力从 $3.96 \times 10^{-3}$ 提升至 $1.38 \times 10^{-2}$（**$3.49\times$ 同相放大**），综合 $\mathrm{ACR}$ 达 **$20.44$（ $+465.1\%$ ）**。
- **超长上下文消除混叠反转（ $L=2,250$ ）**：固定 $\alpha=0.50$ 因载波混叠跌至 $\mathrm{ACR}=0.99$（劣于基线 $1.72$ ），而自适应 $\alpha(L)=0.789$ 成功恢复单调优势（ $\mathrm{ACR}=2.17 > 1.72$ ）。
- **极限长程保持（ $L=18,030$ ）**：自适应 $\alpha(L)=0.875$ 维持 $\mathrm{ACR}=2.87$（较基线 $1.31$ 提升 $+118.4\%$ ），诱饵抑制比达 **$2.33\times$**。
- **自然语言零退化（WikiText-103）**：在真实连续门控激活下，困惑度完全保持 **$7.48 \to 7.48$（ $0.0\%$ 退化）**。

### 5.2 Lean 4 形式化验证门禁（`formal_verification/MobiusAttention.lean`）
全库包含 **17 条核心实代数与动力系统定理**（`lake build` 1946 项任务 100% 通过，0 `sorry`，0 自定义公理）：
- **几何与群结构（9 条）**： $\mathrm{SO}(2)$ 正交保范性、逆旋转恒等式、 $\pi$ 反相抵消（`mobius_anti_phase_negation`）、 $2\pi$ 双重否定恢复（`mobius_double_negation_recovery`）、子空间正交解耦、自适应频段无混叠界、1D 吸引子不动点、恢复漂移严格递减性及局部指数收敛定理（`local_attractor_exponential_contraction`）。
- **多轮配分函数收缩（8 条）**：偶数/奇数反转三角极性简化、单段收缩正性（`segment_contraction_positive`）、多轮收缩非负可加性、配分函数严格缩减（`multi_turn_partition_contraction_sum`）、跨段均匀同相放大率（`uniform_in_phase_amplification_factor`）、诱饵缓冲上界（`decoy_cushioned_suppression_bound`）以及 $K$ 轮 ACR 分母相消恒等式（`k_turn_acr_compound_identity`）。

---

## 6. 代码仓库快速索引

- **核心算子与门控实现**：[`src/mobius_rope.py`](file:///c:/Users/安卓人/Documents/antigravity/算法研究/src/mobius_rope.py)、[`src/gauge_predictor.py`](file:///c:/Users/安卓人/Documents/antigravity/算法研究/src/gauge_predictor.py)
- **Lean 4 形式化证明**：[`formal_verification/MobiusAttention.lean`](file:///c:/Users/安卓人/Documents/antigravity/算法研究/formal_verification/MobiusAttention.lean)
- **SymPy 符号验证与零信任审计脚本**：[`scripts/verify_formula_with_sympy.py`](file:///c:/Users/安卓人/Documents/antigravity/算法研究/scripts/verify_formula_with_sympy.py)、[`scripts/verify_paper_claims.py`](file:///c:/Users/安卓人/Documents/antigravity/算法研究/scripts/verify_paper_claims.py)
- **多轮收缩测试套件（46 项）**：[`tests/test_multi_turn_contraction.py`](file:///c:/Users/安卓人/Documents/antigravity/算法研究/tests/test_multi_turn_contraction.py)
