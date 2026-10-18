# SG-Möbius Attention (子空间反相注意力)
## 🤖 AI 开发者与全自动 Agent 终极工作交接规范 (AI Handover & Maintenance Guide)

> **版本**：v4.0 (Production & Camera-Ready Grade)  
> **更新时间**：2026-10-18  
> **适用对象**：接手本项目的后续 AI Assistant (Cursor, Claude Code, Antigravity, Devin, Codex, Windsurf, Copilot 等) 及人类维护者。  
> **核心原则**：**零信任（Zero-Trust）、单一真理源（SSOT）、代数与工程 100% 闭环、零幻觉（Zero-Hallucination）**。

---

### 目录 (Table of Contents)
1. [项目核心使命与架构背景](#1-项目核心使命与架构背景)
2. [数学推导与理论单一真理源 (SSOT)](#2-数学推导与理论单一真理源-ssot)
3. [仓库拓扑结构与文件职责分类](#3-仓库拓扑结构与文件职责分类)
4. [后续 AI 绝对禁止触犯的“红线” (Anti-Hallucination & Golden Rules)](#4-后续-ai-绝对禁止触犯的红线-anti-hallucination--golden-rules)
5. [标准研发、测试与 CI 门禁操作指令集](#5-标准研发测试与-ci-门禁操作指令集)
6. [往轮审稿交锋核心争议与已定论对账清单](#6-往轮审稿交锋核心争议与已定论对账清单)
7. [当前版本状态与后续演进路线图](#7-当前版本状态与后续演进路线图)

---

### 1. 项目核心使命与架构背景

#### 1.1 解决的行业核心痛点
标准 Transformer 在长文本推理中存在致命的**前因条件干扰与失效指令泄露（Premise Invalidation & Softmax Leakage）**：
- **Softmax 严格正性限制**：标准自注意力机制的权重永远是正数（$A_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_l \exp(q_i^\top k_l / \sqrt{d})} > 0$）；
- **上下文推翻失效**：即便后文明确写入了“作废前述 A 规则，改用 B 规则”，模型给历史中的“A 规则”分配的注意力依旧大于零，导致长程记忆中作废信息残留，产生严重的条件倒置幻觉。

#### 1.2 核心解法：子空间反相旋转（SG-Möbius-RoPE）
本项目在开源主流模型（实测验证：**Qwen3.5-9B**）上构建了一套免训练、即插即用、硬件友好的子空间相位相消机制：
1. **子空间正交分解（Subspace Decoupling）**：
   将 RoPE 位置旋转空间划分为：
   $$\mathbb{R}^d = \mathcal{S}_{\text{anchor}} \oplus \mathcal{S}_{\text{mob}}$$
   - **锚点子空间 $\mathcal{S}_{\text{anchor}}$（高频通道）**：旋转基频焊死不动，保留标准 RoPE 位置编码，确保语法、标点与局部上下文表征不崩塌；
   - **演化子空间 $\mathcal{S}_{\text{mob}}$（低频通道）**：引入由语篇语义驱动的连续相位流 $\Phi(t)$。
2. **主动反相抵消（Destructive Phase Cancellation）**：
   当检测到语篇转折与否定时，$\mathcal{S}_{\text{mob}}$ 的相对相移旋转半圈（$\Delta\Phi = \pi$），使得该子空间内积严格变号：
   $$\mathbf{R}(\theta_m + \pi) = -\mathbf{R}(\theta_m) \implies \langle q_i, k_j \rangle_{\mathcal{S}_{\text{mob}}} = -\langle q_i, k_j \rangle_{\mathcal{S}_{\text{mob}}}^{\mathrm{RoPE}}$$
   从而使 pre-Softmax logit 变成深负数，经 Softmax 指数压缩后将作废信息权重压制至接近零。
3. **否定之否定原地复活（Cyclic Restoration）**：
   若后续出现双重否定（如“恢复原方案”），相位流再转半圈回到 $2\pi$（$\cos(2\pi) = +1$），无需修改任何历史 KV-Cache 即可原地恢复初始关注度。

---

### 2. 数学推导与理论单一真理源 (SSOT)

后续 AI 在修改任何论文源文件（`paper/main.tex`, `paper/main_zh.tex`）或代码（`src/`）时，**必须严格遵循以下已形式化闭环的数学定义与命题体系**：

#### 2.1 语义转移探测器与死区门控（Definition 1）
- **标量未缩放投影**：
  $$z_t \triangleq w_g^\top h_t + b_g$$
  其中单位方向向量 $w_g \triangleq \frac{\mu_{\text{rev}} - \mu_{\text{cont}}}{\|\mu_{\text{rev}} - \mu_{\text{cont}}\|_2} \in \mathbb{R}^d$ 由无偏 unembedding 词表中反转词集合 $\mathcal{V}_{\text{rev}}$ 与中性词集合 $\mathcal{V}_{\text{cont}}$ 的质心差分得到。
- **软饱和死区硬阻断门控（Softsign Dead-Zone Gate）**：
  $$g(h_t) \triangleq \omega_{\max} \cdot \mathrm{softsign}\Big(\mathrm{ReLU}\big(|z_t| - \tau_{\text{gate}}\big) \cdot \mathrm{sgn}(z_t)\Big)$$
  - 当 $|z_t| < \tau_{\text{gate}}$ 时，$\mathrm{ReLU}(|z_t| - \tau_{\text{gate}}) = 0 \implies g(h_t) \equiv 0$（说明文常规文本零扰动，困惑度 $7.48$ 零恶化）；
  - 当 $|z_t| \ge \tau_{\text{gate}}$ 时，平滑饱和激活，且输出严格收敛在 $[-\omega_{\max}, \omega_{\max}]$ 之间。
- **一阶粘性平滑（Viscous Smoothing）**：
  $$\omega_t = (1 - \nu)\omega_{t-1} + \nu g(h_t), \quad \bar{\Phi}_t = \sum_{\tau=0}^t \omega_\tau$$
  > **数值基准注意**：默认推荐配置 $\omega_{\max} = 0.05, \tau_{\text{gate}} = 0.50, \nu = 0.5$。若超阈值投影 $z_t = 1.0$，则原始门控 $g(h_t) = 0.05 \times \frac{0.5}{1.5} = 0.01667$ rad/step，初始步平滑角速度为 $\omega_t = 0.5 \times 0.01667 = 0.00833$ rad/step。

#### 2.2 推论 4（Corollary 4）：全前缀配分函数收缩与概率质量缓冲机制
- **名义收缩（$\Delta\Phi \equiv \pi$）**：
  失效前缀区间 $\mathcal{P} = \{l \mid l < t_2\}$ 内所有 token 的演化子空间 logit 变负，全局配分函数分母收缩：
  $$\Delta Z_{\mathcal{P}} \triangleq Z_i^{\mathrm{std}} - \tilde{Z}_i = \sum_{l \in \mathcal{P}} \exp(A_{il}^{\mathrm{std}})\left(1 - \exp(-2A_{il}^{\mathrm{mob}})\right) > 0$$
- **目标指令注意力概率放大**：
  $$P_{\mathrm{mob}}(k \mid i) = \frac{\exp(A_{ik}^{\mathrm{std}})}{Z_i^{\mathrm{std}} - \Delta Z_{\mathcal{P}}} = P_{\mathrm{std}}(k \mid i) \cdot \left(\frac{1}{1 - \frac{\Delta Z_{\mathcal{P}}}{Z_i^{\mathrm{std}}}}\right) > P_{\mathrm{std}}(k \mid i)$$
- **分母收缩对诱饵表观概率的缓冲（Cushioning Mechanism）**：
  在 $L=1,116$ 尺度下，前缀包含 576 个 token（含句首 Attention Sink $l=0$）。由于全局分母缩小约 $3.48\times$，将原本沉淀于前缀和 Attention Sink 中的概率质量强力注入目标指令（使 $P(\text{target})$ 暴涨 $3.491\times$）；而诱饵 token 虽然自身未归一化权重下降约 $5.65\times$，但由于分母同时缩小了 $3.48\times$，其实际观测到的归一化概率降幅被缓冲至 $5.65 / 3.48 \approx 1.62\times$。
  由此，目标/诱饵对比度 ACR 复合跃升：$3.491 \times 1.619 \approx 5.65\times$（$+465.1\%$，从 $3.62$ 跃升至 $20.44$），代数与宏观实验完全闭环。
- **扰动展开与正交分量界**：
  在存在有界相移扰动 $\Delta\Phi = \pi \pm \delta_l$（$|\delta_l| \le \pi/6$）时，二维平面旋转引入正交项：
  $$\tilde{A}_{il} = A_{il}^{\mathrm{std}} - (1 + \cos\delta_l)A_{il}^{\mathrm{mob}} \pm \sin\delta_l B_{il}^{\mathrm{mob}}$$
  其中 $B_{il}^{\mathrm{mob}} \triangleq \frac{1}{\sqrt{d}}\sum_m (q_i^{(m)})^\top \mathbf{J} \mathbf{R}((l-i)\theta_m) k_l^{(m)}$，$\mathbf{J} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$。收缩充要条件为 $(1+\cos\delta_l)A_{il}^{\mathrm{mob}} > |\sin\delta_l B_{il}^{\mathrm{mob}}|$。

#### 2.3 命题 5（Proposition 5）：KV-Cache 会话级不变性与解码等价
- 在会话初始化时全局冻结会话预算 $L_{\max}$，由此固定子空间截断比 $\alpha(L_{\max}) \in [0.5, 0.875]$ 与子空间掩码 $\mathbf{m}_{\text{sub}}$；
- 严密保证历史已缓存 Key $\tilde{k}_j$（$j < t$）在后续自回归自增生成时不发生任何字节级篡改或追溯重写（Byte-for-byte immutable），解码步保持严格 $\mathcal{O}(1)$ 复杂度。

#### 2.4 命题 6（Proposition 6）：Higham FP16 数值前向误差界
- 在评测基准所覆盖的至多 $K \le 10$ 次语篇反转场景下（总变差 $M_\Phi \triangleq \sum |\omega_\tau| \le 10\pi$），树状前缀求和扫描在 IEEE FP16（$u = 2^{-11}$）下的前向误差界满足：
  $$|\Delta \bar{\Phi}_L| \le 15 \times (4.88 \times 10^{-4}) \times (10\pi) \approx 0.230\,\text{rad} \approx 13.2^\circ < \frac{\pi}{6}$$
- 严格位于周期势能 $V(\bar{\Phi}) = -\frac{\eta}{2}\cos(2\bar{\Phi})$ 的自稳定吸引势阱内，杜绝偶发符号翻转；
- 对 $K > 10$ 的极端密集语境，采用 FP32 累加标量前缀和再转 FP16，可将长达 $L=10^6$ 的超长文本累积漂移控制在 $< 10^{-5}$ rad 以内。

---

### 3. 仓库拓扑结构与文件职责分类

```
算法研究/ (mobius-attention)
├── src/                                  # [核心工程源码] - 任何修改必须保证 407 测试全通
│   ├── gauge_predictor.py                # 语篇转换探测器、门控 softsign、质心构建、粘性平滑 (612 行)
│   ├── mobius_attention.py               # 子空间解耦注意力层实现、RoPE 旋转注入
│   └── kv_cache.py                       # 增量自回归解码与 KV-Cache 管理器
├── formal_verification/                  # [Lean 4 形式化数学证明] - 严禁破坏 0 sorry 保证
│   ├── MobiusAttention.lean              # 9 大核心定理形式化证明（实数域反相收缩与一阶稳定性）
│   ├── lakefile.toml                     # Lean 4 Lake 构建配置
│   ├── lean-toolchain                    # 固定 Lean 4.33.1 工具链
│   └── verify_properties.py              # Z3 SMT 普遍 UNSAT 证明与 2000 次性质模糊测试
├── tests/                                # [端到端测试套件] - 407 项测试（4 大层次 + 鲁棒性）
│   ├── test_gauge_gate_definition1.py    # 针对 Definition 1 门控软饱和、死区拦截与质心差分的测试
│   ├── test_prefill_decoding_equivalence.py # Prefill 与 Step 解码机器精度等价测试 (< 10^-12)
│   ├── test_m2_continuous_mobius.py      # 连续相位流与 2D 旋转内积断言
│   ├── test_causality.py                 # 因果时间不可倒流与严格下三角矩阵掩码
│   ├── test_tier1_features.py            # 第一层：基础功能单元测试 (60 项)
│   ├── test_tier2_boundaries.py          # 第二层：极端边界条件测试 (60 项)
│   ├── test_tier3_combinations.py        # 第三层：复杂组合对抗测试 (12 项)
│   ├── test_tier4_scenarios.py           # 第四层：现实长文本业务场景测试 (5 项)
│   └── test_readme_latex.py              # 文档与数学公式 AST 一致性自动化测试 (14 项)
├── paper/                                # [NeurIPS / ICLR 投稿级论文源文件]
│   ├── main.tex                          # 英文主论文源文件 (521 行)
│   ├── main_zh.tex                       # 中文对照版论文源文件 (506 行)
│   ├── references.bib                    # 严格校验的学术参考文献数据库 (15 篇真值，零幻觉)
│   ├── main.pdf                          # GitHub Actions 编译出的最新英文论文 PDF
│   ├── main_zh.pdf                       # GitHub Actions 编译出的最新中文论文 PDF
│   └── figures/                          # 论文矢量插图与 300 DPI 渲染资产
├── scripts/                              # [零信任自动化门禁与绘图脚本]
│   ├── verify_paper_claims.py            # SSOT 数据一致性、单调性修辞与 5 大审稿补丁门禁
│   ├── check_paper_integrity.py          # LaTeX 交叉引用 \ref, \cite 与 arXiv 真实性核验
│   ├── check_markdown_math.py            # GitHub Flavored Markdown 数学语法检查器
│   ├── render_scaling_figures.py         # 论文 Figure 1(a), 4(a) 等扩展性实验数据重绘脚本
│   └── plot_english_figures.py           # 英文论文专用高清全景注意力热力图绘制脚本
├── experiments/                          # [实测真值源]
│   └── kaggle_experiment_results.json    # 在双路 T4 上全量运行 Qwen3.5-9B (L=545~18,030) 的基准 JSON
├── assets/                               # README 与展示用 300 DPI 高清图像
├── .github/workflows/ci.yml              # GitHub Actions 8 并行全自动流水线
├── README.md                             # 英文全景技术白皮书与工程文档 (含 407 测试徽章)
├── README_zh.md                          # 中文版全景技术白皮书
├── CITATION.cff                          # 官方学术引用规范 (含 ORCID 0009-0008-2312-6913)
└── AI_HANDOVER.md                        # 本交接文件 (所有后续 AI 必读)
```

---

### 4. 后续 AI 绝对禁止触犯的“红线” (Anti-Hallucination & Golden Rules)

任何接入本项目的 AI Agent 必须无条件遵守以下开发红线：

#### 🛑 红线 1：绝对禁止凭空捏造行号、函数名与文件
- **事实背景**：此前部分未连网的模型假装审查 GitHub，凭空编造了不存在的行号（如谎称 `src/gauge_predictor.py` 有 1975 行，而实际只有 612 行；谎称 `paper/main.tex` 有 1682 行，而实际只有 521 行）。
- **要求**：引用任何代码或论文内容时，**必须以本地工作区的真实物理行号为准**；严禁在没有任何证据的情况下脑补“某行还有旧代码”。

#### 🛑 红线 2：绝对禁止篡改 Definition 1 的门控位置与结构
- **核心逻辑**：门控硬阻断必须在标量投影 $z_t = w_g^\top h_t + b_g$ 处计算 $\mathrm{ReLU}(|z_t| - \tau_{\text{gate}})$；
- **禁止操作**：绝对不能在 `omega_max * tanh(...)` 得到幅值后再做 `abs() >= tau_gate` 拦截，因为幅值上界为 $0.05$，而推荐阈值为 $0.50$，后置拦截会导致**所有 Token 输出恒为零的致命死锁**。

#### 🛑 红线 3：绝对禁止在学术表述中使用“虚假单调性”或颠倒机理
- **单调性修辞门禁**：对压制比数列 $[1.90, 1.62, 1.17, 1.13, 1.27, 1.34]$，严禁使用“strictly monotonic suppression”，必须统一使用“strictly above-unity suppression ($> 1.0\times$)”；
- **配分函数分母表述**：严禁写“抬高诱饵分母”，必须准确表述为“**配分函数收缩降低了全局分母，部分抵消/缓冲了诱饵 logit 下降带来的表观概率降幅**”。

#### 🛑 红线 4：绝对禁止夸大数值边界与误差承诺
- **命题编号**：Proposition 5 恒为 KV-Cache 解码等价，Proposition 6 恒为 Higham FP16 界；
- **误差适用前提**：严禁宣称 FP16 稳定性界在“任意文本、无条件成立”，必须明确限定在“评测基准所覆盖的至多 $K \le 10$ 次语篇反转场景下（$M_\Phi \le 10\pi$）”；
- **混合精度界限**：严禁写“FP32 across arbitrary sequence lengths”，必须严谨表述为“在超长文本（$L \le 10^6$）下将前缀累积漂移控制在 $< 10^{-5}$ rad 以内”。

#### 🛑 红线 5：绝对禁止破坏 Lean 4 形式化证明
- `formal_verification/MobiusAttention.lean` 包含 9 大核心定理；
- 严禁引入任何 `sorry`，严禁新增自定义公理（custom axioms），必须保持 `lake build`（1946 jobs）100% 成功构建。

---

### 5. 标准研发、测试与 CI 门禁操作指令集

在完成任何代码修改或论文润色后，后续 AI 必须在终端中按顺序执行以下指令验证闭环：

#### 5.1 运行全量单元测试与模糊测试 (407 项)
```bash
# 执行全部 407 项测试（利用多进程加速，耗时约 25 秒）
pytest tests/ -n auto -q

# 预期输出：407 passed in ~25s
```

#### 5.2 运行零信任论文断言与修辞门禁
```bash
# 验证 Table 1 浮点计算、ACR 统一定义、Captions 与 Kaggle JSON 绝对一致
python scripts/verify_paper_claims.py

# 预期输出：[RESULT] PASSED: All static claim, terminology, and arithmetic assertions passed!
```

#### 5.3 运行 LaTeX 语法、文献与交叉引用门禁
```bash
# 核验英文主论文
python scripts/check_paper_integrity.py paper/main.tex

# 核验中文主论文
python scripts/check_paper_integrity.py paper/main_zh.tex

# 预期输出：[RESULT] PASSED: All paper integrity and anti-hallucination audits passed!
```

#### 5.4 验证 Lean 4 形式化证明
```bash
cd formal_verification
lake build
cd ..

# 预期输出：Build completed successfully (1946 jobs).
```

#### 5.5 校验 Markdown 数学排版
```bash
python scripts/check_markdown_math.py README.md
python scripts/check_markdown_math.py README_zh.md
pytest tests/test_readme_latex.py

# 预期输出：0 corrupted math lines, 14 passed
```

#### 5.6 代码质量审查
```bash
ruff check src/ tests/ scripts/

# 预期输出：All checks passed!
```

---

### 6. 往轮审稿交锋核心争议与已定论对账清单

为了防止后续 AI 被外部模糊评审意见带偏，此处对关键争议做最终定论备份：

| 争议点 / 审稿人疑虑 | 历史状态与漏洞 | 当前已定论解决方案 (Ground Truth) |
| :--- | :--- | :--- |
| **门控死锁问题** | 门控在 $\omega_{\max} \tanh(z)$ 后判死区，导致推荐参数下恒为 0 | 门控前移至投影 $z_t$，严格使用 $\text{softsign}(\text{ReLU}(|z| - \tau)\text{sgn}(z))$，超阈值平滑输出 |
| **角速度数值混淆** | 曾将超阈值角速度粗略写为 0.01667 | 严格区隔：原始门控 $g = 0.01667$，平滑后角速度 $\omega_t = \nu g = 0.00833$，测试已断言 |
| **推论 4 机理阐述** | 曾将分母效果错写为“抬高诱饵分母” | 修正为“配分函数收缩降低全局分母，缓冲诱饵概率降幅”，解释了单点 $1.62\times$ 与 ACR $+465.1\%$ |
| **Higham 误差界前提** | 曾宣称“无条件成立”且 FP32“任意长度无误差” | 明确限定在基准评测覆盖的 $K \le 10$ 次反转语境下；FP32 限定在 $L \le 10^6$ 上下文下漂移 $< 10^{-5}$ rad |
| **Lean 4 覆盖边界** | 曾被质疑虚假宣传证明了 FP16 浮点 | 论文已明确声明：Lean 4 证明的是实数域下的正交反相与一阶李雅普诺夫稳定性，数值误差归入 Proposition 6 |
| **香农信息熵波动** | 曾试图淡化 Table 3 中的熵波动 | 在正文 Section 5.3 坦诚披露 $L=2,250$ 处的 $+0.174$ 瞬态熵波动（$6.592 \to 6.766$），消除指控 |
| **测试套件总数** | 历史版本曾从 367 扩展到 404，现扩展到 407 | 407 项测试（新增 3 项针对 Definition 1 门控软饱和测试），README 与代码库 100% 同步 |

---

### 7. 当前版本状态与后续演进路线图

#### 7.1 当前版本状态 (Current State)
- **Git Commit**：`6ec8f15` (同步至 GitHub `origin/main`)
- **CI / CD 状态**：8 个并行作业（Lean 4、pytest、Ruff、LaTeX 英文/中文 PDF 编译等）持续保持 100% 绿标；
- **论文稿件**：18 页完整 NeurIPS / ICLR 投稿格式中英文 PDF 编译完成，图表标题纯净无冗余，公式与编号完全对齐。

#### 7.2 推荐后续 AI 推进的高阶任务 (Future Roadmap)
若后续 AI 接收到进一步优化或深化研究的需求，建议优先推进以下方向：
1. **FlashAttention-2 / Triton 算子融合**：
   当前实现已将相位角速度增量融合在 QKV Projection Epilogue。可进一步开发 Triton / CUDA 原生 Kernel，将 2D 旋转内积与 FlashAttention-2 核心循环原地融合，彻底消除显存往返；
2. **多任务评测基准拓展 (Multi-Task Evaluation)**：
   在现有的 Inverted NIAH 之外，扩展 RULER 基准中的多跳反转追踪任务（Multi-turn Conflict Resolution & Tracking）；
3. **针对 Camera-Ready 的格式微调**：
   根据目标会议（NeurIPS / ICLR / ICML）最新的正文页数要求，进行附录拆分与排版微调。

---
**致接手的 AI 同仁**：本仓库的代码与理论经过多轮极为苛刻的学术刺客级审计，所有的数学公式、边界界定与测试断言均已做到极致严谨。请务必继承本指南中的“零信任”原则，任何改动务必跑通 `verify_paper_claims.py` 与 407 项 pytest，保持本项目的工业级与顶会级品质！
