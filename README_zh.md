# 选择性反相注意力

这是一个最小研究仓库，用来验证：语义驱动的旋转能否真正压制长上下文里已经作废的前提。

**当前状态：研究重启与真实模型压力测试。** 我们把“代数正确”“注意力发生变化”“下游任务变好”分成三层证据。所有模型结论均附带原始 JSON 数据与复现命令，绝不隐瞒负结果或边界条件。

[English](README.md) · [完整研究计划](RESEARCH_PLAN.md) · [受控数学结果](experiments/controlled_math_results.json) · [Qwen2.5 报告](experiments/qwen25_reversal_report.md) · [Qwen3 报告](experiments/qwen3_06b_reversal_report.md) · [Qwen3.5 报告](experiments/qwen35_08b_reversal_report.md)

## 要解决的问题

标准 Softmax 会给每个可见 token 保留正权重。因此，后文的更正可能只是在和旧前提竞争，而不是把旧前提真正删除。

我们的做法是在标准 RoPE 之后增加语义相位旋转。但下面三种机制不能混为一谈：

| 机制 | 怎么转 | 选择性 | 历史 K 缓存 |
| --- | --- | --- | --- |
| 只转 Query | 给当前 Q 加一个额外旋转 | 所有历史 Key 一起受影响 | 不变 |
| QK 分段相位 | Q、K 保留各自 token 的相位 | 同段不变，跨段调制 | 写入一次，之后不改 |
| 点对点 / 实体绑定门控 | 每个 Query-Key 对或实体子空间单独计算门控 | 实体级或 token 对级 | 取决于实现 |

当前测试的主假设是第二种（`segment`）。它是全局 Query 旋转和 CoPE 式点对点控制之间的 O(Ld) 工程折中，但一维全局时间阶梯只能表达“时间分段 / 状态版本”的选择性，不能自动区分同一时间段内的不同实体。

## 数学能证明什么

对选中的 RoPE 二维平面，相对相位达到半圈时，这些平面贡献的注意力 logit 会变号。旋转保范数，也兼容“Key 写入时旋转、之后不改”的缓存方式。

但它不能证明：

- 只转部分通道时，总 logit 一定变负；
- 旧 token 一定被压制，因为原本为负的分量可能被反向放大；
- Softmax 权重会严格归零；
- Value 会发生抵消，或信息被真正删除；
- 一维全局时间相位能避免误伤同时间段定义的无关实体；
- 模型能学会实体级的反转识别；
- 下游答案正确率会优于同边界的加法偏置。

精确公式、训练风险、工程约束和停止条件见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。

## 最小实现

`src/phase_attention.py` 只包含：

- 标准 RoPE 表和旋转；
- Query 全局相位与 QK 分段相位；
- 带可选加法对照的因果注意力归一化；
- 因果反转相位阶梯。

核心模块没有语义检测器、拓扑包装、自定义缓存管理器、关联扫描、绘图依赖或模型动态补丁。

`experiments/kaggle_reversal_eval.py` 是独立的评测适配器，支持 `Qwen2`、`Qwen3` 和 `Qwen3.5`（含 QK-Norm、`partial_rotary_factor` 部分旋转维度，以及混合架构中的全注意力层筛选），刻意不放进核心数学模块。

## 已完成的受控反证实验

在仓库根目录运行：

```bash
python -m experiments.run_controlled_math
```

例子包含一个旧正分 Key、一个旧负分 Key 和一个新正分 Key：

| 方案 | 旧正分 | 旧负分 | 新正分 |
| --- | ---: | ---: | ---: |
| 标准 RoPE | 1.4062 | -1.4062 | 1.4062 |
| Query 全局反相 | -1.4062 | 1.4062 | -1.4062 |
| QK 分段反相 | -1.4062 | 1.4062 | 1.4062 |
| 分段反相，一半通道 | 0.5938 | -0.5938 | 1.4062 |
| 分段加法对照 | -0.5938 | -3.4062 | 1.4062 |

这是张量级检查，**不是大模型效果结果**。它直接暴露两个问题：全局 Query 反相会误伤新信息；反相也可能把旧的负分抬高。

## 测试

```bash
python -m pytest tests/test_phase_attention.py -q
python -m pytest tests/test_qwen_phase_eval.py -q
```

第一组覆盖相对位置性质、同段等价、负分反向放大、部分通道、梯度有限性和 prefill / decode 等价。

第二组使用随机初始化的微型 `Qwen2`、`Qwen3` 与 `Qwen3.5` 模型，不下载权重，验证评测补丁在零相位时与原模型输出严格一致，并能在退出后恢复原前向函数。

## Kaggle 预训练模型三代架构对比实验（Qwen2.5 vs Qwen3 vs Qwen3.5）

我们在 Kaggle 双卡 T4 上使用相同的 Oracle 反转边界，在上下文长度 512、2048、8192 下评测了三代 Qwen 架构（每个模型 3 个长度 × 4 个场景 × 8 个独立样本 × 4 种模式 = **384 次成对前向**）：

```bash
python -m experiments.kaggle_reversal_eval \
  --model Qwen/Qwen3.5-0.8B \
  --lengths 512 2048 8192 \
  --cases 8 \
  --output experiments/qwen35_08b_reversal_results.json
```

每个样本只做**一次前向**：正确与过期两个候选答案取自同一个 next-token Softmax，报告的 Margin 是成对对数概率差（正确减过期）。同一 `(长度, 场景, 样本)` 组的四种模式始终在同一张 GPU 上完成。

### 三代模型主结果汇总（跨 L = 512, 2048, 8192 的准确率 / 平均 Margin）

| 模型与旋转架构特性 | 场景 | `standard`（原模型） | `query`（只转 Query） | `segment`（QK 分段旋转） | `additive`（同边界减分） |
| --- | --- | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B-Instruct**<br>*(24 层全注意力，100% 维度旋转，无 QK-Norm)*<br>[JSON](experiments/qwen25_reversal_results.json) · [报告](experiments/qwen25_reversal_report.md) | `single`（单次推翻） | 1.000 / +3.49 | 0.792 / +1.37 | 0.833 / +1.75 | 0.000 / -2.90 |
| | `double`（二次推翻） | 1.000 / +1.85 | 1.000 / +1.20 | 1.000 / +1.69 | 1.000 / +2.71 |
| | `unrelated`（无关实体推翻） | 0.208 / -0.86 | 0.250 / -1.21 | 0.167 / -1.56 | 1.000 / +6.35 |
| | `none`（无推翻） | 1.000 / +4.41 | 1.000 / +4.41 | 1.000 / +4.41 | 1.000 / +4.41 |
| **Qwen3-0.6B**<br>*(28 层全注意力，100% 维度旋转，含 QK-Norm)*<br>[JSON](experiments/qwen3_06b_reversal_results.json) · [报告](experiments/qwen3_06b_reversal_report.md) | `single`（单次推翻） | 1.000 / +3.44 | 0.000 / -1.01 | 0.292 / -0.80 | 0.750 / +2.10 |
| | `double`（二次推翻） | 0.000 / -1.72 | 0.625 / +0.50 | 0.458 / -0.57 | 0.792 / +0.55 |
| | `unrelated`（无关实体推翻） | 0.000 / -1.40 | 1.000 / +1.13 | 0.708 / +0.70 | 0.458 / -0.33 |
| | `none`（无推翻） | 0.917 / +1.35 | 0.917 / +1.35 | 0.917 / +1.35 | 0.917 / +1.35 |
| **Qwen3.5-0.8B**<br>*(6 层全注意力 + 18 层线性注意力，25% 维度旋转)*<br>[JSON](experiments/qwen35_08b_reversal_results.json) · [报告](experiments/qwen35_08b_reversal_report.md) | `single`（单次推翻） | 1.000 / +1.37 | 0.000 / -1.05 | **1.000 / +1.82** | 1.000 / +2.17 |
| | `double`（二次推翻） | 0.000 / -0.48 | 0.625 / +0.04 | **1.000 / +0.65** | 1.000 / +0.30 |
| | `unrelated`（无关实体推翻） | 0.042 / -0.39 | 1.000 / +1.04 | 0.000 / -1.42 | 0.000 / -1.29 |
| | `none`（无推翻） | 1.000 / +0.48 | 1.000 / +0.48 | **1.000 / +0.48** | 1.000 / +0.48 |

### 三个核心实证发现

1. **为什么 `Qwen3.5-0.8B`（25% 部分旋转）零样本直接生效，而 `Qwen2.5` / `Qwen3`（100% 旋转）会退化**：
   `Qwen2.5` 和 `Qwen3` 的 `partial_rotary_factor = 1.0`（每个注意力头 100% 的维度都参与 RoPE 旋转）。在未经微调的模型上把全部 100% 通道直接翻转半圈，会连带破坏负责基础语法和局部定位的注意力特征（`Qwen3-0.6B` 在 `single` 场景从 `1.000` 跌至 `0.292`）。
   相反，`Qwen3.5-0.8B` 原生采用 `partial_rotary_factor = 0.25`（256 维 head_dim 中仅前 64 维参与 RoPE 旋转，后 192 维即 75% 是完全不旋转的语义锚定子空间），且仅在 6 层全注意力层做几何旋转。在 `Qwen3.5-0.8B` 上，`segment` 分段旋转将撤销场景准确率从 **0.500 提升至 1.000（+50.00 pp）**：不仅把原模型完全答错的 `double`（二次反转：valid -> invalid -> valid）从 `0.000` 恢复到 `1.000`，还让 `single` 场景的正向 Margin 在所有长度上全面扩大（`+1.37 -> +1.82`）。
2. **定量验证了两个核心理论批评**：
   - **「只转 Query 相当于让全体 Key 一起转」**：`query` 模式在 `Qwen3-0.6B` 和 `Qwen3.5-0.8B` 的 `single` 场景下准确率均为 **`0.000`**。因为只转 Query 时，前文的旧前提与后文的修订句同时相对 Query 反相，无法区分新旧段落。
   - **「一维全局时间相位必然误伤无关实体（`unrelated` 场景）」**：当 `ALPHA` 在开头第 0 段定义、后文插入无关实体 `BETA` 的反转（将全局相位推至半圈），最后提问 `ALPHA` 时，`ALPHA` 同样承受了跨段反相惩罚（`Qwen3.5-0.8B` 上 `unrelated` 准确率为 `0.000`，Margin 从 `-0.39` 恶化至 `-1.42`）。
3. **实体绑定相位（`entity_segment`）在 `Qwen3.5-0.8B` 上彻底解决跨实体误伤并击败同边界加法偏置（[JSON](experiments/qwen35_08b_entity_results.json) · [报告](experiments/qwen35_08b_entity_report.md)）**：
   当每个 token 的静态标量相位绑定到其所属实体的版本计数（[`src/phase_attention.py`](src/phase_attention.py) 中的 `entity_reversal_phase`，依然在写入 KV Cache 时一次性旋转、复杂度保持 $O(Ld)$ 且无需 $O(L^2)$ 矩阵）时，`Entity BETA` 的反转只会将 `BETA` 段落旋转半圈，而 `Entity ALPHA` 的早期定义与末尾提问均保持 0 相位：
   - 在 `single`、`double`、`unrelated`、`none` **全部四个场景与全部三个长度（512 / 2048 / 8192）上，`entity_segment` 取得 1.000 准确率（96/96 全对）**。
   - 在 `unrelated` 场景下，同样给定完全相同的 Oracle 实体边界，同边界加法偏置对照（`entity_additive`）准确率仅为 **0.500（Margin -0.02）**，而 `entity_segment` 相位旋转达到 **1.000（Margin +0.68，领先 +50.00 pp）**，从而使预注册的四项判据全部通过（有效性 +50.00 pp、全场景旋转必要性 +12.50 pp、安全场景退化 -47.92 pp、延迟开销 -4.24%）。下一步的核心问题是脱离 Oracle 边界训练该实体状态门控（E3）。

## 仓库结构

```text
src/phase_attention.py                       最小可微相位干预算子
experiments/run_controlled_math.py           确定性张量反证矩阵
experiments/kaggle_reversal_eval.py          Qwen2 / Qwen3 / Qwen3.5 评测框架
experiments/controlled_math_results.json     受控数学实验输出
experiments/qwen25_reversal_results.json     Qwen2.5-0.5B-Instruct Kaggle E2 结果
experiments/qwen25_reversal_report.md        Qwen2.5-0.5B-Instruct 预注册判据报告
experiments/qwen3_06b_reversal_results.json  Qwen3-0.6B Kaggle E2 结果
experiments/qwen3_06b_reversal_report.md     Qwen3-0.6B 预注册判据报告
experiments/qwen35_08b_reversal_results.json Qwen3.5-0.8B Kaggle E2 结果
experiments/qwen35_08b_reversal_report.md    Qwen3.5-0.8B 预注册判据报告
scripts/build_kaggle_kernel.py               Kaggle 双 T4 单文件打包脚本
scripts/monitor_kaggle_e2.py                 Kaggle 任务轮询与结果下载
scripts/summarize_kaggle_e2.py               预注册判据核对与报告生成
tests/test_phase_attention.py                核心相位算子单元测试
tests/test_qwen_phase_eval.py                Qwen2 / Qwen3 / Qwen3.5 零相位等价测试
RESEARCH_PLAN.md                             假设、实验阶段和停止条件
legacy/                                      旧版 Agent 原型与历史交接文档归档
paper/                                       历史论文手稿（不代表当前路径结论）
formal_verification/                         Lean 4 历史代数形式化证明（17 条定理）
```

旧版 SG-Möbius 模块（`src/mobius_attention.py`、`src/gauge_predictor.py`、`src/kv_cache.py`）、`paper/` 和 `formal_verification/` 仍保留用于回归追溯。它们的测试和证明是代数自洽性回归检查，不代表“模型学会失效前提”或“下游准确率提升”的证据。

## 相关真实论文

- [CoPE，arXiv:2405.18719](https://arxiv.org/abs/2405.18719)：Query-Key 对级的上下文位置；原论文尚未验证数十亿参数模型。
- [Round and Round We Go，arXiv:2410.06205](https://arxiv.org/abs/2410.06205)：RoPE 频率分工、语义漂移与 p-RoPE。
- [Differential Transformer，arXiv:2410.05258](https://arxiv.org/abs/2410.05258)：用两个 Softmax 注意力图之差抑制噪声，ICLR 2025 Oral。
- [Selective Attention，arXiv:2411.12892](https://arxiv.org/abs/2411.12892)：学习控制注意力稀疏度与相关性。

## 许可证

Apache-2.0 与 MulanPSL-2.0，详见现有许可证文件。
