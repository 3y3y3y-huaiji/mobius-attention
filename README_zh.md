# 选择性反相注意力

这是一个最小研究仓库，用来验证：语义驱动的旋转能否真正压制长上下文里已经作废的前提。

**当前状态：研究重启，不是已经验证的论文结论。** 我们把“代数正确”“注意力发生变化”“下游任务变好”分成三层证据。没有结果 JSON 和复现命令，就不宣称新的模型实验成功。

[English](README.md) · [完整研究计划](RESEARCH_PLAN.md) · [受控数学结果](experiments/controlled_math_results.json)

## 要解决的问题

标准 Softmax 会给每个可见 token 保留正权重。因此，后文的更正可能只是在和旧前提竞争，而不是把旧前提真正删除。

我们的做法是在标准 RoPE 之后增加语义相位旋转。但下面三种机制不能混为一谈：

| 机制 | 怎么转 | 选择性 | 历史 K 缓存 |
| --- | --- | --- | --- |
| 只转 Query | 给当前 Q 加一个额外旋转 | 所有历史 Key 一起受影响 | 不变 |
| QK 分段相位 | Q、K 保留各自 token 的相位 | 同段不变，跨段调制 | 写入一次，之后不改 |
| 点对点门控 | 每个 Query-Key 对单独计算门控或偏置 | token 对级 | 取决于实现 |

当前主假设是第二种。它是全局 Query 旋转和 CoPE 式点对点控制之间的工程折中，但只能表达“分段 / 状态版本”的选择性，不能任意抹除某个实体。

## 数学能证明什么

对选中的 RoPE 二维平面，相对相位达到半圈时，这些平面贡献的注意力 logit 会变号。旋转保范数，也兼容“Key 写入时旋转、之后不改”的缓存方式。

但它不能证明：

- 只转部分通道时，总 logit 一定变负；
- 旧 token 一定被压制，因为原本为负的分量可能被反向放大；
- Softmax 权重会严格归零；
- Value 会发生抵消，或信息被真正删除；
- 模型能学会实体级的反转识别；
- 下游答案正确率会提高。

精确公式、训练风险、工程约束和停止条件见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。

## 最小实现

`src/phase_attention.py` 只包含：

- 标准 RoPE 表和旋转；
- Query 全局相位与 QK 分段相位；
- 带可选加法对照的因果注意力归一化；
- 因果反转相位阶梯。

核心模块没有语义检测器、拓扑包装、自定义缓存管理器、关联扫描、绘图依赖或模型动态补丁。

`experiments/kaggle_reversal_eval.py` 是独立的 Qwen2 评测适配器，刻意不放进核心数学模块。

## 已完成的受控反证实验

运行：

```bash
python -m experiments.run_controlled_math
```

必须在仓库根目录运行：模块入口让 `src` 可直接导入，不需要 `sys.path` 补丁。

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

第二组使用随机初始化的微型 Qwen2，不下载权重，验证评测补丁在零相位时与原模型输出一致，并能在退出后恢复原前向函数。它还固定了成对评分的单次前向契约：两个候选答案的对数概率都必须有限、非正，并且共享同一个 Softmax 归一化因子。

## Kaggle 模型实验

第一轮使用 Qwen2.5-0.5B-Instruct。四种方案共用相同的 Oracle 反转边界：标准、Query 全局、QK 分段、同门控加法对照。

```bash
python -m experiments.kaggle_reversal_eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --lengths 512 2048 8192 \
  --cases 8 \
  --output experiments/qwen25_reversal_results.json
```

每个 case 都是**不同的 Prompt**（实体名、填充文本相位、措辞都会变化），运行器会断言不存在重复 Prompt；重复样本只会带来空转计算。前向过程只计算最后一个位置的 logits。

在 Kaggle 上，`scripts/build_kaggle_kernel.py` 把两个模块打包成单文件 kernel（Kaggle 只上传 `code_file`）。smoke 阶段先验证 Transformers 前向契约并实测各长度的单次前向耗时，再按成本把完整的 `(长度, 场景, 样本)` 组贪心装箱到两张 T4；同一组的四种模式绝不拆卡，因此模式间配对比较始终在同一张卡上完成。`scripts/monitor_kaggle_e2.py` 负责取回输出，`scripts/summarize_kaggle_e2.py` 负责核对预注册判据。

每个样本只做**一次前向**：正确与过期两个候选答案取自同一个 next-token Softmax，因此报告的 margin 是成对对数概率差，而不是两次独立归一化后再相减。

主指标是正确答案相对过期答案的对数概率差、答案 NLL 和准确率。数据包含双重反转、无反转和无关实体反转，用来测量误伤。注意力对比度不再充当任务成功的替代指标。

首轮 0.5B 实验已记录在 [`experiments/qwen25_reversal_results.json`](experiments/qwen25_reversal_results.json)（384 次成对前向、双 T4、Oracle 边界），预注册判据核对见 [`experiments/qwen25_reversal_report.md`](experiments/qwen25_reversal_report.md)。

**结论：主假设未获支持。** QK 分段相位没有带来答案级收益（撤销场景相对标准 RoPE 为 −8.3 pp，且标准基线已接近满分，存在天花板效应）。同门控加法对照在单次撤销场景崩塌（0%），所以表面上的「旋转必要性 +41.7 pp」只是 `penalty=2.0` 未标定造成的假象，不能证明旋转是必要的；无关实体反转场景的误伤也超过 1 pp 预算。按 [停止条件](RESEARCH_PLAN.md)，本轮结果不支持进入 E3 门控训练，应先重标定加法对照或提高任务难度。

## 仓库结构

```text
src/phase_attention.py                  最小可微干预
experiments/run_controlled_math.py      确定性反证矩阵
experiments/kaggle_reversal_eval.py     预训练 Qwen2 评测
experiments/controlled_math_results.json
tests/test_phase_attention.py
tests/test_qwen_phase_eval.py
RESEARCH_PLAN.md                       假设、实验阶段和停止条件
legacy/                                旧 Agent 原型与交接材料
paper/                                 历史论文，不代表当前已验证
formal_verification/                   历史代数证明
```

旧 SG-Möbius 模块和论文完整性门禁仍保留用于追溯。它们的测试和证明应被理解为代数回归检查，不是“模型学会失效前提”或“下游准确率提升”的证据。重复原型和叙事性交接文档已经移入 `legacy/`，没有删除历史。

`experiments/` 下还保留 `kaggle_experiment_results.json`、`benchmark_results.json`、`kernel-metadata.json`、`data/` 和两张 `fig_*.png`。它们是冻结论文表格的输入，属于历史 SG-Möbius 材料，不是当前路径的结果。

## 相关真实论文

- [CoPE，arXiv:2405.18719](https://arxiv.org/abs/2405.18719)：Query-Key 对级的上下文位置；原论文尚未验证数十亿参数模型。
- [Round and Round We Go，arXiv:2410.06205](https://arxiv.org/abs/2410.06205)：RoPE 频率分工、语义漂移与 p-RoPE。
- [Differential Transformer，arXiv:2410.05258](https://arxiv.org/abs/2410.05258)：用两个 Softmax 注意力图之差抑制噪声，ICLR 2025 Oral。
- [Selective Attention，arXiv:2411.12892](https://arxiv.org/abs/2411.12892)：学习控制注意力稀疏度与相关性。

## 许可证

Apache-2.0 与 MulanPSL-2.0，详见现有许可证文件。
