# 失效前提的选择性反相：最小研究计划

## 0. 当前结论

这个直觉**值得做，但必须先收窄命题**：

1. 旋转本身在数学和反向传播上合法，不会因为“转了 180 度”就天然破坏梯度。
2. 只给当前 Query 加旋转，会对它看到的所有历史 Key 同时生效，不能选择性处理某一个前提。
3. 当前仓库真正实现的 Q、K 累积相位提供了折中：同一段内不变，跨段反相；历史 K 写入后不变。这比全局 Query 旋转更有研究价值，但只有段级选择性。
4. 反相只保证所选子空间 logit 变号，不保证总 logit 变负，也不保证 Softmax 权重归零。Value 没有被抵消，因此不能称为字面意义上的“信息删除”。
5. 旧测试主要验证代数恒等式、自洽性和缓存行为。是否提高答案正确率、门控能否学习、无关前提是否被误伤，必须重新实验。

## 1. 三种机制必须分开

记标准 RoPE 注意力分数为：

$$
s_{ij} = \frac{q_i^\top R((j-i)\theta)k_j}{\sqrt d}
$$

### A. Query 全局反相

标准 RoPE 后，只给 Query 加相位：

$$
\tilde q_i = R(\phi_i)R(i\theta)q_i,\qquad \tilde k_j = R(j\theta)k_j
$$

于是：

$$
\tilde s_{ij} = \frac{q_i^\top R((j-i)\theta-\phi_i)k_j}{\sqrt d}
$$

这不是“没有 RoPE”，而是“RoPE 加了一个由 Query 决定的额外角度”。问题是对固定的 Query，所有历史 Key 都承受同一个额外角度，包括仍有效的新信息和无关实体。

### B. QK 累积相位分段反相

Q、K 都保留各自写入时的相位：

$$
\tilde q_i = R(\phi_i)R(i\theta)q_i,\qquad \tilde k_j = R(\phi_j)R(j\theta)k_j
$$

相对角度变为：

$$
(j-i)\theta + (\phi_j-\phi_i)
$$

- 同段：相位差为零，严格退化为标准 RoPE。
- 跨段：相位差为 pi 时，被选二维平面内的点积变号。
- 缓存：每个 Key 只在写入时旋转一次，之后无需修改。
- 局限：它表达的是“时间分段 / 状态版本”，不是任意 Query-Key 对的独立语义关系。

### C. 点对点抑制

直接构造每个 Query-Key 对的门控或偏置，例如 CoPE 的 $g_{ij}$。表达力最高，但需要 pairwise 状态、额外显存和更复杂的 kernel。它应作为性能上界或后期路线，不应成为第一版工程负担。

## 2. 反相不等于压制

对被选子空间 $S$，跨段 pi 反相后的 logit 是：

$$
\tilde s_{ij} = s_{ij} - 2s_{ij}^{S}
$$

只有 $s_{ij}^{S}>0$ 时，logit 才下降；若该分量原本为负，反相反而会抬高它。只旋转部分通道时，未旋转通道仍可让总 logit 保持正值。

此外：

- Softmax 对有限 logit 仍给出正权重；
- Value 向量没有变号或删除；
- 概率质量会重新分配给所有其他 token，不一定只分配给正确目标。

因此论文和图表应使用“logit 调制 / 相对重分配”，不能使用“物理抹除”“严格归零”“Value 相消”。

## 3. 训练与数值稳定性

### 固定或 Oracle 相位

旋转矩阵正交，范数和固定相位下的输入 Jacobian 范数保持不变。它本身不引入梯度爆炸；这是能由单元测试严格验证的部分。

### 学习相位

真正的风险在门控：

- 硬 0/pi 切换不可微；
- 过陡 sigmoid 容易饱和，门控长期学不到信号；
- 全局或段级误触发会同时改变大量 logit，造成优化干扰；
- “识别哪个前提被推翻”比“检测 however / 但是”困难得多，词表质心不能代表实体级逻辑关系。

第一版可令 $\phi_t=\pi\sigma(a_t/T)$，从较平滑温度和较弱正则开始，并记录门控饱和度、梯度范数、误触发率和实体级混淆矩阵。不要在第一轮引入 EMA、锁相吸引子、迟滞窗口、引号规则和并行扫描。

## 4. 工程可达性

| 方案 | 额外状态 | 对历史 K | 对标准 attention kernel | 研究定位 |
| --- | --- | --- | --- | --- |
| Query 全局相位 | 当前 Query 相位 | 不改 | 旋转后仍可调用 | 必做对照，预期天花板低 |
| QK 分段相位 | 每 token 一个或少量标量相位 | 写入后不改 | 旋转后仍可调用 | 当前主假设 |
| 加法分段偏置 | 同样段标记 | 不改 | 需要加 bias | 判断旋转是否必要的强基线 |
| 点对点语义门控 | Query-Key 对状态 | 取决于实现 | 通常需要改 kernel | 上界 / 后续路线 |

分段相位的额外旋转是逐 token、逐通道操作，理论开销为 $O(Ld)$，不需要保存 $L\times L$ 的相位矩阵。但实际 wall-clock 必须测量，不能由复杂度直接宣称“零开销”。

## 5. 实验矩阵

### E0：代数与反证测试（已完成）

文件：`src/phase_attention.py`、`tests/test_phase_attention.py`、`experiments/run_controlled_math.py`。

覆盖：

- 零相位严格退化；
- 同段严格退化；
- 跨段子空间变号；
- Query 全局误伤新信息；
- 负 logit 被反向放大；
- 部分通道不保证总 logit 为负；
- 固定相位反向传播有限且 Jacobian 保范；
- 全量 prefill 与单步读取等价。

这些结果只证明算子行为，不是模型效果。

### E1：合成任务小模型训练

目的：回答“能不能训出来”，而不是继续证明三角恒等式。

- 模型：2–4 层、hidden 128、4 heads，从头训练。
- 数据：程序生成的实体—状态—撤销任务，训练长度 128，测试长度 512 / 2048。
- 场景：单次撤销、恢复、两个独立实体、引号中的否定、无撤销、多个干扰项。
- 对照：标准 RoPE、Query 全局、QK 分段、同门控加法偏置、随机相位、Oracle 点对点偏置。
- 训练：至少 5 个随机种子；训练、验证、测试使用不同模板和实体词。
- 指标：最终状态准确率、答案 NLL、泛化长度、误触发率、梯度 p50/p99、loss spike、吞吐和显存。

### E2：预训练模型 Oracle 边界实验（已完成三代架构对比）

目的：在暂时不要求模型学门控的情况下，验证“如果有完美边界，旋转是否有用”，并检验不同 RoPE 架构对零样本相位翻转的敏感性。

- 模型与对照矩阵：
  - `Qwen/Qwen2.5-0.5B-Instruct`（24 层全注意力，`partial_rotary_factor = 1.0`，无 QK-Norm）：[`experiments/qwen25_reversal_results.json`](experiments/qwen25_reversal_results.json) / [`experiments/qwen25_reversal_report.md`](experiments/qwen25_reversal_report.md)
  - `Qwen/Qwen3-0.6B`（28 层全注意力，`partial_rotary_factor = 1.0`，含 QK-Norm）：[`experiments/qwen3_06b_reversal_results.json`](experiments/qwen3_06b_reversal_results.json) / [`experiments/qwen3_06b_reversal_report.md`](experiments/qwen3_06b_reversal_report.md)
  - `Qwen/Qwen3.5-0.8B`（6 层全注意力 + 18 层线性注意力 GatedDeltaNet，`partial_rotary_factor = 0.25`，含 QK-Norm 与输出门控）：[`experiments/qwen35_08b_reversal_results.json`](experiments/qwen35_08b_reversal_results.json) / [`experiments/qwen35_08b_reversal_report.md`](experiments/qwen35_08b_reversal_report.md)
- 方法：同一批提示词分别运行 `standard` / `query` / `segment` / `additive`（每个长度 32 个独立样本，长度 512、2,048、8,192，总计每模型 384 次前向）。
- 实证结论：
  1. **部分旋转子空间（25% RoPE + 75% 语义锚定）是零样本相位翻转不破坏底座的关键**：在 `partial_rotary_factor = 1.0` 的 `Qwen2.5-0.5B` 和 `Qwen3-0.6B` 上，翻转全部 100% 旋转通道会破坏基础语法与定位特征（`Qwen3-0.6B` 在撤销场景下跌 12.50 pp）；而在 `partial_rotary_factor = 0.25` 的 `Qwen3.5-0.8B` 上，仅前 64/256 维参与旋转、后 192 维保留为不旋转语义锚点，`segment` 将撤销场景准确率从 `0.500` 提升至 **`1.000`（+50.00 pp）**，其中 `double` 二次推翻从 `0.000` 恢复至 `1.000`，`single` 单次推翻的 Margin 全面扩大。
  2. **「只转 Query」被实验证伪**：`query` 模式在 `Qwen3-0.6B` 和 `Qwen3.5-0.8B` 的 `single` 场景准确率均为 `0.000`，因为只转 Query 会对前文所有 Key（无论旧前提还是新更正）一视同仁地反相。
  3. **一维全局时间轴相位必然误伤无关实体（`unrelated` 场景）**：在 `Qwen3.5-0.8B` 上，当无关实体 `BETA` 在后文反转时，全局相位阶梯同样把早期定义的 `ALPHA` 置于反相（准确率 `0.000`，Margin 从 `-0.39` 降至 `-1.42`）；同时同边界的 `additive` 加法对照在 `Qwen3.5-0.8B` 撤销场景同样达到 `1.000`（`segment - additive = 0.00 pp`）。
- 下一步前置条件（进入 E3 之前）：必须将一维全局时间相位升级为**实体绑定相位（Key-Bound Phase）**，并消除测试集中 `REVISION` 紧贴末尾的位置偏差，验证在保全无关实体 `unrelated` 的前提下旋转是否真正优于加法偏置。

### E3：学习门控与公平对照

- 先只训练门控，冻结主干；再做门控 + LoRA。
- Query、segment、additive 共用同一个门控和训练数据。
- 用实体级标签监督“哪个前提被撤销”，不能只监督转折词。
- 报告学习门控与 Oracle 门控的差距，定位瓶颈在检测器还是旋转机制。
- 至少 3 个随机种子，并保留所有失败运行。

### E4：扩大模型与真实长文任务

只有 E1–E3 通过后才启动 Qwen3.5-4B / 9B：

- 只修改全注意力层，保留其他层结构；
- 加入多实体状态追踪、长对话撤回、工具结果作废等任务；
- 用独立通用文本测困惑度和通用能力回归；
- 与 CoPE 式 pairwise 上界比较收益和成本，而不是宣称取代 CoPE。

## 6. 预注册判据

建议在运行前固定以下判据，避免事后挑图：

1. **有效性**：分段旋转在撤销任务上相对标准 RoPE 至少提高 5 个百分点，并且跨随机种子方向一致。
2. **旋转必要性**：相对同门控加法偏置至少提高 2 个百分点；否则优先采用更简单的加法机制。
3. **低误伤**：无撤销和无关实体场景准确率下降不超过 1 个百分点，通用文本 PPL 相对恶化不超过 1%。
4. **可训练性**：无非有限梯度，p99 梯度范数不超过基线的 10 倍，门控不长期塌缩到全 0 或全 1。
5. **工程价值**：同硬件同批量下额外延迟低于 10%，且收益随长度不完全消失。

停止条件：只改善注意力对比度而不改善答案；加法基线同等或更好；独立实体被持续误伤；学习门控在模板外失效。触及任一条件时先简化或否定假设，不继续增加形式化包装。

## 7. 数据与报告纪律

- 不挑选“最好的一层 / 一个头”作为主结果。
- 不用单个生成样例证明机制有效。
- 不把 attention entropy 或 ACR 当作答案正确率。
- 不混用训练模板与测试模板。
- 所有 JSON 标明模型版本、提交哈希、硬件、随机种子和运行命令。
- 失败结果同样保存；历史 `experiments/kaggle_experiment_results.json` 不视作当前独立复现。

## 8. 真实参考资料

- CoPE：Golovneva et al., *Contextual Position Encoding: Learning to Count What’s Important*, arXiv:2405.18719。其门控是 query-key 对级，论文明确尚未验证数十亿参数模型。
- p-RoPE：Barbero et al., *Round and Round We Go! What makes Rotary Positional Encodings useful?*, arXiv:2410.06205。低频语义通道与长上下文漂移是重要依据，但其实验不等于本假设已被验证。
- Differential Transformer：Ye et al., arXiv:2410.05258，ICLR 2025 Oral。它通过两个 Softmax 图之差抑制噪声，是“注意力相消”必须比较的相关工作。
- Selective Attention：Zhang et al., arXiv:2411.12892。说明调节注意力分布本身也是活跃方向，不能只比较 RoPE 变体。
