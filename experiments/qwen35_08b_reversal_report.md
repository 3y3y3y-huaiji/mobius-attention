# E2 Oracle 边界实验结果（Qwen3.5-0.8B）

- 记录数：**384**
- 模型：`Qwen/Qwen3.5-0.8B`
- 设备：cuda (cost-balanced shards)，dtype：torch.float16
- transformers：5.20.2
- Oracle 边界：True；KV cache：False

## 运行元数据

- 提交：`cff9f5d`；kernel：`tinacherry/phase-reversal-e2`
- 随机种子：0；模型 revision：`2fc06364715b967f1860aea9cf38778875588b17`
- GPU：Tesla T4, Tesla T4；torch：2.11.0+cu130
- 起止：2026-11-06T14:44:33.555808+00:00 → 2026-11-06T14:48:05.051359+00:00（211s）
- 命令：`python -m experiments.kaggle_reversal_eval --model Qwen/Qwen3.5-0.8B --lengths 512 2048 8192 --cases 8 --modes standard query segment additive --output experiments/qwen35_08b_reversal_results.json (executed as GPU shards)`
- 实测单次前向：L=2048: 435ms, L=512: 307ms, L=8192: 2560ms
- 双卡分片：各组数 [48, 48]，预计耗时 [211333.5, 211333.5] ms，绑定 ['CUDA_VISIBLE_DEVICES=0', 'CUDA_VISIBLE_DEVICES=1']

> 样本说明：Every case is a distinct prompt; no repeated forwards.

> 控制变量：A (length, scenario, case_index) group is never split: all four modes of a paired comparison run on the same GPU. Cross-length wall-clock comparisons may mix devices, so only mode-to-mode deltas are interpreted.

## 主结果

每个单元格为 `准确率 / 平均 margin（正确 − 过期 logprob）`。

### 目标长度 512

| 场景 | standard | query | segment | additive |
| --- | --- | --- | --- | --- |
| single | 1.000 / +1.2412 | 0.000 / -1.0625 | 1.000 / +1.6699 | 1.000 / +2.0283 |
| double | 0.000 / -0.4062 | 0.875 / +0.1152 | 1.000 / +0.5889 | 1.000 / +0.2861 |
| unrelated | 0.000 / -0.2803 | 1.000 / +1.1406 | 0.000 / -1.2686 | 0.000 / -1.2676 |
| none | 1.000 / +0.5771 | 1.000 / +0.5771 | 1.000 / +0.5771 | 1.000 / +0.5771 |

### 目标长度 2048

| 场景 | standard | query | segment | additive |
| --- | --- | --- | --- | --- |
| single | 1.000 / +1.4062 | 0.000 / -1.1567 | 1.000 / +1.8994 | 1.000 / +2.2041 |
| double | 0.000 / -0.5176 | 0.500 / +0.0088 | 1.000 / +0.6260 | 1.000 / +0.2803 |
| unrelated | 0.125 / -0.4199 | 1.000 / +1.1660 | 0.000 / -1.4658 | 0.000 / -1.3145 |
| none | 1.000 / +0.4385 | 1.000 / +0.4385 | 1.000 / +0.4385 | 1.000 / +0.4385 |

### 目标长度 8192

| 场景 | standard | query | segment | additive |
| --- | --- | --- | --- | --- |
| single | 1.000 / +1.4756 | 0.000 / -0.9170 | 1.000 / +1.8828 | 1.000 / +2.2793 |
| double | 0.000 / -0.5088 | 0.500 / -0.0107 | 1.000 / +0.7490 | 1.000 / +0.3242 |
| unrelated | 0.000 / -0.4775 | 1.000 / +0.8086 | 0.000 / -1.5264 | 0.000 / -1.2812 |
| none | 1.000 / +0.4170 | 1.000 / +0.4170 | 1.000 / +0.4170 | 1.000 / +0.4170 |

## 预注册判据

| 判据 | 阈值 | 实测 | 结论 |
| --- | --- | --- | --- |
| 有效性：segment − standard（撤销场景） | ≥ +5 pp | +50.00 pp | 通过 |
| 旋转必要性：segment − additive（撤销场景） | ≥ +2 pp | +0.00 pp | 未通过 |
| 低误伤：standard − segment（安全场景） | ≤ 1 pp | +2.08 pp | 未通过 |
| 工程价值：segment 相对 standard 额外延迟 | < 10% | -4.57% | 通过 |

## 结论

撤销场景 standard 准确率 0.500、segment 1.000、additive 1.000。

1. **有效性判据通过（+50.00 pp）**：Oracle 边界下 QK 分段相位将撤销场景准确率从 0.500 提升至 1.000（主要来自 `double` 二次反转从 0.000 恢复并加强，`single` 的正 margin 也进一步扩大）。
2. 旋转必要性按预注册阈值判定，additive 对照未出现崩塌。
3. **误伤未达标的风险**：安全场景 standard − segment = +2.08 pp，unrelated（无关实体反转）下 segment 在长序列表现尤其差，符合「独立实体被误伤」的停止条件信号。
4. `none` 场景（无反转边界）四种模式数值完全一致，验证了零干预时 controller 与原模型等价。
5. 按 RESEARCH_PLAN 的停止条件（答案不改善、独立实体持续误伤），**不应据此进入 E3 门控训练**；下一步应先重标定加法对照或提高任务难度，再判断假设是否值得继续。

## 说明

- 指标为答案级对数概率，不是注意力对比度；正确与过期候选来自同一次前向的同一 Softmax。
- 门控为 Oracle 边界，未训练检测器；本结果不能证明模型能学会识别撤销。
- `query` 为全局 Query 相位对照，`additive` 为同边界加法偏置对照。
- 同一 (长度, 场景, 样本) 组的四种模式运行在同一张 GPU；跨长度的绝对耗时可能混合设备，因此只解释模式间差值。
