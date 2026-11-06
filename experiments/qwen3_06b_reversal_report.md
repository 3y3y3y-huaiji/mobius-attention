# E2 Oracle 边界实验结果（Qwen3-0.6B）

- 记录数：**384**
- 模型：`Qwen/Qwen3-0.6B`
- 设备：cuda (cost-balanced shards)，dtype：torch.float16
- transformers：5.20.2
- Oracle 边界：True；KV cache：False

## 运行元数据

- 提交：`cff9f5d`；kernel：`tinacherry/phase-reversal-e2`
- 随机种子：0；模型 revision：`c1899de289a04d12100db370d81485cdf75e47ca`
- GPU：Tesla T4, Tesla T4；torch：2.11.0+cu130
- 起止：2026-11-06T14:35:48.096838+00:00 → 2026-11-06T14:43:26.034569+00:00（458s）
- 命令：`python -m experiments.kaggle_reversal_eval --model Qwen/Qwen3-0.6B --lengths 512 2048 8192 --cases 8 --modes standard query segment additive --output experiments/qwen3_06b_reversal_results.json (executed as GPU shards)`
- 实测单次前向：L=2048: 458ms, L=512: 110ms, L=8192: 6297ms
- 双卡分片：各组数 [48, 48]，预计耗时 [439310.6, 439310.6] ms，绑定 ['CUDA_VISIBLE_DEVICES=0', 'CUDA_VISIBLE_DEVICES=1']

> 样本说明：Every case is a distinct prompt; no repeated forwards.

> 控制变量：A (length, scenario, case_index) group is never split: all four modes of a paired comparison run on the same GPU. Cross-length wall-clock comparisons may mix devices, so only mode-to-mode deltas are interpreted.

## 主结果

每个单元格为 `准确率 / 平均 margin（正确 − 过期 logprob）`。

### 目标长度 512

| 场景 | standard | query | segment | additive |
| --- | --- | --- | --- | --- |
| single | 1.000 / +3.8730 | 0.000 / -0.6440 | 0.375 / -0.6553 | 0.375 / -0.2285 |
| double | 0.000 / -2.1797 | 0.000 / -0.7598 | 0.750 / +0.7275 | 0.750 / +0.2695 |
| unrelated | 0.000 / -1.5742 | 1.000 / +0.8814 | 0.750 / +0.7441 | 0.625 / +0.8867 |
| none | 1.000 / +1.3438 | 1.000 / +1.3438 | 1.000 / +1.3438 | 1.000 / +1.3438 |

### 目标长度 2048

| 场景 | standard | query | segment | additive |
| --- | --- | --- | --- | --- |
| single | 1.000 / +3.3809 | 0.000 / -1.1094 | 0.500 / -0.3403 | 0.875 / +2.1719 |
| double | 0.000 / -1.7207 | 0.875 / +0.9023 | 0.500 / +0.1162 | 0.625 / +0.2598 |
| unrelated | 0.000 / -1.2461 | 1.000 / +1.2373 | 0.625 / +0.5313 | 0.625 / -0.1426 |
| none | 1.000 / +1.5273 | 1.000 / +1.5273 | 1.000 / +1.5273 | 1.000 / +1.5273 |

### 目标长度 8192

| 场景 | standard | query | segment | additive |
| --- | --- | --- | --- | --- |
| single | 1.000 / +3.0723 | 0.000 / -1.2720 | 0.000 / -1.4038 | 1.000 / +4.3525 |
| double | 0.000 / -1.2637 | 1.000 / +1.3672 | 0.125 / -2.5669 | 1.000 / +1.1289 |
| unrelated | 0.000 / -1.3770 | 1.000 / +1.2832 | 0.750 / +0.8140 | 0.125 / -1.7451 |
| none | 0.750 / +1.1699 | 0.750 / +1.1699 | 0.750 / +1.1699 | 0.750 / +1.1699 |

## 预注册判据

| 判据 | 阈值 | 实测 | 结论 |
| --- | --- | --- | --- |
| 有效性：segment − standard（撤销场景） | ≥ +5 pp | -12.50 pp | 未通过 |
| 旋转必要性：segment − additive（撤销场景） | ≥ +2 pp | -39.58 pp | 未通过 |
| 低误伤：standard − segment（安全场景） | ≤ 1 pp | -35.42 pp | 通过 |
| 工程价值：segment 相对 standard 额外延迟 | < 10% | +2.02% | 通过 |

## 结论

撤销场景 standard 准确率 0.500、segment 0.375、additive 0.771。

1. **主假设未获支持**：Oracle 边界下 QK 分段相位没有带来答案级收益（-12.50 pp）。未达到预注册的 +5 pp 门槛。
2. 旋转必要性按预注册阈值判定，additive 对照未出现崩塌。
3. **误伤未达标的风险**：安全场景 standard − segment = -35.42 pp，unrelated（无关实体反转）下 segment 在长序列表现尤其差，符合「独立实体被误伤」的停止条件信号。
4. `none` 场景（无反转边界）四种模式数值完全一致，验证了零干预时 controller 与原模型等价。
5. 按 RESEARCH_PLAN 的停止条件（答案不改善、独立实体持续误伤），**不应据此进入 E3 门控训练**；下一步应先重标定加法对照或提高任务难度，再判断假设是否值得继续。

## 说明

- 指标为答案级对数概率，不是注意力对比度；正确与过期候选来自同一次前向的同一 Softmax。
- 门控为 Oracle 边界，未训练检测器；本结果不能证明模型能学会识别撤销。
- `query` 为全局 Query 相位对照，`additive` 为同边界加法偏置对照。
- 同一 (长度, 场景, 样本) 组的四种模式运行在同一张 GPU；跨长度的绝对耗时可能混合设备，因此只解释模式间差值。
