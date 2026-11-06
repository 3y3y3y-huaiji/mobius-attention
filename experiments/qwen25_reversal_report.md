# E2 Oracle 边界实验结果（Qwen2.5-0.5B-Instruct）

- 记录数：**384**
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 设备：cuda (cost-balanced shards)，dtype：torch.float16
- transformers：5.20.2
- Oracle 边界：True；KV cache：False

## 运行元数据

- 提交：`c96191a`；kernel：`tinacherry/phase-reversal-e2`
- 随机种子：0；模型 revision：`7ae557604adf67be50417f59c2c2f167def9a775`
- GPU：Tesla T4, Tesla T4；torch：2.11.0+cu130
- 起止：2026-11-04T11:53:01.612373+00:00 → 2026-11-04T11:58:05.105794+00:00（303s）
- 命令：`python -m experiments.kaggle_reversal_eval --model Qwen/Qwen2.5-0.5B-Instruct --lengths 512 2048 8192 --cases 8 --modes standard query segment additive --output experiments/qwen25_reversal_results.json (executed as GPU shards)`
- 实测单次前向：L=2048: 291ms, L=512: 96ms, L=8192: 3708ms
- 双卡分片：各组数 [48, 48]，预计耗时 [262075.6, 262075.6] ms，绑定 ['CUDA_VISIBLE_DEVICES=0', 'CUDA_VISIBLE_DEVICES=1']

> 样本说明：Every case is a distinct prompt; no repeated forwards.

> 控制变量：A (length, scenario, case_index) group is never split: all four modes of a paired comparison run on the same GPU. Cross-length wall-clock comparisons may mix devices, so only mode-to-mode deltas are interpreted.

## 主结果

每个单元格为 `准确率 / 平均 margin（正确 − 过期 logprob）`。

### 目标长度 512

| 场景 | standard | query | segment | additive |
| --- | --- | --- | --- | --- |
| single | 1.000 / +3.7852 | 0.500 / -0.2886 | 0.500 / +0.1089 | 0.000 / -0.7617 |
| double | 1.000 / +1.5801 | 1.000 / +1.4629 | 1.000 / +1.5791 | 1.000 / +2.1387 |
| unrelated | 0.000 / -1.7197 | 0.625 / +0.5781 | 0.500 / +0.0366 | 1.000 / +4.5684 |
| none | 1.000 / +5.1787 | 1.000 / +5.1787 | 1.000 / +5.1787 | 1.000 / +5.1787 |

### 目标长度 2048

| 场景 | standard | query | segment | additive |
| --- | --- | --- | --- | --- |
| single | 1.000 / +3.2461 | 0.875 / +2.2375 | 1.000 / +2.7036 | 0.000 / -3.7617 |
| double | 1.000 / +1.9434 | 1.000 / +1.2227 | 1.000 / +1.6493 | 1.000 / +2.6270 |
| unrelated | 0.500 / -0.1426 | 0.000 / -2.1977 | 0.000 / -2.5942 | 1.000 / +7.0273 |
| none | 1.000 / +4.2988 | 1.000 / +4.2988 | 1.000 / +4.2988 | 1.000 / +4.2988 |

### 目标长度 8192

| 场景 | standard | query | segment | additive |
| --- | --- | --- | --- | --- |
| single | 1.000 / +3.4502 | 1.000 / +2.1733 | 1.000 / +2.4463 | 0.000 / -4.1855 |
| double | 1.000 / +2.0156 | 1.000 / +0.9062 | 1.000 / +1.8540 | 1.000 / +3.3682 |
| unrelated | 0.125 / -0.7051 | 0.125 / -2.0005 | 0.000 / -2.1333 | 1.000 / +7.4473 |
| none | 1.000 / +3.7480 | 1.000 / +3.7480 | 1.000 / +3.7480 | 1.000 / +3.7480 |

## 预注册判据

| 判据 | 阈值 | 实测 | 结论 |
| --- | --- | --- | --- |
| 有效性：segment − standard（撤销场景） | ≥ +5 pp | -8.33 pp | 未通过 |
| 旋转必要性：segment − additive（撤销场景） | ≥ +2 pp | +41.67 pp | 通过（对照崩塌，不可采信） |
| 低误伤：standard − segment（安全场景） | ≤ 1 pp | +2.08 pp | 未通过 |
| 工程价值：segment 相对 standard 额外延迟 | < 10% | -0.09% | 通过 |

## 结论

撤销场景 standard 准确率 1.000、segment 0.917、additive 0.500。

1. **主假设未获支持**：Oracle 边界下 QK 分段相位没有带来答案级收益（-8.33 pp）。standard 在撤销场景已接近满分，存在天花板效应，预注册的「+5 pp」判据在该任务上结构上不可满足；这不等于证明机制无效，但也不构成支持它的证据。
2. **「旋转必要性」的通过不可采信**：additive 在 single 场景准确率 0.000，penalty=2.0 的加法偏置过强且未与旋转做等强度标定；segment 只是「没那么糟」，不是优于一个公平的加法基线。
3. **误伤未达标的风险**：安全场景 standard − segment = +2.08 pp，unrelated（无关实体反转）下 segment 在长序列表现尤其差，符合「独立实体被误伤」的停止条件信号。
4. `none` 场景（无反转边界）四种模式数值完全一致，验证了零干预时 controller 与原模型等价。
5. 按 RESEARCH_PLAN 的停止条件（答案不改善、独立实体持续误伤），**不应据此进入 E3 门控训练**；下一步应先重标定加法对照或提高任务难度，再判断假设是否值得继续。

## 说明

- 指标为答案级对数概率，不是注意力对比度；正确与过期候选来自同一次前向的同一 Softmax。
- 门控为 Oracle 边界，未训练检测器；本结果不能证明模型能学会识别撤销。
- `query` 为全局 Query 相位对照，`additive` 为同边界加法偏置对照。
- 同一 (长度, 场景, 样本) 组的四种模式运行在同一张 GPU；跨长度的绝对耗时可能混合设备，因此只解释模式间差值。
