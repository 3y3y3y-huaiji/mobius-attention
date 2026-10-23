# 变更交接文档：多轮反转统一配分函数收缩理论（Theorem 7 / Corollary 8）

> **交接对象**：接手修改本仓库的后续 AI
> **变更日期**：2026-10-22 ~ 2026-10-23
> **状态**：已完成并通过全量验证（六步流水线全绿），**未提交 git**（工作区 10 改 + 2 新增）
> **必读前置**：`AGENTS.md`（红线与验证流水线）、`AI_HANDOVER.md`（项目总交接规范）
> **规格文档**：`.trae/specs/prove-multi-turn-partition-contraction/`（spec.md / tasks.md / checklist.md，14 项检查点全部核验通过）

---

## 1. 本轮变更的数学内容（唯一真值源：`paper/main.tex` L399–L455）

把核心理论 Corollary 4（`cor:partition_siphoning`，单次语篇反转的配分函数收缩）推广到 **K ≤ 10 次反转**的奇偶交替统一模型（与 Proposition 6 的 M_Φ ≤ 10π 域一致），新增两个结果（共享 theorem 计数器，编号恰为 **7、8**，位于 Proposition 5/6 之后，故既有编号零漂移）：

### Theorem 7（`thm:multi_turn_contraction`，多轮奇偶交替配分函数收缩）
设反转位置 t_1 < ⋯ < t_K ≤ i，R(l) ≜ #{r : t_r ≤ l}，段 S_s = [t_s, t_{s+1})（t_0 = 0，t_{K+1} = i+1），则 ΔΦ_{l→i} ≡ (K−s)π (mod 2π)：
- **(i) 名义**：(K−s) 偶 → 同相复活 Ã=A^std；奇 → 反相作废 Ã=A^std−2A^mob。作废索引集 I_inv ≜ {l : K−R(l) 奇}，
  $$\tilde{Z}_i = Z_i^{\mathrm{std}} - \Delta Z_{\mathrm{tot}},\quad \Delta Z_{\mathrm{tot}} = \sum_{l\in\mathcal{I}_{\mathrm{inv}}} \exp(A_{il}^{\mathrm{std}})\big(1-\exp(-2A_{il}^{\mathrm{mob}})\big)$$
  A^mob > 0 时严格收缩；复活段贡献与标准质量严格相等；**K=1 逐项退化为 Corollary 4**。
- **(ii) 扰动**（ΔΦ=(K−s)π±δ_l，|δ_l| ≤ π/6）：反相段 Ã=A^std−(1+cos δ)A^mob∓sin δ·B^mob；同相段 Ã=A^std−(1−cos δ)A^mob±sin δ·B^mob（二阶泄漏 1−cos δ ≤ π²/72）；收缩充分条件 (1+cos δ)A^mob > |sin δ·B^mob| 逐点成立且反相收缩总量超过同相泄漏；1+cos δ ≥ 1+√3/2 ≈ 1.866。

### Corollary 8（`cor:k_turn_acr`，同相均匀放大、诱饵缓冲与 ACR 复合）
1. **同相均匀放大**：∀k ∉ I_inv，P_mob/P_std = T ≜ 1/(1−ΔZ_tot/Z^std) > 1，与 token 身份无关；
2. **诱饵缓冲**：反相 token j，P_mob(j)/P_std(j) = exp(−2A_j^mob)·T —— 表述必须是"**配分函数收缩降低全局分母，从而部分缓冲诱饵 logit 衰减带来的概率降幅**"；
3. **ACR 复合恒等式**：exp(2A_j^mob) = T × D_j^cushion（D^cushion ≜ 1/(exp(−2A^mob)·T)），单轮实测 T=3.49、D=1.62、乘积 5.65（+465.1%，3.62→20.44）；
4. **奇偶交替诚实披露**：新增一次反转使各段奇偶翻转（作废⇄复活），抑制量随 K **不具单调性**。

---

## 2. 变更文件清单（git 已核实：10 改 + 2 新增，+381/−48 行）

| 文件 | 变更 | 关键位置 |
| :--- | :--- | :--- |
| `paper/main.tex` | +58 行：新增 `\subsection{Multi-Turn Generalization of Partition Contraction}`（Theorem 7 + Corollary 8 + 完整证明）；Corollary 4 末尾 +1 句前向引用 | 新小节 L399–L455；前向引用 L302 |
| `paper/main_zh.tex` | +58 行：中文版逐符号同步（`\subsection{配分函数收缩的多轮推广}`）；推论 4 末尾 +1 句前向引用 | 新小节 L392–L447；前向引用 L298 |
| `paper/PAPER_ZH.md` | +30 行：新增 `### 3.4 配分函数收缩的多轮推广（定理 7 / 推论 8 摘要）` | L315–L341 |
| `formal_verification/MobiusAttention.lean` | +135 行：末尾新增 **Section 8**（L690–L823），8 个实数代数定理 | 见 §3 |
| `tests/test_multi_turn_contraction.py` | **新建** 252 行，7 个测试函数 / 46 用例 | 见 §4 |
| `README.md` / `README_zh.md` | +45 行各：理论章节新增 Theorem 7 / Corollary 8 摘要；SSOT 元数据（徽章、测试数、定理数、耗时） | 摘要段 README L209–L231、README_zh L203–L225 |
| `AGENTS.md` | 元数据：453 passed、17 theorems、main.tex 578 lines，并按零幻觉红线修正 2 处历史失实行数 | §1 Prime Directives、§2 |
| `AI_HANDOVER.md` | 元数据同步 + §6 对账表追加本轮定论行 | 全文 |
| `PROJECT.md` / `.github/copilot-instructions.md` / `.cursorrules` | 元数据同步（453 tests / 17 theorems / 真实行数） | — |
| `scripts/verify_formula_with_sympy.py` | 新增 `[6/6]` Theorem 7 & Corollary 8 符号代数自动验证 | L152–L192 |

## 3. Lean 4 新增定理（Section 8，docstring 标注论文编号，0 `sorry`、0 自定义公理）

| 定理名 | 行号 | 对应 |
| :--- | :--- | :--- |
| `multi_turn_parity_cos_phase` | L702 | Thm 7(i)：cos(nπ)=(−1)ⁿ |
| `multi_turn_parity_sign_flip` | L712 | Thm 7(i)：Vec2 旋转内积符号随奇偶翻转 |
| `multi_turn_voided_term_identity` | L728 | Thm 7(i)：作废项恒等式 |
| `multi_turn_partition_contraction` | L740 | Thm 7(i)：单项贡献 exp(a)(1−e^{−2m})>0 |
| `multi_turn_partition_contraction_sum` | L754 | Thm 7(i)：Finset 求和版 Z̃=Z−ΔZ<Z |
| `uniform_in_phase_amplification` | L773 | Cor 8(1)：均匀放大恒等式 |
| `uniform_in_phase_amplification_factor` | L790 | Cor 8(1)：∃T>1 与 token 无关 |
| `one_add_cos_lower_bound` | L809 | Thm 7(ii)：1+cos δ ≥ 1+√3/2 |

注意：Lean 文件原有内部编号 "Theorem 7: Epsilon-Perturbation Robustness"（L637）系旧编号，**未改动**；新章节 docstring 明确标注 "paper Theorem 7 (thm:multi_turn_contraction)" 以示区分。文档计数口径："9 theorems → 17 theorems"（9 经典核心 + 8 新增）。

## 4. 新增测试（`tests/test_multi_turn_contraction.py`）

7 个函数 / 46 用例（`py -m pytest tests/test_multi_turn_contraction.py -q` → 46 passed）：
`test_closed_form_partition_contraction`（K=1..10 闭式 vs 直接计算，容差 <1e-10）、`test_parity_alternating_segment_partition_k3`、`test_uniform_in_phase_amplification`、`test_acr_compounding_identity`（与 K 无关）、`test_perturbed_sufficient_condition`、`test_perturbed_boundary_counterexample`（边界反例实测 Z̃≥Z^std）、`test_k1_regression_matches_corollary4`。

## 5. SSOT 元数据变更表（修改其他内容时必须保持全库一致）

| 事实 | 旧值 → 新值 | 出现位置 |
| :--- | :--- | :--- |
| 测试总数 | 407 → **453**（407 + 46） | README、README_zh（含徽章）、PROJECT、AI_HANDOVER、AGENTS、copilot-instructions |
| Lean 定理数 | 9 → **17** | 同上 |
| `lake build` jobs | **1946（不变）** | AGENTS、AI_HANDOVER、PROJECT |
| `paper/main.tex` | 521 → **578** 行 | AGENTS、AI_HANDOVER、PROJECT |
| `paper/main_zh.tex` | 506 → **563** 行 | AI_HANDOVER、PROJECT |
| `src/gauge_predictor.py` | 612 → **615** 行（历史失实值的零幻觉修正） | AGENTS、AI_HANDOVER |
| `tests/test_gauge_gate_definition1.py` | 97 → **138** 行（同上） | AGENTS |
| 全量测试耗时 | ~25s → **~37s**（实测 453 passed in 36.66s） | README、README_zh、AI_HANDOVER、PROJECT |

SSOT 实验数字**严禁改动**：L ∈ [545, 18030]、ACR +465.1%（3.62→20.44）、诱饵 1.62×、目标 3.49×、困惑度 7.48（0.0% 恶化）。

## 6. 修改时必须遵守的红线（详见 AGENTS.md §4）

1. **零幻觉**：引用行号/函数/文件必须以本地实测为准（行数见 §5）。
2. **不得改动既有定理编号与正文**：Corollary 4、Proposition 5、Proposition 6 逐字未动（git diff 佐证：main.tex/main_zh.tex 均 58 插入 / 0 删除）；新结果编号恰为 7、8。若你要在论文中**再插入**新定理，必须放在 Theorem 7/Corollary 8 **之后**（编号顺延 9、10…），否则会挤掉 7/8 的编号。
3. **修辞红线（红线 3）**：禁止 "strictly monotonic suppression"/"单调递增" 类表述；压制只能说 "strictly above-unity suppression"；K 依赖只有奇偶交替语义（允许写 "not monotonic in K / 不具单调性" 的诚实披露）；缓冲机制只能写"配分函数收缩降低全局分母、缓冲诱饵概率降幅"，禁止"抬高诱饵分母"。
4. **Definition 1 门控红线（红线 2）**：死区门控必须在未缩放投影 z_t 上做 ReLU(|z|−τ)，禁止后置。
5. **Lean 红线（红线 5）**：0 `sorry`、0 自定义公理、`lake build` 100% 通过。
6. **Proposition 编号红线**：Proposition 5 恒为 KV-Cache 解码等价，Proposition 6 恒为 Higham FP16 界；误差界前提（K ≤ 10、M_Φ ≤ 10π、L ≤ 10⁶ 漂移 <10⁻⁵ rad）不得放宽措辞。

## 7. 修改后必跑的六步验证流水线（预期输出见括号）

```powershell
# 注意：本机 `python` 不在 PATH，用 `py` 启动器；lake 需经 elan 工具链调用（见下）
py -m pytest tests/ -n auto -q                                  # 453 passed in ~37s
py scripts/verify_paper_claims.py                               # PASSED
py scripts/check_paper_integrity.py paper/main.tex               # PASSED
py scripts/check_paper_integrity.py paper/main_zh.tex            # PASSED
py scripts/check_markdown_math.py README.md                      # 0 corrupted
py scripts/check_markdown_math.py README_zh.md                   # 0 corrupted
py -m pytest tests/test_readme_latex.py -q                       # 14 passed
cd formal_verification; lake build; cd ..                        # Build completed successfully (1946 jobs)
                                                                # （PATH 无 lake 时用：
                                                                #  & "$env:USERPROFILE\.elan\toolchains\leanprover--lean4---v4.33.1\bin\lake.exe" build）
py -m ruff check src/ tests/ scripts/                            # All checks passed!
```

六步全部 0 错误才算闭环。若你新增/删除测试，需按 §5 更新全库测试总数；若增删 Lean 定理，需同步定理数。

## 8. 修改入口建议

- 改理论：先改 `paper/main.tex`（真值源），再同步 `main_zh.tex`（公式逐符号）、Lean（实数代数层面）、测试（数值断言）、README 双语摘要。
- 改实验数字：**先取得 `experiments/kaggle_experiment_results.json` 真值**，再走 `verify_paper_claims.py` 门禁（它会核对 Table/Caption 与 JSON 一致性）。
- 未竟研究方向（`AI_HANDOVER.md` §7.2 + `paper/main.tex` L514 未来工作）：FlashAttention-2/Triton 算子融合、RULER 多跳反转评测、可微相位投影学习、2D ViT/视频多维推广；本轮 Theorem 7/Corollary 8 的 K 轮理论可直接支撑多轮反转实验设计。
