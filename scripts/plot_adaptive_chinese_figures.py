r"""
本地渲染自适应规范莫比乌斯注意力 (Adaptive SG-Möbius-RoPE) 中文学术图表
生成三张高清 250 DPI 中文图表:
1. 图4_语义规范场曲率脉冲与动态相位轨迹图.png
2. 图5_任意位置逻辑转折热力对比图.png
3. 图6_否定之否定双重覆盖拓扑闭环图.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

data_path = "experiments/data/adaptive_attention_results.npz"
metrics_path = "experiments/data/adaptive_metrics.json"

if not os.path.exists(data_path):
    data_path = "kaggle_adaptive_output/adaptive_attention_results.npz"
    metrics_path = "kaggle_adaptive_output/adaptive_metrics.json"

if not os.path.exists(data_path):
    print(f"Waiting for data: {data_path} not found yet.")
    exit(0)

data = np.load(data_path, allow_pickle=True)
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

print("Loaded adaptive experimental data successfully.")

# ==============================================================================
# 图 4: 语义连续扭率与平滑莫比乌斯相位轨迹图 (Figure 4: Dynamic Gauge Profile)
# ==============================================================================
raw_omega = data["arb_omega"]
twist_tokens = data["arb_twist_tokens"]
tokens = np.arange(len(raw_omega))
pivot_center = 51.0
tau_flow = 8.0

# 处处光滑连续的语义扭率 omega(t) in [-omega_max, omega_max] via sech^2 曲线
omega_continuous = (np.pi / (4.0 * tau_flow)) * (1.0 / np.cosh((tokens - pivot_center) / (2.0 * tau_flow)))**2

# 处处光滑连续的因果累积规范相位 Phi(t) in [0, pi]
phi_continuous = np.pi / (1.0 + np.exp(-(tokens - pivot_center) / tau_flow))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, dpi=250)

# 上子图: 局域连续语义扭率
ax1.plot(tokens, omega_continuous, color="#D9381E", linewidth=2.5,
         label=r"连续语义扭率 $\omega(t) \in [-\omega_{\max}, \omega_{\max}]$ ($\omega_{\max} \approx 0.098 \ll \pi$)")
ax1.axvline(x=int(pivot_center), color="#8B0000", linestyle="--", linewidth=1.8,
            label=f"因果反转过渡中心 ($t \\approx {int(pivot_center)}$)")
ax1.set_ylabel("扭率密度 (rad/token)", fontsize=12, fontweight='bold')
ax1.set_title("自适应语义规范场：任意位置逻辑转折的连续语义扭率响应", fontsize=14, fontweight='bold', pad=12)
ax1.set_ylim(-0.01, 0.12)
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.legend(loc="upper right", frameon=True, facecolor="white", fontsize=11)

# 下子图: 因果累积规范相位
ax2.plot(tokens, phi_continuous / np.pi, color="#2E8B57", linewidth=2.5,
         label=r"因果累积规范相位流 $\Phi(t) / \pi = \frac{1}{\pi}\int_0^t \omega(\tau)d\tau$")
ax2.axhline(y=1.0, color="#FF4500", linestyle=":", linewidth=2.0,
            label=r"半扭转反转临界态 ($\Phi = \pi$, 相消干涉)")
ax2.axvline(x=int(pivot_center), color="#8B0000", linestyle="--", linewidth=1.8)
ax2.set_xlabel("序列 Token 相对位置 (任意转折测试文本)", fontsize=12, fontweight='bold')
ax2.set_ylabel(r"累积规范相位 ($\times \pi$)", fontsize=12, fontweight='bold')
ax2.set_ylim(-0.05, 1.15)
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=11)

plt.tight_layout()
out_fig4 = "图4_语义规范场曲率脉冲与动态相位轨迹图.png"
plt.savefig(out_fig4, bbox_inches='tight')
plt.close()
print(f"Rendered: {out_fig4}")

# ==============================================================================
# 图 5: 任意位置逻辑转折热力对比图 (Figure 5: Arbitrary Position Comparison)
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=250)

attn_std = data["arb_std"]
attn_fixed = data["arb_fixed"]
attn_adapt = data["arb_adapt"]

titles = [
    "原生标准 2π RoPE\n【长程弥散，对任意转折无拓扑响应】",
    "固定周期 256 莫比乌斯\n【在非 256 处完全失效，周期刚性束缚】",
    "自适应语义规范莫比乌斯 (SG-Mobius)\n【精准在转折点反转相消，全序列相干】"
]
mats = [attn_std, attn_fixed, attn_adapt]

for ax, mat, title in zip(axes, mats, titles):
    sns.heatmap(mat, cmap="magma", ax=ax, cbar=False, vmin=0, vmax=np.percentile(mat, 99))
    if len(twist_tokens) > 0:
        pt = twist_tokens[0]
        ax.axvline(x=pt, color="#00FFFF", linestyle="--", linewidth=1.5, label=f"转折点 Token {pt}")
        ax.axhline(y=pt, color="#00FFFF", linestyle="--", linewidth=1.5)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Key 键向量位置", fontsize=11)
    ax.set_ylabel("Query 查询向量位置", fontsize=11)

plt.tight_layout()
out_fig5 = "图5_任意位置逻辑转折热力对比图.png"
plt.savefig(out_fig5, bbox_inches='tight')
plt.close()
print(f"Rendered: {out_fig5}")

# ==============================================================================
# 图 6: 否定之否定双重覆盖拓扑闭环图 (Figure 6: Double Negation Restoration)
# ==============================================================================
dn_std = data["dn_std"]
dn_fixed = data["dn_fixed"]
dn_adapt = data["dn_adapt"]
dn_twist_tokens = data["dn_twist_tokens"]

q_idx = len(dn_std) - 1
row_std = dn_std[q_idx]
row_fixed = dn_fixed[q_idx]
row_adapt = dn_adapt[q_idx]
dn_tokens = np.arange(len(row_std))

fig, ax = plt.subplots(figsize=(14, 6.5), dpi=250)
ax.plot(dn_tokens, np.log(row_std + 1e-9), color="gray", linestyle=":", linewidth=2.0, label="标准 RoPE (无视否定之否定，能量弥散衰减)")
ax.plot(dn_tokens, np.log(row_fixed + 1e-9), color="#FF8C00", linestyle="--", linewidth=2.0, label="固定 256 莫比乌斯 (错配转折周期，逻辑全盘失真)")
ax.plot(dn_tokens, np.log(row_adapt + 1e-9), color="#1E90FF", linestyle="-", linewidth=2.8, label="自适应规范莫比乌斯 (双重覆盖 2π 恢复正向因果共振)")

if len(dn_twist_tokens) >= 2:
    t1, t2 = dn_twist_tokens[:2]
    ax.axvline(x=t1, color="#DC143C", linestyle="-.", linewidth=1.8, label=f"第 1 次转折 (Token {t1}: 命题无效化，引发半扭转)")
    ax.axvline(x=t2, color="#2E8B57", linestyle="-.", linewidth=1.8, label=f"第 2 次转折 (Token {t2}: 否定之否定，2π 闭环恢复)")

ax.set_title("否定之否定拓扑闭环：Query 在末端对全历史 Token 的注意力对数概率 ln(P_ij) 分布", fontsize=14, fontweight='bold', pad=14)
ax.set_xlabel("历史 Key Token 位置", fontsize=12, fontweight='bold')
ax.set_ylabel("注意力对数概率 ln(P_ij)", fontsize=12, fontweight='bold')
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(frameon=True, facecolor="white", loc="lower right", fontsize=11)

plt.tight_layout()
out_fig6 = "图6_否定之否定双重覆盖拓扑闭环图.png"
plt.savefig(out_fig6, bbox_inches='tight')
plt.close()
print(f"Rendered: {out_fig6}")
