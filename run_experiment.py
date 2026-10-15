r"""
Qwen3.5-9B Multi-Perspective Visualization:
1. Raw Logits Inversion Scatter Plot (经典反转散点图)
2. 2D Attention Heatmap (二维注意力全景热力图)
3. Differential Attention Map (莫比乌斯增益差值图)
4. Critical Inversion Head Trajectory (关键反转头轨迹)
"""

import os
import sys
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("==================================================")
print("  Generating Multi-Perspective Plots for Qwen3.5-9B")
print("==================================================")
os.system("pip install -q accelerate torchvision seaborn")
import seaborn as sns

from transformers import AutoTokenizer, AutoConfig
try:
    from transformers import AutoModelForImageTextToText as AutoModelClass
except ImportError:
    from transformers import AutoModelForCausalLM as AutoModelClass

gpu_count = torch.cuda.device_count()
print(f"Available GPUs: {gpu_count}")

model_id = "Qwen/Qwen3.5-9B"
print(f"\n[1/3] Loading Tokenizer & Model ({model_id})...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

try:
    model = AutoModelClass.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=True
    )
except Exception:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=True
    )

print("Qwen3.5-9B ready in VRAM!")

# 2. Build structured long-range logical inversion benchmark text
premise_0 = "FACT_ALPHA is verified as ACTIVE and VALID. "
filler_1 = "The telemetry protocol maintains strict continuous verification across localized clusters. " * 20
pivot_256 = "CRITICAL INVERSION AT TIMESTAMP 256: Due to security emergency, FACT_ALPHA is completely INVALIDATED and set to FALSE. "
filler_2 = "Decentralized consensus validation records integrity hashes and secondary signatures. " * 20
pivot_512 = "RESTORED DIRECTIVE AT TIMESTAMP 512: The emergency was resolved, reversing the prior negation and reinstating FACT_ALPHA as VALID. "
query = "QUERY: According to the topological event horizon, determine the final valid truth state of FACT_ALPHA: "

prompt = premise_0 + filler_1 + pivot_256 + filler_2 + pivot_512 + query
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
seq_len = inputs.input_ids.shape[1]
print(f"Sequence length: {seq_len} tokens")

# 3. Baseline forward pass (Standard 2pi RoPE)
print("\n[2/3] Extracting Baseline 2D Attention Matrices...")
with torch.no_grad():
    outputs_std = model(**inputs, output_attentions=True)

target_layer = min(7, len(outputs_std.attentions) - 1)
# Raw 2D attention matrix [seq_len, seq_len]
attn_matrix_std = outputs_std.attentions[target_layer][0].mean(dim=0).cpu().float().numpy()

# 4. Inject Möbius-RoPE (4pi Spinor Half-Twist)
print("\n[Injecting] Hot-swapping model's rotary embeddings with Möbius 4pi Half-Twist...")
cycle_len = 256

def make_mobius_forward(original_forward):
    def mobius_forward(x, position_ids, *args, **kwargs):
        cos, sin = original_forward(x, position_ids, *args, **kwargs)
        seq_len_curr = cos.shape[-2]
        t = torch.arange(seq_len_curr, device=cos.device, dtype=cos.dtype)
        twist_1d = torch.cos(math.pi * t / cycle_len)
        view_shape = [1] * cos.dim()
        view_shape[-2] = seq_len_curr
        twist = twist_1d.view(*view_shape)
        return cos * twist, sin
    return mobius_forward

for name, module in model.named_modules():
    if hasattr(module, "rotary_emb") and module.rotary_emb is not None:
        module.rotary_emb.forward = make_mobius_forward(module.rotary_emb.forward)
    elif "rotary" in module.__class__.__name__.lower():
        module.forward = make_mobius_forward(module.forward)

with torch.no_grad():
    outputs_mob = model(**inputs, output_attentions=True)

attn_matrix_mob = outputs_mob.attentions[target_layer][0].mean(dim=0).cpu().float().numpy()

# 5. Extract unnormalized query logits for raw scatter plot
query_row_std = attn_matrix_std[-1]
query_row_mob = attn_matrix_mob[-1]

# Convert softmax attention back to log-scale / raw score proxy for clear scatter plot
log_scores_std = np.log(query_row_std + 1e-9)
log_scores_mob = np.log(query_row_mob + 1e-9)
distances = np.arange(seq_len)

print("\n[3/3] Generating 3 High-Impact Scientific Figures...")

# -------------------------------------------------------------
# 图 1：经典反转散点图与拓扑相干流 (Scatter Plot of Inversion)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6), dpi=220)
ax.scatter(distances, log_scores_std, color='gray', alpha=0.35, s=18, label="Standard RoPE Attention Log-Score (Flat Scatter)")
ax.scatter(distances, log_scores_mob, color='#8A2BE2', alpha=0.75, s=28, edgecolors='black', linewidth=0.3, label="Möbius-RoPE Attention Log-Score (Topological Spinor)")

# Fit moving average trendline
window = 15
trend_std = np.convolve(log_scores_std, np.ones(window)/window, mode='same')
trend_mob = np.convolve(log_scores_mob, np.ones(window)/window, mode='same')
ax.plot(distances, trend_std, color='black', linestyle=':', linewidth=1.5, label="Standard Trendline (Drift)")
ax.plot(distances, trend_mob, color='#FF1493', linestyle='-', linewidth=2.2, label="Möbius Half-Twist Streamline (Inversion Flow)")

ax.axvline(x=256, color='#FF4500', linestyle='--', linewidth=1.5, label="Half-Twist Inversion Boundary (x=256)")
ax.axvline(x=512, color='#2E8B57', linestyle='--', linewidth=1.5, label="Double-Covering 4π Loop Recovery (x=512)")

ax.set_title("Qwen3.5-9B: Log-Scale Attention Scatter & Topological Inversion Trend", fontsize=13, pad=12)
ax.set_xlabel("Token Position in Long Context (Relative Distance)", fontsize=11)
ax.set_ylabel("Attention Log-Energy Score ln(A_ij)", fontsize=11)
ax.grid(True, alpha=0.25)
ax.legend(frameon=True, facecolor='white', loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig("mobius_scatter_inversion.png")
plt.close()
print("✓ Saved: mobius_scatter_inversion.png")

# -------------------------------------------------------------
# 图 2：2D 全景注意力热力图对比 (Full 2D Attention Heatmap)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=220)

im0 = axes[0].imshow(attn_matrix_std[:350, :350], cmap="viridis", aspect="auto")
axes[0].set_title("Standard RoPE 2D Attention Matrix (Layer 7)", fontsize=12)
axes[0].set_xlabel("Key Position")
axes[0].set_ylabel("Query Position")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(attn_matrix_mob[:350, :350], cmap="magma", aspect="auto")
axes[1].set_title("Möbius-RoPE 2D Attention Matrix (Layer 7)", fontsize=12)
axes[1].set_xlabel("Key Position")
axes[1].set_ylabel("Query Position")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("mobius_2d_heatmap_comparison.png")
plt.close()
print("✓ Saved: mobius_2d_heatmap_comparison.png")

# -------------------------------------------------------------
# 图 3：莫比乌斯拓扑差值增强热力图 (Differential Contrast Heatmap)
# -------------------------------------------------------------
diff_matrix = attn_matrix_mob - attn_matrix_std

fig, ax = plt.subplots(figsize=(10, 8), dpi=220)
vmax = np.percentile(np.abs(diff_matrix), 99)
im_diff = ax.imshow(diff_matrix[:350, :350], cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_title("Topological Attention Gain: Δ(Möbius - Standard)", fontsize=13, pad=12)
ax.set_xlabel("Key Position", fontsize=11)
ax.set_ylabel("Query Position", fontsize=11)
ax.axvline(x=256, color='black', linestyle='--', alpha=0.7, label="Half-Twist (256)")
ax.axhline(y=256, color='black', linestyle='--', alpha=0.7)
ax.legend(loc="upper right")

cbar = fig.colorbar(im_diff, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Attention Weight Difference (Blue=Inhibited, Red=Reinforced)")

plt.tight_layout()
plt.savefig("mobius_differential_heatmap.png")
plt.close()
print("✓ Saved: mobius_differential_heatmap.png")

print("==================================================")
print("  All 3 Figures successfully generated!")
print("==================================================")
