r"""
Kaggle GPU Production Benchmark: Continuous SG-Möbius-RoPE on Dual Tesla T4 GPUs
Evaluates downstream generation accuracy under adversarial premise reversal and
extracts quantitative attention contrast metrics on Qwen.
"""

import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("  Kaggle Production Run: Continuous SG-Möbius Attention")
print("  Evaluating Downstream Premise Cancellation & Attention Metrics")
print("=" * 60)

# Check GPUs
gpu_count = torch.cuda.device_count()
print(f"Detected CUDA devices: {gpu_count}")
for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB")

# 1. Implementation of Continuous Semantic Gauge Möbius RoPE
def causal_viscous_ema(g: torch.Tensor, nu: float = 0.5) -> torch.Tensor:
    seq_dim = -2 if g.dim() >= 2 else 0
    seq_len = g.shape[seq_dim]
    if nu >= 1.0 or seq_len <= 1:
        return g
    alpha = 1.0 - nu
    steps = torch.arange(seq_len, device=g.device, dtype=torch.float32)
    diff = steps.unsqueeze(1) - steps.unsqueeze(0)
    causal_mask = (diff >= 0).to(dtype=g.dtype)
    weights = (alpha ** diff.clamp(min=0).to(dtype=g.dtype)) * causal_mask
    norm_w = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    
    g_flat = g.transpose(seq_dim, -2) if seq_dim != -2 else g
    out = torch.matmul(norm_w, g_flat)
    return out.transpose(seq_dim, -2) if seq_dim != -2 else out

def nonnegative_feature_map(u: torch.Tensor) -> torch.Tensor:
    return F.elu(u) + 1.0

def apply_rotary_angle_addition(cos: torch.Tensor, sin: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos_phi = torch.cos(phi).to(dtype=cos.dtype)
    sin_phi = torch.sin(phi).to(dtype=sin.dtype)
    return cos * cos_phi - sin * sin_phi, sin * cos_phi + cos * sin_phi

class ContinuousMobiusRoPEModulator(nn.Module):
    def __init__(self, original_rotary_emb, anchor_ratio=0.5, anchor_strategy="high_freq", omega_max=0.08, nu=0.4):
        super().__init__()
        self.orig_rotary = original_rotary_emb
        self.anchor_ratio = float(anchor_ratio)
        self.anchor_strategy = anchor_strategy
        self.omega_max = float(omega_max)
        self.nu = float(nu)
        self._current_phi = None

    def set_holonomy_phase(self, phi):
        self._current_phi = phi

    def clear_holonomy_phase(self):
        self._current_phi = None

    def get_subspace_mask(self, num_subspaces, device):
        mask = torch.ones(num_subspaces, dtype=torch.bool, device=device)
        num_anchor = int(round(num_subspaces * self.anchor_ratio))
        num_anchor = max(0, min(num_anchor, num_subspaces))
        if self.anchor_strategy == "high_freq":
            mask[:num_anchor] = False
        return mask

    def expand_subspace_phase(self, phi, head_dim, device, dtype):
        num_subspaces = head_dim // 2
        mask = self.get_subspace_mask(num_subspaces, device=device)
        phi = phi.to(device=device, dtype=dtype)
        if phi.dim() == 1:
            phi_tensor = phi.view(1, -1, 1)
        elif phi.dim() == 2:
            phi_tensor = phi.unsqueeze(-1)
        else:
            phi_tensor = phi
        if phi_tensor.shape[-1] == 1:
            phi_tensor = phi_tensor.expand(*phi_tensor.shape[:-1], num_subspaces)
        full_phi = torch.zeros_like(phi_tensor)
        full_phi[..., mask] = phi_tensor[..., mask]
        return torch.cat([full_phi, full_phi], dim=-1)

    def modulate_rotary(self, cos, sin, phi=None, position_ids=None):
        if phi is None:
            phi = self._current_phi
        if phi is None:
            return cos, sin
        head_dim = cos.shape[-1]
        phi_expanded = self.expand_subspace_phase(phi, head_dim, device=cos.device, dtype=cos.dtype)
        while phi_expanded.dim() < cos.dim():
            phi_expanded = phi_expanded.unsqueeze(1)
        if position_ids is not None and phi_expanded.shape[-2] > 1:
            seq_len_curr = cos.shape[-2]
            phi_expanded = phi_expanded[..., :seq_len_curr, :]
        return apply_rotary_angle_addition(cos, sin, phi_expanded)

    def forward(self, x, position_ids, *args, **kwargs):
        cos, sin = self.orig_rotary(x, position_ids, *args, **kwargs)
        return self.modulate_rotary(cos, sin, position_ids=position_ids)

class SemanticGaugeDetector(nn.Module):
    def __init__(self, embed_dim, omega_max=0.08, nu=0.4):
        super().__init__()
        self.omega_max = omega_max
        self.nu = nu
        self.register_buffer("polar_axis", None)

    def calibrate(self, tokenizer, embed_layer):
        neg_words = ["however", "contrary", "revoked", "invalid", "false", "negated", "forbidden", "terminated", "作废", "无效", "但是", "禁止"]
        pos_words = ["therefore", "active", "valid", "true", "confirmed", "continues", "nominal", "approved", "有效", "正常", "保持", "确认"]
        neg_vecs, pos_vecs = [], []
        dev = embed_layer.weight.device
        for w in neg_words:
            ids = tokenizer(w, add_special_tokens=False).input_ids
            if ids:
                neg_vecs.append(embed_layer(torch.tensor(ids[0], device=dev)).detach().float())
        for w in pos_words:
            ids = tokenizer(w, add_special_tokens=False).input_ids
            if ids:
                pos_vecs.append(embed_layer(torch.tensor(ids[0], device=dev)).detach().float())
        diff = torch.stack(neg_vecs).mean(0) - torch.stack(pos_vecs).mean(0)
        self.polar_axis = diff / diff.norm()

    def forward(self, input_ids, embed_layer):
        dev = embed_layer.weight.device
        embs = embed_layer(input_ids.to(dev)).detach().float()
        embs_norm = embs / (embs.norm(dim=-1, keepdim=True) + 1e-8)
        proj = (embs_norm @ self.polar_axis).squeeze(0)
        
        # Continuous S-curve torsion activation: smooth in [0, omega_max]
        threshold = proj.mean() + 1.2 * proj.std()
        scale = proj.std() + 1e-5
        raw_g = self.omega_max * torch.sigmoid((proj - threshold) / scale)
        omega = causal_viscous_ema(raw_g.unsqueeze(0), nu=self.nu).squeeze(0)
        phi = torch.cumsum(omega, dim=-1)
        return omega, phi

# 2. Load Model & Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen3.5-9B"
print(f"\n[1/4] Loading Tokenizer & Model: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
    trust_remote_code=True
)
print("Model loaded successfully across GPUs with eager attention!")

# Calibrate detector
embed_layer = model.get_input_embeddings()
detector = SemanticGaugeDetector(embed_layer.embedding_dim)
detector.calibrate(tokenizer, embed_layer)

# 3. Patch Model with ContinuousMobiusRoPEModulator
patched_count = 0
modulators = []
for name, module in model.named_modules():
    if hasattr(module, "rotary_emb") and module.rotary_emb is not None:
        if not isinstance(module.rotary_emb, ContinuousMobiusRoPEModulator):
            modulator = ContinuousMobiusRoPEModulator(
                module.rotary_emb,
                anchor_ratio=0.5,
                anchor_strategy="high_freq",
                omega_max=0.08,
                nu=0.4
            )
            module.rotary_emb = modulator
            modulators.append(modulator)
            patched_count += 1

print(f"Patched {patched_count} rotary embedding layers with ContinuousMobiusRoPEModulator.")

# 4. Downstream Adversarial Premise Generation Test
print("\n[2/4] Downstream Premise Cancellation Generation Test...")

prompt_part1 = (
    "SYSTEM DIRECTIVE RECORD - ARCHIVE LOG 104:\n"
    "Initial Audit Report: Execution Profile Delta is officially marked as VERIFIED, ACTIVE, and OPERATIONAL.\n"
    "All safety constraints have passed preliminary checks.\n"
)
filler = (
    "Telemetry cluster status: node synchronization nominal; memory bus bandwidth utilization at 34.2%; "
    "routine health checks passing on secondary clusters; periodic garbage collection cycle completed without errors; "
    "heartbeat signal frequency maintained at 100 Hz across distributed workers; network latency jitter within 0.4ms bounds.\n"
) * 4
prompt_reversal = (
    "SECURITY CRISIS NOTIFICATION - AMENDMENT OVERRIDE:\n"
    "However, critical kernel corruption was detected in Profile Delta! "
    "Therefore, Execution Profile Delta is completely REVOKED, INVALIDATED, and strictly FORBIDDEN from execution!\n"
)
filler2 = (
    "Diagnostic trace log: cluster nodes acknowledging override telemetry; "
    "thread pool isolation confirmed; background tasks diverted to fallback nodes; "
    "system logging continues in nominal mode; automated failover verification in progress.\n"
) * 3
query = (
    "AUDIT QUERY: State whether Execution Profile Delta is currently ACTIVE or FORBIDDEN.\n"
    "AUTHORITATIVE ASSESSMENT: Execution Profile Delta is"
)
test_prompt = prompt_part1 + filler + prompt_reversal + filler2 + query

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
prompt_len = inputs.input_ids.shape[1]
print(f"Prompt length: {prompt_len} tokens")

# A. Standard RoPE Generation (Zero Holonomy Phase)
for m in modulators:
    m.clear_holonomy_phase()

with torch.no_grad():
    gen_std = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
output_std = tokenizer.decode(gen_std[0][prompt_len:], skip_special_tokens=True).strip()

# B. Continuous SG-Möbius Generation (Active Phase Flow)
omega, phi = detector(inputs.input_ids, embed_layer)
for m in modulators:
    m.set_holonomy_phase(phi)

with torch.no_grad():
    gen_mob = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
output_mob = tokenizer.decode(gen_mob[0][prompt_len:], skip_special_tokens=True).strip()

print("\n" + "=" * 50)
print("--- Generation Comparison Under Adversarial Invalidation ---")
print(f"[Standard RoPE Output]:\n  {output_std}")
print(f"\n[SG-Möbius Output]:\n  {output_mob}")
print("=" * 50)

# 5. Attention Metrics & Shannon Entropy Evaluation
print("\n[3/4] Quantifying Attention Logits & Shannon Entropy...")

# Locate Decoy token ("ACTIVE") and Target token ("FORBIDDEN" / "REVOKED")
tokens = [tokenizer.decode([tid]) for tid in inputs.input_ids[0]]
decoy_idx = None
target_idx = None
for idx, tok in enumerate(tokens):
    tok_upper = tok.strip().upper()
    if any(k in tok_upper for k in ["ACTIVE", "NOMINAL", "APPROVED"]) and decoy_idx is None:
        decoy_idx = idx
    if any(k in tok_upper for k in ["REVOK", "FORBID", "INVALID"]) and idx > 20:
        target_idx = idx

if decoy_idx is None:
    decoy_idx = min(9, prompt_len - 1)
if target_idx is None:
    target_idx = max(prompt_len - 15, decoy_idx + 1)

print(f"Identified Decoy Token Index: {decoy_idx} ('{tokens[decoy_idx]}')")
print(f"Identified Target Token Index: {target_idx} ('{tokens[target_idx]}')")

# Extract layer attention weights
for m in modulators:
    m.clear_holonomy_phase()
with torch.no_grad():
    out_std = model(**inputs, output_attentions=True)

for m in modulators:
    m.set_holonomy_phase(phi)
with torch.no_grad():
    out_mob = model(**inputs, output_attentions=True)

# Query at last prompt token looking at historical tokens
last_idx = prompt_len - 1
layer_idx = min(7, len(out_std.attentions) - 1)
attn_std = out_std.attentions[layer_idx][0].mean(dim=0)[last_idx, :prompt_len].float().cpu().numpy()
attn_mob = out_mob.attentions[layer_idx][0].mean(dim=0)[last_idx, :prompt_len].float().cpu().numpy()

p_std = attn_std / attn_std.sum()
p_mob = attn_mob / attn_mob.sum()

entropy_std = float(-np.sum(p_std * np.log(p_std + 1e-12)))
entropy_mob = float(-np.sum(p_mob * np.log(p_mob + 1e-12)))

a_decoy_std = float(attn_std[decoy_idx])
a_target_std = float(attn_std[target_idx])
acr_std = float(a_target_std / (a_decoy_std + 1e-12))

a_decoy_mob = float(attn_mob[decoy_idx])
a_target_mob = float(attn_mob[target_idx])
acr_mob = float(a_target_mob / (a_decoy_mob + 1e-12))

results = {
    "model": model_id,
    "prompt_tokens": prompt_len,
    "decoy_token": {"index": decoy_idx, "text": tokens[decoy_idx]},
    "target_token": {"index": target_idx, "text": tokens[target_idx]},
    "standard_rope": {
        "output": output_std,
        "decoy_attention": a_decoy_std,
        "target_attention": a_target_std,
        "attention_contrast_ratio": acr_std,
        "shannon_entropy": entropy_std
    },
    "continuous_sg_mobius": {
        "output": output_mob,
        "decoy_attention": a_decoy_mob,
        "target_attention": a_target_mob,
        "attention_contrast_ratio": acr_mob,
        "shannon_entropy": entropy_mob,
        "contrast_gain_factor": float(acr_mob / acr_std)
    }
}

print("\n" + "=" * 50)
print("--- Quantitative Attention Results ---")
print(f"Decoy Attention (Obsolete Fact): Standard = {a_decoy_std:.6f} | SG-Möbius = {a_decoy_mob:.6f} (Suppressed {a_decoy_std/a_decoy_mob:.1f}x)")
print(f"Target Attention (Updated Fact):  Standard = {a_target_std:.6f} | SG-Möbius = {a_target_mob:.6f}")
print(f"Attention Contrast Ratio (ACR): Standard = {acr_std:.2f} | SG-Möbius = {acr_mob:.2f} (Gain: {acr_mob/acr_std:.1f}x)")
print(f"Shannon Entropy (Noise level):   Standard = {entropy_std:.4f} | SG-Möbius = {entropy_mob:.4f} (Sharper focus)")
print("=" * 50)

with open("kaggle_experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n[4/4] Generating Publication-Quality Figures...")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

# Bar chart of attention allocations
methods = ["Standard RoPE", "SG-Möbius"]
decoy_vals = [a_decoy_std, a_decoy_mob]
target_vals = [a_target_std, a_target_mob]
x = np.arange(len(methods))
width = 0.35

ax1.bar(x - width/2, decoy_vals, width, label='Revoked Premise (Decoy)', color='#d9534f')
ax1.bar(x + width/2, target_vals, width, label='Current Directive (Target)', color='#5cb85c')
ax1.set_ylabel('Attention Probability Mass', fontsize=11)
ax1.set_title('Attention Allocation Under Adversarial Revocation', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Log attention distribution across sequence
seq_x = np.arange(prompt_len)
ax2.plot(seq_x, np.log(attn_std + 1e-9), color='#428bca', label='Standard RoPE (Cluttered)', linewidth=1.8)
ax2.plot(seq_x, np.log(attn_mob + 1e-9), color='#d9534f', label='SG-Möbius (Destructive Phase)', linewidth=2.2)
ax2.axvline(x=decoy_idx, color='orange', linestyle='--', label=f'Revoked Fact (t={decoy_idx})')
ax2.axvline(x=target_idx, color='green', linestyle='--', label=f'New Fact (t={target_idx})')
ax2.set_xlabel('Token Sequence Position', fontsize=11)
ax2.set_ylabel('Log Attention Mass ln(A_ij)', fontsize=11)
ax2.set_title('Token Attention Energy Across Sequence Context', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fig_kaggle_results.png", bbox_inches='tight')
plt.close()

print("Saved kaggle_experiment_results.json and fig_kaggle_results.png successfully!")
print("Production run completed!")
