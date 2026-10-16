#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render publication-grade scaling figures from official Kaggle benchmark data.
Outputs:
  - assets/fig_scaling_acr_entropy.png
  - paper/figures/fig_scaling_acr_entropy.png
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render():
    json_path = "experiments/kaggle_experiment_results.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    benchmarks = data["scaling_benchmarks"]
    lengths = [benchmarks[k]["seq_len"] for k in benchmarks]
    std_acrs = [benchmarks[k]["standard_rope"]["attention_contrast_ratio"] for k in benchmarks]
    mob_acrs = [benchmarks[k]["mobius_rope"]["attention_contrast_ratio"] for k in benchmarks]
    std_entropies = [benchmarks[k]["standard_rope"]["shannon_entropy"] for k in benchmarks]
    mob_entropies = [benchmarks[k]["mobius_rope"]["shannon_entropy"] for k in benchmarks]

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # Panel 1: Attention Contrast Ratio
    ax1.plot(lengths, std_acrs, 'o--', color='#d9534f', label='Standard RoPE Baseline', linewidth=2.0, markersize=7)
    ax1.plot(lengths, mob_acrs, 's-', color='#2e6da4', label='SG-Möbius-RoPE (Ours, +118.0% at 18k)', linewidth=2.5, markersize=8)
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Context Sequence Length (Tokens)', fontweight='bold')
    ax1.set_ylabel('Attention Contrast Ratio (ACR)', fontweight='bold')
    ax1.set_title('(a) Attention Contrast Ratio Scaling ($L \\in [545, 18030]$)', fontweight='bold', pad=10)
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92)

    # Panel 2: Shannon Entropy
    ax2.plot(lengths, std_entropies, 'o--', color='#d9534f', label='Standard RoPE Baseline', linewidth=2.0, markersize=7)
    ax2.plot(lengths, mob_entropies, 's-', color='#3c763d', label='SG-Möbius-RoPE (Ours)', linewidth=2.5, markersize=8)
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Context Sequence Length (Tokens)', fontweight='bold')
    ax2.set_ylabel('Attention Shannon Entropy $H(P)$', fontweight='bold')
    ax2.set_title('(b) Attention Shannon Entropy Across Scaling Regime', fontweight='bold', pad=10)
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92)

    plt.tight_layout()

    out_dirs = ["assets", "paper/figures"]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        target = os.path.join(d, "fig_scaling_acr_entropy.png")
        plt.savefig(target, bbox_inches='tight')
        print(f"Saved: {target}")

    plt.close()

if __name__ == "__main__":
    render()
