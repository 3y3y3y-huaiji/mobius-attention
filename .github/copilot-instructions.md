# GitHub Copilot Instructions for SG-Möbius Attention

> **STATUS (2026-11)**: Historical SG-Möbius contract for the retained `src/` modules and `paper/`.
> The current research path is `src/phase_attention.py` plus the two `python -m experiments...` entry points;
> see `README.md` and `RESEARCH_PLAN.md`. The test count and benchmark metrics below are archived, not current validation.

## Project Overview
This repository implements Subspace Anti-Phase Attention (SG-Möbius-RoPE) for large language models (verified on Qwen3.5-9B).
It decouples the embedding space into an anchor subspace S_anchor (preserving standard RoPE) and an evolution subspace S_mob (undergoing destructive phase inversion Delta Phi = pi upon narrative invalidation).

## Implementation & Style Standards
- Model Code: `src/` (PyTorch, FP16/BF16/FP32).
- Formal Verification: `formal_verification/` (Lean 4, lake).
- Unit Tests: `tests/` (Pytest, pytest-xdist, Hypothesis). All 453 tests must pass.
- Paper: `paper/` (LaTeX NeurIPS/ICLR style).

## Key Implementation Constraints
1. Transition Detector Gating:
   In `src/gauge_predictor.py`, the dead-zone barrier `tau_gate` operates directly on the unscaled linear projection `z = w_g^T x + b_g`:
   `g(h_t) = omega_max * softsign(ReLU(|z| - tau_gate) * sgn(z))`
2. Prefill vs. Decoding Equivalence:
   Parallel sequence scan in `forward()` and step-by-step decoding in `step()` must produce mathematically identical results (`< 1e-12` in FP64).
3. Zero-Trust SSOT:
   Never change benchmark metrics without verifying against `experiments/kaggle_experiment_results.json`.
4. Consult `AI_HANDOVER.md` for complete architectural and mathematical specifications.
