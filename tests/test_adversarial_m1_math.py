# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite for SG-Möbius-RoPE Milestone 1
Target:
1. Theorem on Continuous Lipschitz Continuity Bound of Rotation Operator:
   ||R_tilde_m(t+1) - R_tilde_m(t)||_2 <= 2 * sin((theta_m + omega_max) / 2)
2. Proposition on Prefill vs Autoregressive KV-Cache Decoding Equivalence:
   A_decode[i, j] == A_prefill[i, j] for all causal pairs j <= i.
"""

import math
import random
import numpy as np
import pytest


def test_adversarial_lipschitz_within_valid_domain():
    """
    Empirically test Lipschitz continuity bound under 15,000 randomized configurations
    satisfying the validity precondition theta_m + omega_max <= pi.
    Asserts diff_norm <= theoretical_bound + 1e-12.
    """
    np.random.seed(1337)
    random.seed(1337)
    
    num_trials = 15000
    violations = 0
    max_ratio = 0.0
    
    for _ in range(num_trials):
        # Sample omega_max in (0, pi)
        omega_max = random.uniform(1e-4, math.pi - 1e-4)
        # Sample theta_m in (0, pi - omega_max] to satisfy theta_m + omega_max <= pi
        theta_m = random.uniform(1e-5, math.pi - omega_max)
        
        # Continuous torsion rate in [-omega_max, omega_max]
        omega_t1 = random.uniform(-omega_max, omega_max)
        
        # Arbitrary historical accumulated phases and positions
        t = random.randint(0, 100000)
        phi_t = random.uniform(-100.0, 100.0)
        phi_t1 = phi_t + omega_t1
        
        angle_t = t * theta_m + phi_t
        angle_t1 = (t + 1) * theta_m + phi_t1
        
        R_t = np.array([
            [np.cos(angle_t), -np.sin(angle_t)],
            [np.sin(angle_t),  np.cos(angle_t)]
        ], dtype=np.float64)
        
        R_t1 = np.array([
            [np.cos(angle_t1), -np.sin(angle_t1)],
            [np.sin(angle_t1),  np.cos(angle_t1)]
        ], dtype=np.float64)
        
        diff_norm = np.linalg.norm(R_t1 - R_t, ord=2)
        theoretical_bound = 2.0 * math.sin((theta_m + omega_max) / 2.0)
        
        if diff_norm > theoretical_bound + 1e-12:
            violations += 1
            
        ratio = diff_norm / theoretical_bound if theoretical_bound > 1e-12 else 0.0
        max_ratio = max(max_ratio, ratio)
        
    assert violations == 0, f"Found {violations} violations of Lipschitz bound in valid domain!"
    assert max_ratio <= 1.0 + 1e-12, f"Max ratio {max_ratio} exceeded 1.0!"
    print(f"Domain theta_m + omega_max <= pi: {num_trials} trials passed, max_ratio = {max_ratio:.6f}")


def test_adversarial_lipschitz_counterexample_search():
    """
    Adversarial search for counterexamples where theta_m + omega_max > pi.
    Demonstrates that without the explicit precondition theta_m + omega_max <= pi,
    the formula 2 * sin((theta_m + omega_max)/2) is strictly VIOLATED.
    """
    np.random.seed(42)
    random.seed(42)
    
    counterexamples = []
    num_trials = 10000
    
    for _ in range(num_trials):
        # Sample theta_m in (0, pi) and omega_max in (0, pi) such that theta_m + omega_max > pi
        omega_max = random.uniform(math.pi / 4, math.pi * 0.99)
        theta_m = random.uniform(max(1e-4, math.pi - omega_max + 0.01), math.pi * 0.99)
        
        # Adversary picks omega_t1 = pi - theta_m so that delta = theta_m + omega_t1 = pi
        # Since pi - theta_m <= omega_max and pi - theta_m >= 0, this omega_t1 is within [-omega_max, omega_max]!
        omega_t1 = math.pi - theta_m
        
        t = 0
        angle_t = 0.0
        angle_t1 = theta_m + omega_t1  # exactly pi
        
        R_t = np.eye(2, dtype=np.float64)
        R_t1 = np.array([
            [np.cos(angle_t1), -np.sin(angle_t1)],
            [np.sin(angle_t1),  np.cos(angle_t1)]
        ], dtype=np.float64)
        
        diff_norm = np.linalg.norm(R_t1 - R_t, ord=2)  # exactly 2.0
        theoretical_bound = 2.0 * math.sin((theta_m + omega_max) / 2.0)  # strictly < 2.0
        
        if diff_norm > theoretical_bound + 1e-12:
            counterexamples.append((theta_m, omega_max, omega_t1, diff_norm, theoretical_bound))
            
    print(f"Adversarial counterexample search: found {len(counterexamples)} / {num_trials} counterexamples when theta_m + omega_max > pi.")
    assert len(counterexamples) > 0, "Expected counterexamples when theta_m + omega_max > pi!"
    # Return worst violation
    worst = max(counterexamples, key=lambda x: x[3] - x[4])
    print(f"Worst violation: theta_m={worst[0]:.4f}, omega_max={worst[1]:.4f}, diff_norm={worst[3]:.4f} > bound={worst[4]:.4f} (excess={worst[3]-worst[4]:.4f})")


@pytest.mark.parametrize("seq_len", [10, 64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_prefill_vs_autoregressive_kv_equivalence(seq_len, dtype):
    """
    Stress-test Prefill vs Autoregressive KV-cache decoding equivalence
    across sequences of lengths L in [10, 2048] with arbitrary continuous omega(t) trajectories.
    Asserts max absolute difference is < 1e-5.
    """
    dim = 64
    np.random.seed(seq_len + (0 if dtype == np.float64 else 1000))
    
    inv_freq = (1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=dtype) / dim))).astype(dtype)
    
    # Generate arbitrary continuous omega(t) trajectory: combination of smooth flow + abrupt jumps
    raw_torsion = np.random.uniform(-0.5, 0.5, seq_len).astype(dtype)
    omega = np.zeros(seq_len, dtype=dtype)
    nu = dtype(0.15)
    for t in range(1, seq_len):
        if t % 50 == 0:
            omega[t] = raw_torsion[t]  # abrupt jump
        else:
            omega[t] = (dtype(1.0) - nu) * omega[t - 1] + nu * raw_torsion[t]
            
    phi = np.cumsum(omega, dtype=dtype)
    
    Q = np.random.randn(seq_len, dim).astype(dtype)
    K = np.random.randn(seq_len, dim).astype(dtype)
    
    # 1. Parallel Prefill Mode
    Q_rot = np.zeros_like(Q)
    K_rot = np.zeros_like(K)
    for t in range(seq_len):
        for m in range(dim // 2):
            th_m = dtype(t) * inv_freq[m] + phi[t]
            cos_th, sin_th = np.cos(th_m), np.sin(th_m)
            
            q1, q2 = Q[t, 2*m], Q[t, 2*m + 1]
            k1, k2 = K[t, 2*m], K[t, 2*m + 1]
            
            Q_rot[t, 2*m] = q1 * cos_th - q2 * sin_th
            Q_rot[t, 2*m + 1] = q1 * sin_th + q2 * cos_th
            
            K_rot[t, 2*m] = k1 * cos_th - k2 * sin_th
            K_rot[t, 2*m + 1] = k1 * sin_th + k2 * cos_th
            
    scale = dtype(1.0 / math.sqrt(dim))
    A_prefill = np.matmul(Q_rot, K_rot.T) * scale
    
    # 2. Autoregressive KV-Cache Step-by-Step Decoding
    kv_cache = []
    A_decode = np.zeros((seq_len, seq_len), dtype=dtype)
    
    for i in range(seq_len):
        # Cache key at step i using strictly causal past tau <= i
        k_cached = np.zeros(dim, dtype=dtype)
        for m in range(dim // 2):
            th_m = dtype(i) * inv_freq[m] + phi[i]
            cos_th, sin_th = np.cos(th_m), np.sin(th_m)
            k1, k2 = K[i, 2*m], K[i, 2*m + 1]
            k_cached[2*m] = k1 * cos_th - k2 * sin_th
            k_cached[2*m + 1] = k1 * sin_th + k2 * cos_th
        kv_cache.append(k_cached)
        
        # Decode query at step i
        q_dec = np.zeros(dim, dtype=dtype)
        for m in range(dim // 2):
            th_m = dtype(i) * inv_freq[m] + phi[i]
            cos_th, sin_th = np.cos(th_m), np.sin(th_m)
            q1, q2 = Q[i, 2*m], Q[i, 2*m + 1]
            q_dec[2*m] = q1 * cos_th - q2 * sin_th
            q_dec[2*m + 1] = q1 * sin_th + q2 * cos_th
            
        for j in range(i + 1):
            A_decode[i, j] = np.dot(q_dec, kv_cache[j]) * scale
            
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    max_abs_diff = float(np.max(np.abs((A_prefill - A_decode)[mask])))
    
    threshold = 1e-12 if dtype == np.float64 else 1e-5
    print(f"SeqLen={seq_len:4d}, Dtype={dtype.__name__}: Max Diff = {max_abs_diff:.3e} (threshold = {threshold})")
    assert max_abs_diff < threshold, f"Prefill vs KV-cache discrepancy too high: {max_abs_diff} >= {threshold}"


if __name__ == "__main__":
    test_adversarial_lipschitz_within_valid_domain()
    test_adversarial_lipschitz_counterexample_search()
    for seq_len in [10, 64, 128, 256, 512, 1024, 2048]:
        test_prefill_vs_autoregressive_kv_equivalence(seq_len, np.float64)
        test_prefill_vs_autoregressive_kv_equivalence(seq_len, np.float32)
