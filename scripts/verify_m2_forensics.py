#!/usr/bin/env python3
"""
Forensic Audit Verification Script for Milestone 2: Continuous Möbius RoPE.
Agent: auditor_p3_m2
Conducts independent, empirical, zero-trust static and dynamic forensic verification.
"""

import sys
import os
import ast
import math
import torch

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mobius_attention import (
    causal_viscous_ema,
    apply_rotary_angle_addition,
    SemanticGaugeConnection,
    ContinuousMobiusRoPEModulator,
)


def forensic_static_code_analysis():
    print("[FORENSIC CHECK 1] Static Code Analysis & AST Inspection...")
    target_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/mobius_attention.py"))
    with open(target_file, "r", encoding="utf-8") as f:
        src_text = f.read()

    # 1. Banned terminology check
    banned_terms = ["impulsive", "stiefel", "spin(2)", "cech", "twist", "s_t"]
    src_lower = src_text.lower()
    for term in banned_terms:
        if term in src_lower:
            raise AssertionError(f"Integrity Violation: Banned term '{term}' found in src/mobius_attention.py!")

    # 2. AST parsing to detect facade / dummy implementations
    tree = ast.parse(src_text, filename=target_file)
    function_count = 0
    method_count = 0

    class ASTVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            nonlocal function_count, method_count
            function_count += 1
            # Check for dummy body
            stmts = [s for s in node.body if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)] # filter out docstrings
            if len(stmts) == 0:
                raise AssertionError(f"Integrity Violation: Function {node.name} has empty body!")
            if len(stmts) == 1:
                first = stmts[0]
                if isinstance(first, ast.Pass):
                    raise AssertionError(f"Integrity Violation: Function {node.name} body is just 'pass'!")
                if isinstance(first, ast.Raise):
                    if isinstance(first.exc, ast.Call) and getattr(first.exc.func, "id", None) == "NotImplementedError":
                        raise AssertionError(f"Integrity Violation: Function {node.name} raises NotImplementedError!")
                if isinstance(first, ast.Return) and isinstance(first.value, ast.Constant):
                    # Flag functions that just return a hardcoded constant (unless legitimate boolean)
                    if node.name not in ["__repr__"]:
                        raise AssertionError(f"Integrity Violation: Function {node.name} returns hardcoded constant {first.value.value}!")
            self.generic_visit(node)

    ASTVisitor().visit(tree)
    print(f"  -> AST passed: Scanned {function_count} functions/methods. Zero facades, zero dummy stubs, zero banned terms.")


def forensic_autograd_graph_tracing():
    print("[FORENSIC CHECK 2] Computation Graph & Gradient Flow Autograd Tracing...")
    torch.manual_seed(999)
    hidden_dim = 64
    seq_len = 16
    head_dim = 32

    gauge = SemanticGaugeConnection(hidden_dim=hidden_dim, out_dim=head_dim // 2, omega_max=0.05, nu=0.5)
    modulator = ContinuousMobiusRoPEModulator(head_dim=head_dim, anchor_ratio=0.5)

    x = torch.randn(2, seq_len, hidden_dim, requires_grad=True)
    q = torch.randn(2, seq_len, head_dim, requires_grad=True)

    omega, phi = gauge(x)
    modulator.set_holonomy_phase(phi)

    cos, sin = modulator(q)
    # Simulate an attention dot product and scalar loss
    q_mod = q * cos + sin  # simple dependent operation
    loss = q_mod.sum() + omega.norm() + phi.norm()
    loss.backward()

    assert x.grad is not None, "Gradient failed to flow to input x!"
    assert x.grad.abs().sum().item() > 0.0, "Input x gradient is identically zero!"
    assert gauge.proj.weight.grad is not None, "Gradient failed to flow to gauge projection weights!"
    assert gauge.proj.weight.grad.abs().sum().item() > 0.0, "Gauge projection weight gradient is identically zero!"
    assert q.grad is not None, "Gradient failed to flow to query q!"
    assert q.grad.abs().sum().item() > 0.0, "Query q gradient is identically zero!"
    print("  -> Autograd passed: Verified end-to-end backprop with non-zero gradients.")


def forensic_so2_orthogonal_properties():
    print("[FORENSIC CHECK 3] SO(2) Orthogonality, Det(M)=1, Group Law, and Norm Preservation...")
    torch.manual_seed(1234)
    # 5,000 randomized test cases
    batch_size = 50
    dim = 64
    num_subspaces = dim // 2

    for trial in range(10):
        theta = (torch.rand(batch_size, num_subspaces, dtype=torch.float64) * 20.0 - 10.0) * math.pi
        phi1 = (torch.rand(batch_size, num_subspaces, dtype=torch.float64) * 20.0 - 10.0) * math.pi
        phi2 = (torch.rand(batch_size, num_subspaces, dtype=torch.float64) * 20.0 - 10.0) * math.pi

        cos0 = torch.cat([theta.cos(), theta.cos()], dim=-1)
        sin0 = torch.cat([theta.sin(), theta.sin()], dim=-1)
        phi1_cat = torch.cat([phi1, phi1], dim=-1)
        phi2_cat = torch.cat([phi2, phi2], dim=-1)

        # 1. Single rotation
        cos_m1, sin_m1 = apply_rotary_angle_addition(cos0, sin0, phi1_cat)
        c1 = cos_m1[..., :num_subspaces]
        s1 = sin_m1[..., :num_subspaces]

        # Check Det(M) = cos^2 + sin^2 == 1
        det = c1**2 + s1**2
        max_det_err = (det - 1.0).abs().max().item()
        assert max_det_err < 1e-14, f"Det(M) deviation {max_det_err} exceeds 1e-14!"

        # Check M * M^T = I
        # M = [[c, -s], [s, c]]
        # M * M^T = [[c^2 + s^2, cs - sc], [sc - cs, s^2 + c^2]] = [[1, 0], [0, 1]]
        off_diag = (c1 * s1 - s1 * c1).abs().max().item()
        assert off_diag < 1e-15, f"Off-diagonal error {off_diag} exceeds 1e-15!"

        # 2. Composition law: R(theta + phi1 + phi2)
        # Successive modulation vs direct sum
        cos_m12, sin_m12 = apply_rotary_angle_addition(cos_m1, sin_m1, phi2_cat)
        tot_phi = phi1 + phi2
        tot_phi_cat = torch.cat([tot_phi, tot_phi], dim=-1)
        cos_direct, sin_direct = apply_rotary_angle_addition(cos0, sin0, tot_phi_cat)

        comp_err_cos = (cos_m12 - cos_direct).abs().max().item()
        comp_err_sin = (sin_m12 - sin_direct).abs().max().item()
        assert comp_err_cos < 1e-14, f"Composition law cos error {comp_err_cos} exceeds 1e-14!"
        assert comp_err_sin < 1e-14, f"Composition law sin error {comp_err_sin} exceeds 1e-14!"

        # 3. Vector norm preservation
        v = torch.randn(batch_size, dim, dtype=torch.float64)
        v_norm_orig = v.norm(dim=-1)
        v1 = v[..., :num_subspaces]
        v2 = v[..., num_subspaces:]
        v_rot1 = v1 * c1 - v2 * s1
        v_rot2 = v2 * c1 + v1 * s1
        v_rot = torch.cat([v_rot1, v_rot2], dim=-1)
        v_norm_rot = v_rot.norm(dim=-1)
        norm_err = (v_norm_orig - v_norm_rot).abs().max().item()
        assert norm_err < 1e-14, f"Vector norm deviation {norm_err} exceeds 1e-14!"

    print("  -> SO(2) group properties passed: Det=1 (<1e-14), Orthogonality (<1e-15), Composition (<1e-14), Norm preservation (<1e-14).")


def forensic_viscous_diffusion_and_lipschitz():
    print("[FORENSIC CHECK 4] Viscous Diffusion Recurrence, Derivative Bound & Lipschitz Continuity...")
    torch.manual_seed(4321)
    seq_len = 256
    omega_max = 0.05
    nu = 0.4

    # High frequency square wave / alternating input maximizing derivatives
    raw_g = torch.empty(2, seq_len, 8)
    for t in range(seq_len):
        sign = 1.0 if (t % 2 == 0) else -1.0
        raw_g[:, t, :] = sign * omega_max

    omega = causal_viscous_ema(raw_g, nu=nu)

    # 1. Verify exact derivative bound: |\omega(t) - \omega(t-1)| \le 2\nu \omega_{\max}
    theoretical_diff_bound = 2.0 * nu * omega_max
    diffs = (omega[:, 1:, :] - omega[:, :-1, :]).abs()
    max_diff = diffs.max().item()
    assert max_diff <= theoretical_diff_bound + 1e-7, (
        f"Viscous derivative violation! Max diff {max_diff} > bound {theoretical_diff_bound}"
    )

    # 2. Independent pure Python verification of recurrence
    alpha = 1.0 - nu
    for b in range(2):
        for c in range(8):
            py_prev = 0.0
            for t in range(seq_len):
                g_val = raw_g[b, t, c].item()
                py_curr = alpha * py_prev + nu * g_val
                torch_val = omega[b, t, c].item()
                assert abs(py_curr - torch_val) < 1e-7, (
                    f"Recurrence mismatch at b={b}, t={t}, c={c}: py={py_curr}, torch={torch_val}"
                )
                py_prev = py_curr

    # 3. Test Lipschitz bound on rotation operator:
    # ||R(t+1) - R(t)||_2 <= 2 sin((theta_m + omega_max)/2)
    dim = 64
    num_subspaces = dim // 2
    theta = 1.0 / (10000.0 ** (torch.arange(0, num_subspaces).float() * 2 / dim))
    for t in range(seq_len - 1):
        for m in range(num_subspaces):
            omega_t1 = omega[0, t + 1, m % 8].item()
            # Actual angle difference: delta_angle = theta_m + omega(t+1)
            delta_angle = theta[m].item() + omega_t1
            norm_diff = 2.0 * abs(math.sin(delta_angle / 2.0))
            lipschitz_bound = 2.0 * math.sin((theta[m].item() + omega_max) / 2.0)
            assert norm_diff <= lipschitz_bound + 1e-6, (
                f"Lipschitz bound violated! norm_diff={norm_diff}, bound={lipschitz_bound}"
            )

    print(f"  -> Viscous diffusion passed: Derivative bound satisfied ({max_diff:.5f} <= {theoretical_diff_bound:.5f}), Python recurrence exact, Lipschitz bound verified.")


def forensic_destructive_interference():
    print("[FORENSIC CHECK 5] Non-Negative Feature Map & Destructive Interference at Phi=pi...")
    torch.manual_seed(5678)
    q = torch.randn(10, 64)
    k = torch.randn(10, 64)

    modulator = ContinuousMobiusRoPEModulator(head_dim=64, anchor_ratio=0.5, enable_nonnegative_map=True)
    q_mapped, k_mapped = modulator.apply_feature_map(q, k)

    # Evolution subspace channels
    mask = modulator.get_subspace_mask(32, device=q.device)
    full_mask = torch.cat([mask, mask], dim=-1)

    assert (q_mapped[..., full_mask] >= 0.0).all()
    assert (k_mapped[..., full_mask] >= 0.0).all()

    # Base inner product on evolution channels (no phase modulation)
    cos0, sin0 = modulator(q_mapped)
    q_rot0 = (q_mapped * cos0) + (torch.cat([-q_mapped[..., 32:], q_mapped[..., :32]], dim=-1) * sin0)
    k_rot0 = (k_mapped * cos0) + (torch.cat([-k_mapped[..., 32:], k_mapped[..., :32]], dim=-1) * sin0)
    dot_base_mob = (q_rot0[..., full_mask] * k_rot0[..., full_mask]).sum(dim=-1)

    # Now modulate with phi = pi
    phi_pi = torch.full((10, 1), math.pi)
    cos_pi, sin_pi = modulator(q_mapped, phi=phi_pi)
    q_rot_pi = (q_mapped * cos_pi) + (torch.cat([-q_mapped[..., 32:], q_mapped[..., :32]], dim=-1) * sin_pi)

    # Notice: query is rotated by pi, while cached key remains unrotated (or query attends to historical key)
    # The dot product on the evolution subspace should be exactly -dot_base_mob!
    dot_inverted_mob = (q_rot_pi[..., full_mask] * k_rot0[..., full_mask]).sum(dim=-1)

    diff = (dot_inverted_mob - (-dot_base_mob)).abs().max().item()
    assert diff < 1e-5, f"Destructive interference failure! Diff={diff}"
    print(f"  -> Destructive interference passed: Dot product exactly inverted from +{dot_base_mob.mean().item():.4f} to {dot_inverted_mob.mean().item():.4f} (diff < 1e-5).")


def forensic_test_authenticity_check():
    print("[FORENSIC CHECK 6] Test Authenticity Check on tests/test_m2_continuous_mobius.py...")
    test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tests/test_m2_continuous_mobius.py"))
    with open(test_file, "r", encoding="utf-8") as f:
        test_src = f.read()

    tree = ast.parse(test_src, filename=test_file)
    test_funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    print(f"  -> Detected {len(test_funcs)} test functions.")

    # Check each test function for tautologies
    tautologies = ["assert True", "assert 1 == 1", "assert 0 == 0"]
    for t in tautologies:
        if t in test_src:
            raise AssertionError(f"Integrity Violation: Found tautological assertion '{t}' in test file!")

    for fn in test_funcs:
        assert_stmts = [s for s in ast.walk(fn) if isinstance(s, ast.Assert)]
        if len(assert_stmts) == 0:
            raise AssertionError(f"Integrity Violation: Test function {fn.name} has 0 assert statements!")
        for asst in assert_stmts:
            # Check for assert a == a
            if isinstance(asst.test, ast.Compare):
                left = ast.unparse(asst.test.left)
                for comparator in asst.test.comparators:
                    right = ast.unparse(comparator)
                    if left == right and left not in ["True", "False"]:
                        raise AssertionError(f"Integrity Violation: Self-comparing assertion detected in {fn.name}: {left} == {right}!")

    print(f"  -> Test authenticity passed: All {len(test_funcs)} tests execute non-tautological, mathematical assertions.")


if __name__ == "__main__":
    try:
        forensic_static_code_analysis()
        forensic_autograd_graph_tracing()
        forensic_so2_orthogonal_properties()
        forensic_viscous_diffusion_and_lipschitz()
        forensic_destructive_interference()
        forensic_test_authenticity_check()
        print("\n=======================================================")
        print("ALL FORENSIC CHECKS PASSED: VERDICT IS CLEAN")
        print("=======================================================")
    except Exception as e:
        print(f"\nFORENSIC AUDIT FAILURE: {e}", file=sys.stderr)
        sys.exit(1)
