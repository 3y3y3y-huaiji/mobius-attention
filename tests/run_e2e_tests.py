# -*- coding: utf-8 -*-
r"""
Standalone E2E Test Suite Runner for SG-Möbius-RoPE
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Executes all 4 test tiers and benchmark suites:
- Tier 1: Feature Coverage (F1 to F12, >= 60 tests)
- Tier 2: Boundary & Corner Cases (F1 to F12, >= 60 tests)
- Tier 3: Cross-Feature Combinations (12 tests)
- Tier 4: Real-World Application Scenarios (5 scenarios)
- Standalone Suites: test_antagonistic, test_cross_arch, test_long_context_niah, test_kv_linearity

Usage:
    python tests/run_e2e_tests.py
    python tests/run_e2e_tests.py --tier 1
    python tests/run_e2e_tests.py --tier 4
    python tests/run_e2e_tests.py --json report.json
"""

import sys
import os
import time
import argparse
import subprocess
import json

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)

TIER_SUITES = {
    "tier1": {
        "name": "Tier 1: Feature Coverage (F1 - F12)",
        "file": os.path.join(TESTS_DIR, "test_tier1_features.py"),
        "expected_min": 60,
    },
    "tier2": {
        "name": "Tier 2: Boundary & Corner Cases (F1 - F12)",
        "file": os.path.join(TESTS_DIR, "test_tier2_boundaries.py"),
        "expected_min": 60,
    },
    "tier3": {
        "name": "Tier 3: Cross-Feature Combinations",
        "file": os.path.join(TESTS_DIR, "test_tier3_combinations.py"),
        "expected_min": 10,
    },
    "tier4": {
        "name": "Tier 4: Real-World Application Scenarios",
        "file": os.path.join(TESTS_DIR, "test_tier4_scenarios.py"),
        "expected_min": 5,
    },
    "benchmarks": {
        "name": "Standalone Benchmark Suites (PROJECT.md)",
        "file": [
            os.path.join(TESTS_DIR, "test_antagonistic.py"),
            os.path.join(TESTS_DIR, "test_cross_arch.py"),
            os.path.join(TESTS_DIR, "test_long_context_niah.py"),
            os.path.join(TESTS_DIR, "test_kv_linearity.py")
        ],
        "expected_min": 20,
    }
}

def run_pytest_target(target_path):
    """Executes pytest on a file or list of files and parses output."""
    if isinstance(target_path, list):
        targets = target_path
    else:
        targets = [target_path]

    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"] + targets
    start_t = time.perf_counter()
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - start_t

    stdout = res.stdout
    stderr = res.stderr

    # Parse passed / failed / collected count
    passed = 0
    failed = 0
    for line in stdout.splitlines():
        if "passed" in line or "failed" in line:
            # e.g. "=== 60 passed in 0.43s ==="
            # or "=== 2 failed, 58 passed in 0.64s ==="
            parts = line.strip("= ").split(",")
            for p in parts:
                p_str = p.strip()
                if "passed" in p_str:
                    try:
                        passed = int(p_str.split()[0])
                    except ValueError:
                        pass
                if "failed" in p_str:
                    try:
                        failed = int(p_str.split()[0])
                    except ValueError:
                        pass

    return {
        "exit_code": res.returncode,
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "elapsed": elapsed,
        "stdout": stdout,
        "stderr": stderr
    }

def main():
    parser = argparse.ArgumentParser(description="SG-Möbius-RoPE E2E Test Suite Runner")
    parser.add_argument("--tier", choices=["1", "2", "3", "4", "benchmarks", "all"], default="all",
                        help="Specific test tier to execute (default: all)")
    parser.add_argument("--json", dest="json_output", help="Optional path to write JSON test report")
    args = parser.parse_args()

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    selected_tiers = []
    if args.tier == "all":
        selected_tiers = ["tier1", "tier2", "tier3", "tier4", "benchmarks"]
    elif args.tier in ["1", "2", "3", "4"]:
        selected_tiers = [f"tier{args.tier}"]
    else:
        selected_tiers = [args.tier]

    print("=" * 80)
    print("  SG-Mobius-RoPE: Opaque-Box E2E Test Suite Execution")
    print("  Project Root: " + PROJECT_ROOT)
    print("  Python Executable: " + sys.executable)
    print("=" * 80)

    overall_passed = 0
    overall_failed = 0
    overall_total = 0
    tier_results = {}
    overall_start = time.perf_counter()

    for tier_key in selected_tiers:
        tier_info = TIER_SUITES[tier_key]
        print(f"\n[Running] {tier_info['name']}...")
        result = run_pytest_target(tier_info["file"])
        tier_results[tier_key] = result

        overall_passed += result["passed"]
        overall_failed += result["failed"]
        overall_total += result["total"]

        status_str = "PASSED" if result["exit_code"] == 0 and result["failed"] == 0 else "FAILED"
        print(f"  -> Result: {status_str} | Passed: {result['passed']} | Failed: {result['failed']} | Duration: {result['elapsed']:.2f}s")
        if result["failed"] > 0:
            print("--- Error Details ---")
            print(result["stdout"][-1500:])

    total_duration = time.perf_counter() - overall_start

    print("\n" + "=" * 80)
    print("  E2E TEST SUITE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Tier / Suite Name':<45} | {'Passed':<8} | {'Failed':<8} | {'Time (s)':<8} | {'Status'}")
    print("-" * 80)

    for tier_key in selected_tiers:
        t_info = TIER_SUITES[tier_key]
        r = tier_results[tier_key]
        status = "PASSED" if r["exit_code"] == 0 and r["failed"] == 0 else "FAILED"
        print(f"{t_info['name']:<45} | {r['passed']:<8} | {r['failed']:<8} | {r['elapsed']:<8.2f} | {status}")

    print("-" * 80)
    final_verdict = "ALL TEST TIERS PASSED [OK]" if overall_failed == 0 and overall_total > 0 else "TEST SUITE FAILED [FAIL]"
    print(f"{'TOTAL AGGREGATE':<45} | {overall_passed:<8} | {overall_failed:<8} | {total_duration:<8.2f} | {final_verdict}")
    print("=" * 80)

    if args.json_output:
        report_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_passed": overall_passed,
            "total_failed": overall_failed,
            "total_tests": overall_total,
            "duration_seconds": total_duration,
            "verdict": "PASSED" if overall_failed == 0 else "FAILED",
            "tier_details": {k: {"passed": v["passed"], "failed": v["failed"], "elapsed": v["elapsed"]} for k, v in tier_results.items()}
        }
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"[Info] Test report saved to {args.json_output}")

    exit_code = 0 if overall_failed == 0 else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
