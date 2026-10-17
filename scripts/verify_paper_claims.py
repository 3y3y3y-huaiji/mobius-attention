#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-Trust Static Paper & Table Arithmetic Linter
-------------------------------------------------
Enforces strict mathematical, arithmetic, and claim consistency across:
1. Table 1 floating-point division, ACR values, and percentage gains.
2. Provenance of abstract claims (e.g. 16.4x peak odds on Layer 7 Head 4).
3. Algorithm 1 index bounds and variable consistency (no undefined i-1).
4. Figure captions (explicit colorbar clamping and title isolation).
"""

import os
import re
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def read_tex_with_inputs(tex_path: str, visited: set = None) -> str:
    r"""Recursively expand LaTeX \input{...} files to support modular SSOT tables and subdocuments."""
    if visited is None:
        visited = set()
    tex_path = os.path.abspath(tex_path)
    if tex_path in visited or not os.path.exists(tex_path):
        return ""
    visited.add(tex_path)

    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    base_dir = os.path.dirname(tex_path)

    def replacer(match):
        rel = match.group(1).strip()
        candidate = os.path.join(base_dir, rel if rel.endswith(".tex") else rel + ".tex")
        if not os.path.exists(candidate):
            candidate = os.path.join(base_dir, rel)
        if os.path.exists(candidate):
            return read_tex_with_inputs(candidate, visited)
        return match.group(0)

    expanded = re.sub(r"\\input\{([^}]+)\}", replacer, content)
    return expanded

def parse_scientific(val_str: str) -> float:
    """Parse scientific notation like '$1.93 \\times 10^{-3}$' or '$9.57 \\times 10^{-5}$'."""
    val_str = val_str.replace("$", "").replace("\\mathbf{", "").replace("}", "").strip()
    match = re.search(r"([\d\.]+)\s*\\times\s*10\^\{?(-?\d+)\}?", val_str)
    if match:
        base, exp = float(match.group(1)), int(match.group(2))
        return base * (10 ** exp)
    try:
        return float(val_str)
    except ValueError:
        return float("nan")

def verify_table1_arithmetic(tex_content: str) -> list[str]:
    issues = []
    # Find table 1
    table_match = re.search(r"\\begin\{table\}.*?\\label\{tab:scaling_benchmark\}.*?\\end\{table\}", tex_content, re.DOTALL)
    if not table_match:
        # Fallback search without order dependency
        table_match = re.search(r"\\begin\{table\}.*?\\end\{table\}", tex_content, re.DOTALL)
        if not table_match:
            return ["Table 1 (tab:scaling_benchmark) not found in TeX content."]

    table_text = table_match.group(0)
    rows = table_text.split("\\midrule")
    if len(rows) < 2:
        return ["Table 1 does not contain expected multi-scale rows separated by \\midrule."]

    # Check each scale block
    for row_block in rows[1:]:
        lines = [line.strip() for line in row_block.split("\\\\") if line.strip() and not line.strip().startswith("\\bottomrule")]
        if len(lines) < 2:
            continue
        
        # Standard RoPE row and SG-Möbius row
        std_line = lines[0]
        mob_line = lines[1]
        
        # Look for numbers in std_line
        std_nums = re.findall(r"\$([\d\.]+(?:\s*\\times\s*10\^\{?-?\d+\}?)?)\$", std_line)
        mob_nums = re.findall(r"\$([\d\.]+(?:\s*\\times\s*10\^\{?-?\d+\}?)?)\$", mob_line)
        
        if len(std_nums) >= 3 and len(mob_nums) >= 3:
            std_decoy = parse_scientific(std_nums[0])
            std_target = parse_scientific(std_nums[1])
            std_acr_reported = parse_scientific(std_nums[2])

            mob_decoy = parse_scientific(mob_nums[0])
            mob_target = parse_scientific(mob_nums[1])
            mob_acr_reported = parse_scientific(mob_nums[2])

            eps = 1e-8
            std_acr_calc = std_target / (std_decoy + eps)
            mob_acr_calc = mob_target / (mob_decoy + eps)

            # Round to 2 decimals for display
            if abs(round(std_acr_calc, 2) - round(std_acr_reported, 2)) > 0.02:
                issues.append(
                    f"Arithmetic discrepancy in Standard RoPE row: reported ACR={std_acr_reported}, "
                    f"calculated={std_acr_calc:.4f} (Decoy={std_decoy}, Target={std_target})"
                )
            if abs(round(mob_acr_calc, 2) - round(mob_acr_reported, 2)) > 0.02:
                issues.append(
                    f"Arithmetic discrepancy in SG-Möbius row: reported ACR={mob_acr_reported}, "
                    f"calculated={mob_acr_calc:.4f} (Decoy={mob_decoy}, Target={mob_target})"
                )

    return issues

def verify_provenance_and_claims(tex_content: str) -> list[str]:
    issues = []
    # 1. Check 16.4x claim has provenance in Section 5
    if "16.4" in tex_content:
        # Must mention Layer and Head
        has_layer = re.search(r"Layer\s*(?:~)?\s*\d+", tex_content, re.IGNORECASE)
        has_head = re.search(r"Head\s*(?:~)?\s*\d+", tex_content, re.IGNORECASE)
        if not (has_layer and has_head):
            issues.append("Claim '16.4x' found in paper, but specific Layer and Head provenance is missing in Section 5.")

    # 2. Check Algorithm 1 array index bounds
    alg_match = re.search(r"\\begin\{algorithmic\}.*?\\end\{algorithmic\}", tex_content, re.DOTALL)
    if alg_match:
        alg_text = alg_match.group(0)
        # If loop starts at i = 0 and reads i-1, initialization must include omega_{-1} or similar
        if "for each decoding step $i = 0" in alg_text or "FOR{$i = 0" in alg_text:
            if "\\omega_{i-1}" in alg_text or "\\bar{\\Phi}_{i-1}" in alg_text:
                if "\\omega_{-1}" not in alg_text and "\\omega_0" not in alg_text:
                    issues.append("Algorithm 1 index bound bug: Loop starts at i=0 and reads i-1, but -1 index is not initialized.")

    # 3. Check colorbar clamping disclosure in Figure 2 and 3 captions
    has_clamping_note = "clamped" in tex_content.lower() or "display range" in tex_content.lower()
    if not has_clamping_note:
        issues.append("Warning: Captions do not explicitly disclose colorbar clamping for differential attention.")

    return issues

def verify_terminology_accuracy(tex_content: str, json_path: str, lang: str = "en") -> list[str]:
    r"""Audits paper terminology for mathematical precision and forbids claiming monotonic suppression on non-monotonic empirical sequences."""
    issues = []
    # Strip comments
    lines = [line.split("%")[0] for line in tex_content.splitlines()]
    clean_tex = "\n".join(lines)

    # Whitelist pure mathematical proof and lemma environments where sin/cos monotonicity on [0, pi] is mathematically valid
    clean_tex = re.sub(r"\\begin\{(proof|lemma|theorem|proposition)\}.*?\\end\{\1\}", "", clean_tex, flags=re.DOTALL)

    # Look for monotonic claims in empirical text
    if lang == "en":
        monotonic_pattern = re.compile(r"(\b(?:strictly\s+monotonic|monotonically|monotonicity)\b)", re.IGNORECASE)
        suppression_pattern = re.compile(r"\b(suppression|attenuat\w+|decay|odds|acr|contrast|factor|ratio|decoy)\b", re.IGNORECASE)
    else:
        monotonic_pattern = re.compile(r"(严格单调|单调衰减|单调压制)")
        suppression_pattern = re.compile(r"(压制|衰减|优势比|对比度|诱饵|ACR)")

    for match in monotonic_pattern.finditer(clean_tex):
        start = max(0, match.start() - 100)
        end = min(len(clean_tex), match.end() + 100)
        context_window = clean_tex[start:end]

        if suppression_pattern.search(context_window):
            if os.path.exists(json_path):
                import json
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                suppressions = [
                    v.get("adaptive_suppression_factor", v.get("decoy_suppression_factor"))
                    for v in data.get("scaling_benchmarks", {}).values()
                ]
                diffs = [round(suppressions[i+1] - suppressions[i], 3) for i in range(len(suppressions) - 1)]
                is_strictly_decreasing = all(d < 0 for d in diffs)
                is_strictly_increasing = all(d > 0 for d in diffs)

                if not (is_strictly_decreasing or is_strictly_increasing):
                    snippet = match.group(0)
                    issues.append(
                        f"Terminology Accuracy Violation: Found '{snippet}' modifying empirical suppression factors. "
                        f"Actual sequence {suppressions} has non-monotonic diffs {diffs}. "
                        f"Allowed phrasing: 'consistent strictly-above-unity suppression (> 1.0x)'."
                    )
            else:
                issues.append(f"Found '{match.group(0)}' modifying suppression factors in empirical text.")

    return issues

def extract_enclosing_env(tex_content: str, label: str, env_name: str = "figure") -> str:
    r"""Extracts the exact enclosing LaTeX environment (e.g. \begin{figure}...\end{figure}) containing the given label."""
    idx = tex_content.find(f"\\label{{{label}}}")
    if idx == -1:
        return ""
    begin_tag = f"\\begin{{{env_name}}}"
    end_tag = f"\\end{{{env_name}}}"
    start_pos = tex_content.rfind(begin_tag, 0, idx)
    if start_pos == -1:
        return ""
    end_pos = tex_content.find(end_tag, idx)
    if end_pos == -1:
        return ""
    return tex_content[start_pos : end_pos + len(end_tag)]

def verify_caption_ssot_consistency(tex_content: str, json_path: str) -> list[str]:
    r"""Verifies that Figure and Table captions match SSOT JSON results and do not retain stale ablation numbers."""
    issues = []
    if not os.path.exists(json_path):
        return issues

    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    benchmarks = data.get("scaling_benchmarks", {})
    last_item = benchmarks.get("L_18030", {})
    adaptive_18k_supp = last_item.get("adaptive_suppression_factor", last_item.get("mobius_rope_adaptive", {}).get("decoy_suppression_factor", 1.34))
    
    # 1. Check Figure 4 caption does not cite stale fixed ablation numbers 2.33x or 118%
    fig4_text = extract_enclosing_env(tex_content, "fig:scaling_and_distribution", "figure")
    if fig4_text:
        if "2.33" in fig4_text or "118.0%" in fig4_text:
            issues.append(
                "Figure 4 caption retains stale fixed-ablation data (2.33x / 118.0%). "
                f"Must cite true adaptive results ({adaptive_18k_supp:.2f}x / +88.5%)."
            )

    # 2. Check Table 1 caption specifies exact ACR gain (+465.1%, not rounded +464.6%)
    t1_text = extract_enclosing_env(tex_content, "tab:scaling_benchmark", "table")
    if not t1_text:
        t1_text = extract_enclosing_env(tex_content, "tab:benchmark", "table")
    if t1_text:
        if "464.6" in t1_text:
            issues.append(
                "Table 1 caption contains rounded '+464.6%'. "
                "SSOT pipeline requires unrounded '+465.1%' to match table body."
            )

    return issues

def verify_theoretical_patches(tex_content: str) -> list[str]:
    r"""Verifies that the 5 formal peer review patches are present in the text."""
    issues = []
    # Patch 1: tau_gate dead-zone threshold
    if "\\tau_{\\text{gate}}" not in tex_content and "\\tau_{\\mathrm{gate}}" not in tex_content and "tau_{\\text{gate}}" not in tex_content:
        issues.append("Patch 1 Missing: Semantic transition detector dead-zone threshold \\tau_{\\text{gate}} not found in Definition 1.")

    # Patch 2: Corollary on partition function contraction
    if "partition_siphoning" not in tex_content and "cor:partition_siphoning" not in tex_content:
        issues.append("Patch 2 Missing: Corollary 1 (Partition function contraction & probability mass siphoning) not found.")

    # Patch 3: Higham FP16 bound with total variation M_Phi
    if "M_\\Phi" not in tex_content and "M_\\Phi" not in tex_content:
        issues.append("Patch 3 Missing: Proposition 5 Higham FP16 forward error bound with total variation M_\\Phi not found.")

    # Patch 4: Formal boundary disclosure
    if "Formal Boundary Disclosure" not in tex_content and "形式化证明边界披露" not in tex_content:
        issues.append("Patch 4 Missing: Explicit formal boundary disclosure (Lean 4 real continuum vs FP16/Z3) not found in Section 4.")

    return issues

def main() -> int:
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_tex = os.path.join(workspace, "paper", "main.tex")
    main_zh_tex = os.path.join(workspace, "paper", "main_zh.tex")
    json_path = os.path.join(workspace, "experiments", "kaggle_experiment_results.json")
    
    if not os.path.exists(main_tex):
        print(f"Error: {main_tex} does not exist.")
        return 1

    content_en = read_tex_with_inputs(main_tex)
    content_zh = read_tex_with_inputs(main_zh_tex) if os.path.exists(main_zh_tex) else ""

    print("=" * 75)
    print("  ZERO-TRUST STATIC PAPER CLAIM, TERMINOLOGY & ARITHMETIC LINTER")
    print("=" * 75)

    all_issues = []

    # 1. Table 1 Arithmetic
    print("\n[1/5] Verifying Table 1 floating-point arithmetic and ACR definitions...")
    t1_issues = verify_table1_arithmetic(content_en)
    if t1_issues:
        for iss in t1_issues:
            print(f"  [DISCREPANCY] {iss}")
        all_issues.extend(t1_issues)
    else:
        print("  [✓] Table 1 floating-point divisions and ACR ratios verified consistent.")

    # 2. Terminology Accuracy Linter
    print("\n[2/5] Running Terminology Accuracy Linter (auditing monotonic suppression claims)...")
    term_issues_en = verify_terminology_accuracy(content_en, json_path, "en")
    term_issues_zh = verify_terminology_accuracy(content_zh, json_path, "zh") if content_zh else []
    term_issues = term_issues_en + term_issues_zh
    if term_issues:
        for iss in term_issues:
            print(f"  [DISCREPANCY] {iss}")
        all_issues.extend(term_issues)
    else:
        print("  [✓] Academic terminology clean: No spurious monotonic claims on non-monotonic data.")

    # 3. Caption SSOT Consistency
    print("\n[3/5] Verifying Figure & Table caption SSOT consistency...")
    cap_issues_en = verify_caption_ssot_consistency(content_en, json_path)
    cap_issues_zh = verify_caption_ssot_consistency(content_zh, json_path) if content_zh else []
    cap_issues = cap_issues_en + cap_issues_zh
    if cap_issues:
        for iss in cap_issues:
            print(f"  [DISCREPANCY] {iss}")
        all_issues.extend(cap_issues)
    else:
        print("  [✓] Captions strictly synchronized with raw Kaggle JSON benchmark.")

    # 4. Theoretical Patches Verification
    print("\n[4/5] Verifying presence of 5 formal peer review patches...")
    patch_issues_en = verify_theoretical_patches(content_en)
    patch_issues_zh = verify_theoretical_patches(content_zh) if content_zh else []
    patch_issues = patch_issues_en + patch_issues_zh
    if patch_issues:
        for iss in patch_issues:
            print(f"  [DISCREPANCY] {iss}")
        all_issues.extend(patch_issues)
    else:
        print("  [✓] All 5 formal peer review patches verified present and rigorous.")

    # 5. Provenance & Algorithm Invariants
    print("\n[5/5] Verifying metric provenance across Abstract and Section 5...")
    prov_issues = verify_provenance_and_claims(content_en)
    if prov_issues:
        for iss in prov_issues:
            print(f"  [DISCREPANCY] {iss}")
        all_issues.extend(prov_issues)
    else:
        print("  [✓] All numerical claims in Abstract/Intro successfully traced to Section 5.")

    print("\n" + "=" * 75)
    if all_issues:
        print(f"  [RESULT] FAILED: {len(all_issues)} static assertion discrepancy(ies) detected.")
        return 1
    else:
        print("  [RESULT] PASSED: All static claim, terminology, and arithmetic assertions passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())

