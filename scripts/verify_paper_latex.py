r"""
LaTeX Preprint Syntactic and Structural Verification Script for paper/main.tex.

Verifies:
1. Environment balancing (\begin{X} matched with \end{X}).
2. Label & Reference integrity (all \ref{} resolved by corresponding \label{}).
3. Citation integrity (all \cite{} resolved by entries in references.bib).
4. Asset integrity (all \includegraphics{} paths exist on disk).
5. Banned terminology check (0 occurrences of banned phrases).
6. Model naming consistency (strictly Qwen3.5-9B).
"""

import os
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def verify_latex(tex_path="paper/main.tex", bib_path="paper/references.bib"):
    print(f"=== Starting LaTeX Verification for {tex_path} ===")
    assert os.path.exists(tex_path), f"File {tex_path} not found"
    assert os.path.exists(bib_path), f"File {bib_path} not found"

    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()

    with open(bib_path, "r", encoding="utf-8") as f:
        bib_content = f.read()

    errors = []
    warnings = []

    # 1. Environment Balancing
    env_stack = []
    env_matches = re.findall(r'\\(begin|end)\{([a-zA-Z0-9*]+)\}', tex_content)
    for action, env in env_matches:
        if action == "begin":
            env_stack.append(env)
        elif action == "end":
            if not env_stack:
                errors.append(f"Unmatched \\end{{{env}}} found.")
            else:
                last_env = env_stack.pop()
                if last_env != env:
                    errors.append(f"Mismatched environment: \\begin{{{last_env}}} closed by \\end{{{env}}}")

    if env_stack:
        for remaining in env_stack:
            errors.append(f"Unclosed environment: \\begin{{{remaining}}}")
    else:
        print("[✓] All LaTeX environments are properly balanced.")

    # 2. Label & Reference Resolution
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex_content))
    refs = set(re.findall(r'\\ref\{([^}]+)\}', tex_content))
    unresolved_refs = refs - labels
    if unresolved_refs:
        errors.append(f"Unresolved \\ref references: {unresolved_refs}")
    else:
        print(f"[✓] All {len(refs)} references are fully resolved against {len(labels)} defined labels.")

    # 3. Citation Resolution
    bib_keys = set(re.findall(r'@\w+\s*\{\s*([^,]+),', bib_content))
    bib_keys = {k.strip() for k in bib_keys}
    
    cite_strings = re.findall(r'\\cite\{([^}]+)\}', tex_content)
    cited_keys = set()
    for s in cite_strings:
        for part in s.split(","):
            cited_keys.add(part.strip())

    unresolved_cites = cited_keys - bib_keys
    if unresolved_cites:
        errors.append(f"Unresolved \\cite references: {unresolved_cites}")
    else:
        print(f"[✓] All {len(cited_keys)} citations are resolved against {len(bib_keys)} bibliography keys.")

    # 4. Figure Inclusions
    img_paths = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex_content)
    missing_images = []
    tex_dir = os.path.dirname(tex_path)
    for p in img_paths:
        full_path = os.path.join(tex_dir, p)
        if not os.path.exists(full_path):
            missing_images.append(full_path)

    if missing_images:
        errors.append(f"Missing image files: {missing_images}")
    else:
        print(f"[✓] All {len(img_paths)} included figures exist on disk.")

    # 5. Banned Terms Audit
    banned_terms = [
        "Impulsive Connection", "Discrete Curvature",
        "Stiefel-Whitney", "w_1(E)",
        r"4\pi-spinor", r"Spin(2)",
        "Čech", "Cech 1-Cocycle"
    ]
    found_banned = {}
    for term in banned_terms:
        count = tex_content.count(term)
        if count > 0:
            found_banned[term] = count

    if found_banned:
        errors.append(f"Banned terms found in {tex_path}: {found_banned}")
    else:
        print("[✓] Banned terminology count is strictly ZERO.")

    # 6. Model Name Consistency
    if "Qwen2.5" in tex_content or "Qwen-2.5" in tex_content:
        errors.append("Legacy model name Qwen2.5 detected. Prompt mandates strictly Qwen3.5-9B.")
    else:
        print("[✓] Model name consistency verified (strictly Qwen3.5-9B).")

    # 7. Summary
    if errors:
        print("\n[!] VERIFICATION FAILED with errors:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("\n=======================================================")
        print("  ALL LATEX STRUCTURE & PREPRINT CHECKS PASSED (100%)")
        print("=======================================================")
        return True

if __name__ == "__main__":
    success = verify_latex()
    sys.exit(0 if success else 1)
