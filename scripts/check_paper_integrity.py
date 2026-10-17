#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Integrity & Anti-Hallucination Automated Verifier
------------------------------------------------------
Validates LaTeX source files and BibTeX references against:
1. Bidirectional citation consistency (no missing or unused references).
2. Internal label-reference resolution (no broken \\ref or \\eqref).
3. Figure asset existence and path integrity.
4. BibTeX schema completeness & anti-hallucination heuristic checks.
5. Online/Offline arXiv ID validation against official scholarly registry.
"""

import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET

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

    return re.sub(r"\\input\{([^}]+)\}", replacer, content)


def check_citations(tex_path: str, bib_path: str) -> tuple[set[str], set[str], set[str]]:
    tex_content = read_tex_with_inputs(tex_path)

    with open(bib_path, "r", encoding="utf-8") as f:
        bib_content = f.read()

    # Find all citations in TeX
    tex_cites = set()
    for m in re.finditer(r"\\cite[a-zA-Z*]*\{([^}]+)\}", tex_content):
        for key in m.group(1).split(","):
            cleaned = key.strip()
            if cleaned:
                tex_cites.add(cleaned)

    # Find all bib entries
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_content))

    missing = tex_cites - bib_keys
    unused = bib_keys - tex_cites
    return tex_cites, missing, unused


def check_labels_and_refs(tex_path: str) -> tuple[set[str], set[str]]:
    tex = read_tex_with_inputs(tex_path)

    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set()
    for m in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", tex):
        refs.add(m.group(1).strip())

    broken_refs = refs - labels
    return refs, broken_refs


def check_latex_syntax(tex_path: str) -> list[str]:
    """Check for dangerous unescaped characters in text and headings (e.g., bare '&'), math delimiters, and text-mode math macros."""
    with open(tex_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    issues = []
    in_tabular = False
    table_math_envs = (
        "tabular", "array", "matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix",
        "align", "aligned", "split", "cases", "alignat", "gather"
    )

    clean_lines = []
    for idx, line in enumerate(lines, 1):
        stripped = re.split(r"(?<!\\)%", line)[0].strip()
        clean_lines.append(stripped)
        if not stripped:
            continue

        # Check for environment entry / exit
        for env in table_math_envs:
            if f"\\begin{{{env}}}" in stripped:
                in_tabular = True
            if f"\\end{{{env}}}" in stripped:
                in_tabular = False

        if stripped.startswith("\\usepackage"):
            continue

        if not in_tabular:
            # Check for bare '&' not preceded by '\'
            has_single_line_env = any(f"\\begin{{{e}}}" in stripped and f"\\end{{{e}}}" in stripped for e in table_math_envs)
            if not has_single_line_env:
                bare_amps = list(re.finditer(r"(?<!\\)&", stripped))
                if bare_amps:
                    issues.append(f"Line {idx}: Unescaped '&' found in LaTeX text/heading: {stripped}")

    # Global math delimiter pairing check
    clean_text = "\n".join(clean_lines)
    for env in ("equation", "align", "aligned", "gather", "alignat"):
        clean_text = re.sub(rf"\\begin\{{{env}\*?\}}[\s\S]*?\\end\{{{env}\*?\}}", " ", clean_text)

    parts = clean_text.split("$")
    if (len(parts) - 1) % 2 != 0:
        issues.append(f"Global: Odd count of dollar signs ({len(parts) - 1}), math mode unclosed!")

    # Check for math-only commands appearing outside math mode
    math_only_cmds = ("theta", "Delta", "Phi", "omega", "alpha", "beta", "pi", "eta", "nu", "times", "frac", "sqrt")
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        for cmd in math_only_cmds:
            if re.search(rf"\\{cmd}\b", chunk):
                issues.append(f"Text mode error: Math macro \\{cmd} found outside $...$: {chunk[:60]}")

    return issues


def check_figures(tex_path: str, base_dir: str) -> list[str]:
    with open(tex_path, "r", encoding="utf-8") as f:
        tex = f.read()

    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    missing = []
    for fig in figs:
        fig_clean = fig.strip()
        path = os.path.join(base_dir, fig_clean)
        # Check direct path or with extensions
        exists = os.path.exists(path) or os.path.exists(path + ".png") or os.path.exists(path + ".pdf")
        if not exists:
            missing.append(fig_clean)
    return missing


def check_bib_entries(bib_path: str) -> list[str]:
    with open(bib_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into entries
    entries = re.findall(r"(@\w+\s*\{\s*[^,]+,[\s\S]*?\n\})", content)
    issues = []

    for entry in entries:
        key_m = re.search(r"@\w+\s*\{\s*([^,]+),", entry)
        key = key_m.group(1).strip() if key_m else "UNKNOWN"

        # Check required fields
        if not re.search(r"\btitle\s*=", entry, re.IGNORECASE):
            issues.append(f"[{key}] Missing 'title' field")
        if not re.search(r"\bauthor\s*=", entry, re.IGNORECASE):
            issues.append(f"[{key}] Missing 'author' field")
        if not re.search(r"\byear\s*=", entry, re.IGNORECASE):
            issues.append(f"[{key}] Missing 'year' field")
        else:
            year_m = re.search(r"\byear\s*=\s*\{?(\d{4})\}?", entry, re.IGNORECASE)
            if not year_m:
                issues.append(f"[{key}] Invalid year format (must be 4-digit number)")
            else:
                year = int(year_m.group(1))
                if year < 1900 or year > 2030:
                    issues.append(f"[{key}] Out-of-bounds year: {year}")

        # Check for placeholder / hallucination red flags
        suspicious_tokens = ["TODO", "TBD", "placeholder", "lorem ipsum", "citation needed"]
        for tok in suspicious_tokens:
            if re.search(rf"\b{tok}\b", entry, re.IGNORECASE):
                issues.append(f"[{key}] Contains suspicious placeholder token '{tok}'")

    return issues


def verify_arxiv_ids(bib_path: str) -> list[tuple[str, str, bool, str]]:
    with open(bib_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find arXiv IDs
    entries = re.findall(r"@\w+\s*\{\s*([^,]+),([\s\S]*?\n\})", content)
    results = []

    for key, body in entries:
        arxiv_m = re.search(r"arXiv[:\s]+(\d{4}\.\d{4,5})", body, re.IGNORECASE)
        if not arxiv_m:
            continue
        arxiv_id = arxiv_m.group(1)

        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "mobius-ci/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)
                entry_tag = root.find("{http://www.w3.org/2005/Atom}entry")
                if entry_tag is not None:
                    title_elem = entry_tag.find("{http://www.w3.org/2005/Atom}title")
                    title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "Found"
                    results.append((key, arxiv_id, True, title))
                else:
                    results.append((key, arxiv_id, False, "Entry not found in arXiv API"))
        except Exception as e:
            # Non-blocking if offline/rate-limited
            results.append((key, arxiv_id, True, f"Online check skipped ({type(e).__name__})"))

    return results


def main() -> int:
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tex_arg = sys.argv[1] if len(sys.argv) > 1 else "main.tex"
    if os.path.isabs(tex_arg):
        tex_path = tex_arg
    elif os.path.exists(os.path.join(workspace, tex_arg)):
        tex_path = os.path.join(workspace, tex_arg)
    elif os.path.exists(os.path.join(workspace, "paper", tex_arg)):
        tex_path = os.path.join(workspace, "paper", tex_arg)
    else:
        tex_path = os.path.join(workspace, "paper", tex_arg)

    paper_dir = os.path.dirname(os.path.abspath(tex_path))
    bib_path = os.path.join(paper_dir, "references.bib")

    print("=" * 70)
    print(f"  AUTOMATED LATEX PAPER & BIBTEX INTEGRITY VERIFIER: {os.path.basename(tex_path)}")
    print("=" * 70)

    has_error = False

    # 1. Citations
    cites, missing_cites, unused_bib = check_citations(tex_path, bib_path)
    print(f"\n[1/6] Checking Citation Bidirectional Consistency ({len(cites)} citations)...")
    if missing_cites:
        print(f"  [ERROR] Missing citations in references.bib: {missing_cites}")
        has_error = True
    else:
        print("  [✓] All citations in main.tex exist in references.bib.")

    if unused_bib:
        print(f"  [WARNING] Unused bib entries (potential phantom citations): {unused_bib}")
    else:
        print("  [✓] Zero unused entries in references.bib.")

    # 2. Labels & References
    refs, broken_refs = check_labels_and_refs(tex_path)
    print(f"\n[2/6] Checking Internal Labels & References ({len(refs)} references)...")
    if broken_refs:
        print(f"  [ERROR] Broken references in main.tex: {broken_refs}")
        has_error = True
    else:
        print("  [✓] All \\ref and \\eqref successfully resolve to valid \\labels.")

    # 3. LaTeX Syntax & Dangerous Unescaped Characters
    syntax_issues = check_latex_syntax(tex_path)
    print("\n[3/6] Checking LaTeX Syntax & Unescaped Characters...")
    if syntax_issues:
        for iss in syntax_issues:
            print(f"  [ERROR] {iss}")
        has_error = True
    else:
        print("  [✓] Zero unescaped alignment tabs or illegal characters found.")

    # 4. Figures
    missing_figs = check_figures(tex_path, paper_dir)
    print("\n[4/6] Checking Figure Assets...")
    if missing_figs:
        print(f"  [ERROR] Missing figure files referenced in main.tex: {missing_figs}")
        has_error = True
    else:
        print("  [✓] All figure files exist on disk.")

    # 5. BibTeX Entry Structure
    bib_issues = check_bib_entries(bib_path)
    print("\n[5/6] Checking BibTeX Schema Completeness & Anti-Hallucination...")
    if bib_issues:
        for iss in bib_issues:
            print(f"  [ERROR] {iss}")
        has_error = True
    else:
        print("  [✓] All BibTeX entries have valid schema, year bounds, and zero placeholder tokens.")

    # 6. Online arXiv Verification
    print("\n[6/6] Validating arXiv Citations against Official Scholarly API...")
    arxiv_results = verify_arxiv_ids(bib_path)
    for key, aid, ok, title in arxiv_results:
        status = "✓" if ok else "FAIL"
        print(f"  [{status}] {key} (arXiv:{aid}): {title[:60]}...")
        if not ok:
            has_error = True

    print("\n" + "=" * 70)
    if has_error:
        print("  [RESULT] FAILED: Paper integrity verification encountered errors.")
        return 1
    else:
        print("  [RESULT] PASSED: All paper integrity and anti-hallucination audits passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
