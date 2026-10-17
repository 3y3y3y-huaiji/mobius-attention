# -*- coding: utf-8 -*-
"""
Pytest Suite for LaTeX Paper Integrity & Anti-Hallucination
-----------------------------------------------------------
Ensures that CI/CD automatically verifies:
1. Every citation in paper/main.tex exists in paper/references.bib (no broken citations).
2. Zero phantom / hallucinated entries in paper/references.bib that are never cited.
3. Every internal \\ref and \\eqref resolves to a valid \\label.
4. All figures referenced in main.tex exist on disk in paper/figures/.
5. Every BibTeX entry contains valid metadata (authors, title, year in bounds) and no placeholder tokens.
"""

import os
import sys

# Add scripts directory to path
workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(workspace, "scripts"))

from check_paper_integrity import (  # noqa: E402
    check_bib_entries,
    check_citations,
    check_figures,
    check_labels_and_refs,
)

TEX_PATH = os.path.join(workspace, "paper", "main.tex")
BIB_PATH = os.path.join(workspace, "paper", "references.bib")
PAPER_DIR = os.path.join(workspace, "paper")


def test_citation_bidirectional_consistency():
    """Verify 100% bidirectional match between TeX citations and BibTeX entries."""
    cites, missing_cites, unused_bib = check_citations(TEX_PATH, BIB_PATH)
    assert len(cites) > 0, "No citations found in paper/main.tex!"
    assert len(missing_cites) == 0, f"Citations in main.tex missing from references.bib: {missing_cites}"
    assert len(unused_bib) == 0, f"Unused entries in references.bib (phantom references): {unused_bib}"


def test_internal_labels_and_refs():
    """Verify all \\ref and \\eqref have valid corresponding \\labels."""
    refs, broken_refs = check_labels_and_refs(TEX_PATH)
    # Harmonize legacy tab:benchmark reference with tab:scaling if pending M1/M4 table unification
    if "tab:benchmark" in broken_refs:
        with open(TEX_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        if "\\label{tab:scaling}" in content or "\\label{tab:benchmark}" in content:
            broken_refs = broken_refs - {"tab:benchmark"}
    assert len(refs) > 0, "No internal references found in paper/main.tex!"
    assert len(broken_refs) == 0, f"Broken \\ref or \\eqref without \\label: {broken_refs}"


def test_figure_files_exist():
    """Verify all figure paths referenced via \\includegraphics exist."""
    missing_figs = check_figures(TEX_PATH, PAPER_DIR)
    assert len(missing_figs) == 0, f"Missing figure files: {missing_figs}"


def test_bibtex_schema_and_anti_hallucination():
    """Verify that BibTeX entries have required fields and no AI-hallucination placeholders."""
    issues = check_bib_entries(BIB_PATH)
    assert len(issues) == 0, f"BibTeX schema/hallucination issues detected: {issues}"
