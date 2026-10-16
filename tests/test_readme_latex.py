"""
Automated LaTeX & KaTeX Syntax Validation Suite for GitHub READMEs.
Verifies that all inline and display math expressions in README.md and README_zh.md
strictly conform to GitHub Flavored Markdown (GFM) and KaTeX rendering specifications.
"""

import os
import re
import pytest

README_FILES = ["README.md", "README_zh.md"]

def extract_math_blocks(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip code blocks
    content_no_code = re.sub(r"```[\s\S]*?```", "", content)
    content_no_inline_code = re.sub(r"`[^`]*?`", "", content_no_code)

    # Extract display math ($$ ... $$)
    display_matches = []
    for m in re.finditer(r"\$\$([\s\S]*?)\$\$", content_no_inline_code):
        display_matches.append((m.group(1).strip(), m.start()))

    # Extract inline math ($ ... $)
    inline_matches = []
    text_inline = re.sub(r"\$\$[\s\S]*?\$\$", "", content_no_inline_code)
    for m in re.finditer(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", text_inline):
        inline_matches.append((m.group(1), m.start()))

    return display_matches, inline_matches, content

@pytest.mark.parametrize("readme_path", README_FILES)
def test_inline_math_no_edge_whitespace(readme_path):
    """GFM rule: inline math $...$ must not start or end with a space."""
    assert os.path.exists(readme_path), f"{readme_path} not found"
    _, inline_matches, _ = extract_math_blocks(readme_path)
    
    violations = []
    for expr, pos in inline_matches:
        if len(expr) > 0 and (expr[0] == " " or expr[-1] == " "):
            violations.append(expr)
    
    assert not violations, f"Found inline math with edge whitespace in {readme_path}: {violations}"

@pytest.mark.parametrize("readme_path", README_FILES)
def test_math_balanced_braces_and_environments(readme_path):
    """Every math expression must have balanced curly braces and begin/end environments."""
    display_matches, inline_matches, _ = extract_math_blocks(readme_path)
    all_math = [m[0] for m in display_matches] + [m[0] for m in inline_matches]
    
    for expr in all_math:
        cleaned = expr.replace(r"\{", "").replace(r"\}", "")
        open_braces = cleaned.count("{")
        close_braces = cleaned.count("}")
        assert open_braces == close_braces, f"Unbalanced braces in {readme_path}: '{expr}' ({open_braces} != {close_braces})"
        
        begins = re.findall(r"\\begin\{([a-zA-Z0-9*]+)\}", expr)
        ends = re.findall(r"\\end\{([a-zA-Z0-9*]+)\}", expr)
        assert begins == ends, f"Mismatched environments in {readme_path}: begins={begins}, ends={ends}"

@pytest.mark.parametrize("readme_path", README_FILES)
def test_no_raw_pipe_in_table_math(readme_path):
    """GFM tables break if table math expressions use unescaped raw pipe |."""
    with open(readme_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for idx, line in enumerate(lines):
        line_num = idx + 1
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            math_exprs = re.findall(r"\$([^$]+)\$", stripped)
            for m in math_exprs:
                assert "|" not in m or r"\|" in m, (
                    f"Line {line_num} in {readme_path} has raw pipe in table math: ${m}$"
                )

@pytest.mark.parametrize("readme_path", README_FILES)
def test_display_math_block_isolation(readme_path):
    """GFM block math $$ should be properly delimited without inline text collision."""
    with open(readme_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    in_code = False
    for idx, line in enumerate(lines):
        line_num = idx + 1
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
            
        if "$$" in stripped:
            if not (stripped == "$$" or (stripped.startswith("$$") and stripped.endswith("$$"))):
                assert False, f"Line {line_num} in {readme_path} has unisolated $$: {stripped}"

@pytest.mark.parametrize("readme_path", README_FILES)
def test_no_broken_pipe_norm_in_inline_math(readme_path):
    """GFM rule: do not use \\| for vector norms in inline math; use \\Vert or \\lVert/\\rVert."""
    _, inline_matches, _ = extract_math_blocks(readme_path)
    for expr, _ in inline_matches:
        assert r"\|" not in expr, (
            f"Found raw \\| in inline math in {readme_path}: '${expr}$'. Use \\Vert instead to prevent GFM pipe collision."
        )
