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

    # Extract GitHub dollar-backtick inline math ($`...`$) FIRST so it is preserved as math
    dollar_backtick_math = []
    def save_dollar_backtick(match):
        expr = match.group(1)
        dollar_backtick_math.append((expr, match.start()))
        return " MATHDOLLARBACKTICK "

    content_handled = re.sub(r"\$`([^`]+?)`\$", save_dollar_backtick, content_no_code)

    # Strip regular inline code spans
    content_no_inline_code = re.sub(r"`[^`]*?`", "", content_handled)

    # Extract display math ($$ ... $$)
    display_matches = []
    for m in re.finditer(r"\$\$([\s\S]*?)\$\$", content_no_inline_code):
        display_matches.append((m.group(1).strip(), m.start()))

    # Extract inline math ($ ... $)
    inline_matches = list(dollar_backtick_math)
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
            math_exprs = re.findall(r"\$`([^`]+)`\$", stripped) + re.findall(r"(?<!`)\$([^$`]+)\$(?!`)", stripped)
            for m in math_exprs:
                # Raw pipe not allowed; \vert, \mid, \| are allowed
                m_no_escaped = m.replace(r"\|", "").replace(r"\mid", "").replace(r"\vert", "")
                assert "|" not in m_no_escaped, (
                    f"Line {line_num} in {readme_path} has raw pipe in table math: ${m}$"
                )

@pytest.mark.parametrize("readme_path", README_FILES)
def test_no_latex_in_headings(readme_path):
    """GFM rule: Markdown headings (#) must not contain LaTeX math or formulas.
    GitHub's heading processor strips underscores (_ becomes <em>), corrupting KaTeX subscripts,
    and heading math conflicts with URL anchor generation."""
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
            
        if stripped.startswith("#"):
            assert "$" not in stripped, (
                f"Line {line_num} in {readme_path} has LaTeX in heading: '{stripped}'. "
                "Use clean descriptive plain text for headings and define math notations in the text below."
            )

@pytest.mark.parametrize("readme_path", README_FILES)
def test_display_math_block_isolation(readme_path):
    """GFM rule: display math $$ must be on its own line and surrounded by blank lines."""
    with open(readme_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    in_code = False
    in_math_block = False
    for idx, line in enumerate(lines):
        line_num = idx + 1
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
            
        if "$$" in stripped:
            assert stripped == "$$", (
                f"Line {line_num} in {readme_path} has unisolated $$: '{stripped}'. "
                "Display math delimiters '$$' must be on their own line without embedded equation text."
            )
            assert not line.startswith(" "), (
                f"Line {line_num} in {readme_path}: Display math '$$' must start at column 0 without indentation. "
                "GitHub math renderer treats indented $$ (e.g. inside lists) as plain text."
            )
            if not in_math_block:
                # Opening $$
                in_math_block = True
                if idx > 0:
                    assert lines[idx - 1].strip() == "", (
                        f"Line {line_num} in {readme_path}: opening '$$' must be preceded by a blank line."
                    )
            else:
                # Closing $$
                in_math_block = False
                if idx < len(lines) - 1:
                    assert lines[idx + 1].strip() == "", (
                        f"Line {line_num} in {readme_path}: closing '$$' must be followed by a blank line."
                    )
    assert not in_math_block, f"Unclosed display math block in {readme_path}"

@pytest.mark.parametrize("readme_path", README_FILES)
def test_no_broken_pipe_norm_in_math(readme_path):
    """GFM rule:
    1. Do not use \\| for vector norms in math; use \\Vert or \\lVert/\\rVert.
    2. Never use \\_, \\!, \\,, \\{, or \\} inside math (GitHub strips backslashes before punctuation).
    3. Never place inline $...$ directly adjacent to CJK fullwidth punctuation without a space, or '-$' outside math.
    """
    display_matches, inline_matches, content = extract_math_blocks(readme_path)
    all_math = [m[0] for m in display_matches] + [m[0] for m in inline_matches]
    for expr in all_math:
        assert r"\|" not in expr, (
            f"Found raw \\| in math in {readme_path}: '{expr}'. Use \\Vert instead to prevent GFM pipe collision."
        )
        assert r"\_" not in expr, f"Forbidden \\_ in math in {readme_path}: '{expr}'"
        assert r"\!" not in expr, f"Forbidden \\! in math in {readme_path}: '{expr}'"
        assert r"\," not in expr, f"Forbidden \\, in math in {readme_path}: '{expr}'"
        assert r"\{" not in expr, f"Forbidden \\{{ in math in {readme_path}: '{expr}' (use \\lbrace)"
        assert r"\}" not in expr, f"Forbidden \\}} in math in {readme_path}: '{expr}' (use \\rbrace)"

    for idx, line in enumerate(content.splitlines(), 1):
        if line.strip().startswith("```"):
            continue
        assert not re.search(r"[（【「『“‘：，、]\$(?!`)|(?<!`)\$[）】」』”’、，。；：]", line), (
            f"Line {idx} in {readme_path} has inline $ directly adjacent to CJK fullwidth punctuation: '{line}'"
        )
        assert "-$" not in line, (
            f"Line {idx} in {readme_path} has '-$' outside math delimiter: '{line}'"
        )

@pytest.mark.parametrize("readme_path", README_FILES)
def test_no_italic_swallowed_latex(readme_path):
    """GFM rule: underscores in LaTeX subscripts must not be parsed as Markdown italics."""
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        pytest.skip("markdown-it-py not installed")

    md = MarkdownIt()
    with open(readme_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_code = False
    in_display_math = False
    for idx, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if s == "$$":
            in_display_math = not in_display_math
            continue
        if in_display_math or "_" not in line:
            continue
        if s.startswith("|") and s.endswith("|"):
            # Table lines containing math are processed by GitHub's MathJax/KaTeX renderer
            continue

        # Protect properly-formed inline math $...$ so valid LaTeX subscripts aren't falsely swallowed by markdown-it
        line_clean = re.sub(r"\$[^$\n]+\$", "MATH_BLOCK", line)
        if "_" not in line_clean:
            continue

        html = md.render(line_clean)
        ems = re.findall(r"<em>(.*?)</em>", html)
        for em in ems:
            if any(ch in em for ch in ("$", "\\", "{", "}", "^", "=", "<", ">", "+", "-")):
                assert False, f"Line {idx} in {readme_path} has corrupted math swallowed into <em> tag: '{em}'"

