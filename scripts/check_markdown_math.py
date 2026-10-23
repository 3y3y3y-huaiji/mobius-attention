import re
import sys

from markdown_it import MarkdownIt

sys.stdout.reconfigure(encoding='utf-8')
md = MarkdownIt()

def audit_file(fname):
    print("\n==========================================")
    print(f"AUDITING FILE: {fname}")
    print("==========================================")
    
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_code = False
    in_display_math = False
    
    corrupted_lines = []
    
    for idx, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            continue
            
        if s == '$$':
            in_display_math = not in_display_math
            continue
        if in_display_math:
            if r'\_' in line or r'\!' in line or r'\,' in line:
                corrupted_lines.append((idx, line.strip(), line.strip(), "Forbidden escaped char in display math"))
                print(f"[CORRUPTED DISPLAY MATH ESCAPE] Line {idx}: {line.strip()}")
            continue

        if s.startswith("|") and s.endswith("|"):
            # Table lines containing math are processed by GitHub's MathJax/KaTeX renderer
            continue

        # Check all inline math spans on this line for forbidden escapes (\_, \!, \,, \{, \}),
        # or CJK fullwidth punctuation immediately adjacent to $ delimiters
        math_spans = re.findall(r'\$`[^`\n]+`\$|\$[^$\n]+\$', line)
        for ms in math_spans:
            if any(esc in ms for esc in (r'\_', r'\!', r'\,', r'\{', r'\}')):
                corrupted_lines.append((idx, line.strip(), ms, "Forbidden escaped char inside math"))
                print(f"[CORRUPTED MATH ESCAPE] Line {idx}: {ms}")
                break
        if re.search(r'[（【「『“‘：，、]\$(?!`)|(?<!`)\$[）】」』”’、，。；：]', line):
            corrupted_lines.append((idx, line.strip(), line.strip(), "Inline $ adjacent to CJK punctuation"))
            print(f"[CJK PUNCTUATION ADJACENT TO $] Line {idx}: {line.strip()}")

        # If there are no underscores, it cannot trigger the italic eating bug
        if '_' not in line:
            continue

        line_clean = re.sub(r'\$`[^`\n]+`\$|\$[^$\n]+\$', 'MATH_BLOCK', line)
        if '_' not in line_clean:
            continue

        # Render the remaining non-math text with CommonMark parser
        html = md.render(line_clean)
        
        # Check if any <em> tag was produced
        # If an <em> tag contains any LaTeX characters or is inside math, it's corrupted!
        ems = re.findall(r'<em>(.*?)</em>', html)
        for em in ems:
            if any(ch in em for ch in ('$', '\\', '{', '}', '^', '=', '<', '>', '+', '-')):
                # Is there a dollar sign in this line?
                if '$' in line:
                    corrupted_lines.append((idx, line.strip(), em, html.strip()))
                    print(f"[CORRUPTED MATH] Line {idx}:")
                    print(f"  Source:     {line.strip()}")
                    print(f"  Swallowed:  {repr(em)}")
                    print(f"  HTML:       {html.strip()}\n")
                    break
                    
    print(f"Found {len(corrupted_lines)} corrupted math lines in {fname}.\n")
    return corrupted_lines

if __name__ == '__main__':
    v1 = audit_file('README.md')
    v2 = audit_file('README_zh.md')
    if len(v1) > 0 or len(v2) > 0:
        print("ERROR: Math corruption detected in documentation!")
        sys.exit(1)
    else:
        print("SUCCESS: All Markdown math syntax verified intact.")
        sys.exit(0)
