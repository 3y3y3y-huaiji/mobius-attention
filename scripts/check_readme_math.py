
def check_file(filename):
    print(f"=== Checking {filename} ===")
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Check lines
    in_code_block = False
    
    for idx, line in enumerate(lines):
        line_num = idx + 1
        stripped = line.strip()
        
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
            
        # Check display math markers
        if "$$" in line:
            # Check if $$ is on its own line
            if stripped != "$$" and not (stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4):
                print(f"Line {line_num}: Malformed $$ placement: {line.rstrip()}")
            if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
                # Single-line display math
                inner = stripped[2:-2]
                if "\\\\" in inner:
                    print(f"Line {line_num}: Display math with \\\\ on single line (may fail on GitHub): {inner}")
        
        # Check inline math
        # Extract $...$
        pos = 0
        while True:
            start = line.find("$", pos)
            if start == -1:
                break
            if start + 1 < len(line) and line[start + 1] == "$":
                # $$ block
                pos = start + 2
                continue
            if start > 0 and line[start - 1] == "\\":
                # Escaped \$
                pos = start + 1
                continue
            
            end = line.find("$", start + 1)
            if end == -1:
                print(f"Line {line_num}: Unclosed inline math starting at pos {start}: {line[start:].rstrip()}")
                break
            if end + 1 < len(line) and line[end + 1] == "$":
                # $$ marker
                pos = end + 2
                continue
            if line[end - 1] == "\\":
                # Escaped
                pos = end + 1
                continue
                
            inner = line[start + 1:end]
            if len(inner) > 0:
                if inner[0] == " " or inner[-1] == " ":
                    print(f"Line {line_num}: Whitespace inside inline math: '${inner}$'")
                if stripped.startswith("|") and ("|" in inner or "\\mid" in inner):
                    print(f"Line {line_num}: Pipe or \\mid inside table math: '${inner}$'")
                if "\\\\" in inner:
                    print(f"Line {line_num}: Double backslash in inline math: '${inner}$'")
            pos = end + 1

check_file("README.md")
check_file("README_zh.md")
