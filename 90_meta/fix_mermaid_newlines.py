#!/usr/bin/env python3
"""Replace literal \\n with <br/> inside ```mermaid fenced blocks only."""

from __future__ import annotations

import re
import sys
from pathlib import Path

MERMAID_BLOCK = re.compile(r"(```mermaid\n)(.*?)(```)", re.DOTALL)


def fix_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    replacements = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal replacements
        prefix, body, suffix = match.group(1), match.group(2), match.group(3)
        if "\\n" not in body:
            return match.group(0)
        count = body.count("\\n")
        replacements += count
        return prefix + body.replace("\\n", "<br/>") + suffix

    new_text = MERMAID_BLOCK.sub(repl, text)
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
    return replacements


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "sglang_reading"
    if not root.is_dir():
        print(f"Missing directory: {root}", file=sys.stderr)
        return 1

    changed: list[tuple[str, int]] = []
    total = 0
    for md in sorted(root.rglob("*.md")):
        count = fix_file(md)
        if count:
            changed.append((str(md.relative_to(root.parent)), count))
            total += count

    print(f"Changed {len(changed)} files, {total} replacements")
    for rel, count in changed:
        print(f"  {count:3d}  {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
