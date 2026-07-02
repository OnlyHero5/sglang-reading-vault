#!/usr/bin/env python3
"""Fix remaining old-style batch file path links after rename migration."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "sglang_reading"

SUFFIX_MAP = {
    "01-核心概念.md": "核心概念",
    "02-源码走读.md": "源码走读",
    "03-数据流与交互.md": "数据流与交互",
    "04-关键问题.md": "关键问题",
    "checkpoint.md": "checkpoint",
    "README.md": "MOC",
}

OLD_PATH = re.compile(
    r"\[([^\]]*)\]\((?:\.\./)*(?:[^/)]+/)*([^/)]+)/(0[1-4]-[^/)]+|checkpoint|README)\.md(?:#([^)]+))?\)"
)
FULL_REL = re.compile(
    r"\[([^\]]*)\]\((\.\./[^)]*(?:0[1-4]-[^/)]+|checkpoint|README)\.md(?:#([^)]+))?)\)"
)


def module_from_tail(mod: str, tail: str) -> str:
    suffix = SUFFIX_MAP.get(tail)
    return f"{mod}-{suffix}" if suffix else ""


def repl_old_path(m: re.Match[str]) -> str:
    label, mod, tail_base, anchor = m.group(1), m.group(2), m.group(3), m.group(4)
    tail = "README.md" if tail_base == "README" else f"{tail_base}.md"
    note = module_from_tail(mod, tail)
    if not note:
        return m.group(0)
    if anchor:
        return f"[[{note}#{anchor}|{label or note}]]"
    return f"[[{note}|{label}]]" if label else f"[[{note}]]"


def repl_full_rel(m: re.Match[str]) -> str:
    label, path, anchor = m.group(1), m.group(2), m.group(3)
    parts = Path(path.replace("\\", "/")).parts
    if len(parts) < 2:
        return m.group(0)
    note = module_from_tail(parts[-2], parts[-1])
    if not note:
        return m.group(0)
    if anchor:
        return f"[[{note}#{anchor}|{label or note}]]"
    return f"[[{note}|{label}]]" if label else f"[[{note}]]"


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    new = OLD_PATH.sub(repl_old_path, text)
    new = FULL_REL.sub(repl_full_rel, new)
    if new != text:
        path.write_text(new, encoding="utf-8")
        return True
    return False


def main() -> None:
    count = 0
    for md in ROOT.rglob("*.md"):
        if "_archive" in md.parts or "_TEMPLATE" in md.parts:
            continue
        if fix_file(md):
            count += 1
            print(md.relative_to(ROOT))
    vault = ROOT.parent
    for name in ("index.md", "AGENTS.md", "90_meta/obsidian-syntax-rules.md"):
        p = vault / name
        if p.exists() and fix_file(p):
            count += 1
            print(name)
    print(f"fixed {count} files")


if __name__ == "__main__":
    main()
