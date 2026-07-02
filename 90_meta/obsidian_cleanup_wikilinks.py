#!/usr/bin/env python3
"""Clean wikilink display aliases and fix directory links to module MOCs."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "sglang_reading"
VAULT = ROOT.parent

# [[target|`]] — broken partial replacement
BROKEN_ALIAS = re.compile(r"\[\[([^|\]#]+)(?:#([^\]|]+))?\|`\]\]")

# [[target|`filename.md`]]
BACKTICK_MD_ALIAS = re.compile(
    r"\[\[([^|\]#]+)(?:#([^\]|]+))?\|`([^`]+\.md)`\]\]"
)

# [[target|filename.md]] or [[target|filename.md §N]] (no backticks)
PLAIN_MD_ALIAS = re.compile(
    r"\[\[([^|\]#]+)(?:#([^\]|]+))?\|([^|\]`]+\.md(?:[^|\]]+)?)\]\]"
)

# [[target|alias]] where alias is batch doc path like 01-核心概念.md
BATCH_MD_ALIAS = re.compile(
    r"\[\[([^|\]#]+)(?:#([^\]|]+))?\|(?:[^/)]+/)?"
    r"(0[1-4]-[^|\]]+|checkpoint|README)\.md(?:[^|\]]+)?\]\]"
)

# [label](../path/ModuleName/) or [label](../path/ModuleName)
DIR_LINK = re.compile(
    r"\[([^\]]*)\]\((?:\.\./|\./)+(?:[^/)]+/)*([^/)]+)/?\)"
)

RENAMES = {
    ROOT / "README.md": ROOT / "SGLang源码阅读指南.md",
    ROOT / "07-总结与索引" / "README.md": ROOT / "07-总结与索引" / "07-总结与索引-00-MOC.md",
}


def _strip_md_alias(alias_body: str) -> str:
    """Return readable suffix after removing filename.md prefix from alias."""
    return re.sub(r"^(?:[^/]+/)?[\w\-]+\.md", "", alias_body).strip()


def _to_note(m: re.Match[str], keep_suffix: bool = False) -> str:
    note, anchor = m.group(1), m.group(2)
    if keep_suffix and m.lastindex and m.lastindex >= 3:
        suffix = _strip_md_alias(m.group(3))
        if anchor:
            return f"[[{note}#{anchor}]]"
        if suffix:
            return f"[[{note}|{suffix}]]"
    if anchor:
        return f"[[{note}#{anchor}]]"
    return f"[[{note}]]"


def clean_aliases(text: str) -> str:
    text = BROKEN_ALIAS.sub(lambda m: _to_note(m), text)
    text = BACKTICK_MD_ALIAS.sub(lambda m: _to_note(m), text)
    text = BATCH_MD_ALIAS.sub(lambda m: _to_note(m), text)
    text = PLAIN_MD_ALIAS.sub(lambda m: _to_note(m, keep_suffix=True), text)
    return text


def fix_dir_links(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        label, mod = m.group(1), m.group(2)
        note = f"{mod}-00-MOC"
        if label and label != mod:
            return f"[[{note}|{label}]]"
        return f"[[{note}]]"

    return DIR_LINK.sub(repl, text)


def process(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    new = clean_aliases(text)
    new = fix_dir_links(new)
    if new != text:
        path.write_text(new, encoding="utf-8")
        return True
    return False


def apply_renames() -> None:
    for old, new in RENAMES.items():
        if old.exists():
            old.rename(new)
            print(f"renamed {old.name} -> {new.name}")


def main() -> None:
    apply_renames()
    count = 0
    sglang_upstream = VAULT / "sglang"
    for base in (ROOT, VAULT):
        for md in base.rglob("*.md"):
            if "_TEMPLATE" in md.parts:
                continue
            if sglang_upstream in md.parents:
                continue
            if process(md):
                count += 1
    print(f"cleaned {count} files")


if __name__ == "__main__":
    main()
