#!/usr/bin/env python3
"""Quality verification script for sglang_reading batches."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["README.md", "01-核心概念.md", "02-源码走读.md", "03-数据流与交互.md", "04-关键问题.md", "checkpoint.md"]
MIN_BLOCKS = 15
MIN_SOURCE_TAGS = 10
FORBIDDEN = re.compile(r"(?:见|详见|参见)\s*`[^`]+\.(?:py|rs|toml|proto)`\s*第?\s*\d")


def count_code_blocks(text: str) -> tuple[int, int]:
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    lines = sum(len(b.splitlines()) for b in blocks)
    return len(blocks), lines


def verify_batch(batch_dir: Path) -> dict:
    issues: list[str] = []
    warnings: list[str] = []
    total_blocks = 0
    total_lines = 0
    source_tags = 0

    for name in REQUIRED:
        p = batch_dir / name
        if not p.exists():
            issues.append(f"缺少文件: {name}")
            continue
        text = p.read_text(encoding="utf-8")
        blocks, lines = count_code_blocks(text)
        total_blocks += blocks
        total_lines += lines
        source_tags += len(re.findall(r"# 来源：", text))
        if name != "checkpoint.md" and blocks == 0:
            issues.append(f"{name} 无内嵌代码块")
        if FORBIDDEN.search(text):
            issues.append(f"{name} 含禁止的纯路径引用（无内嵌代码）")
        if "**Explain" not in text and name not in ("checkpoint.md", "README.md"):
            warnings.append(f"{name} 可能缺少 ETC Explain 段")

    if total_blocks < MIN_BLOCKS:
        issues.append(f"代码块总数 {total_blocks} < {MIN_BLOCKS}")
    if total_lines < 200:
        issues.append(f"代码总行约 {total_lines} < 200")
    if source_tags < MIN_SOURCE_TAGS:
        warnings.append(f"来源注释仅 {source_tags} 处（建议 ≥{MIN_SOURCE_TAGS}）")

    cp = batch_dir / "checkpoint.md"
    if cp.exists() and "- [ ]" in cp.read_text(encoding="utf-8"):
        warnings.append("checkpoint.md 仍有未勾选项")

    return {
        "dir": str(batch_dir.relative_to(ROOT)),
        "files_ok": len([n for n in REQUIRED if (batch_dir / n).exists()]),
        "blocks": total_blocks,
        "lines": total_lines,
        "source_tags": source_tags,
        "issues": issues,
        "warnings": warnings,
        "pass": len(issues) == 0,
    }


def find_batches() -> list[Path]:
    dirs = []
    for p in sorted(ROOT.rglob("checkpoint.md")):
        if "_TEMPLATE" in str(p) or "_qa" in str(p):
            continue
        dirs.append(p.parent)
    return dirs


def main() -> int:
    batches = find_batches()
    if not batches:
        print("未找到任何批次目录")
        return 1

    passed = failed = 0
    for d in batches:
        r = verify_batch(d)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"\n[{status}] {r['dir']}")
        print(f"  文件 {r['files_ok']}/6 | 代码块 {r['blocks']} | 约 {r['lines']} 行 | 来源注释 {r['source_tags']}")
        for i in r["issues"]:
            print(f"  [FAIL] {i}")
        for w in r["warnings"]:
            print(f"  [WARN] {w}")
        if r["pass"]:
            passed += 1
        else:
            failed += 1

    print(f"\n合计: {passed} 通过, {failed} 未通过, 共 {len(batches)} 批")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
