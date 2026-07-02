#!/usr/bin/env python3
"""Audit MOC files for missing documents."""
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "sglang_reading"
doc_types = ["MOC", "核心概念", "源码走读", "数据流与交互", "关键问题", "checkpoint"]

batch_dirs = []
for moc in root.rglob("*-MOC.md"):
    if "_TEMPLATE" in str(moc) or "_archive" in str(moc):
        continue
    batch_dirs.append(moc.parent)

results_missing = []
results_stubs = []

for batch_dir in sorted(set(batch_dirs)):
    mocs = list(batch_dir.glob("*-MOC.md"))
    if not mocs:
        continue
    moc = mocs[0]
    prefix = moc.stem.replace("-MOC", "")
    actual = {f.name for f in batch_dir.glob("*.md")}

    missing = []
    for dt in doc_types:
        fname = f"{prefix}-{dt}.md"
        if fname not in actual:
            missing.append(fname)

    stubs = []
    for f in batch_dir.glob("*.md"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        if len(text.strip()) < 500:
            stubs.append((f.name, len(text)))
        elif any(p in text for p in ["（模块职责）", "待补充", "占位", "TODO:"]):
            stubs.append((f.name, "placeholder"))

    if missing:
        results_missing.append((prefix, str(batch_dir.relative_to(root)), missing, sorted(actual)))
    if stubs:
        results_stubs.append((prefix, stubs))

print("=== MISSING DOCS ===")
for prefix, dirpath, missing, have in results_missing:
    print(f"\n{prefix} ({dirpath})")
    print(f"  missing: {missing}")
    print(f"  have: {have}")

print("\n=== STUBS/THIN (<500 chars or placeholder) ===")
for prefix, stubs in results_stubs:
    print(f"{prefix}: {stubs}")

print(f"\nTotal batch dirs: {len(set(batch_dirs))}")
print(f"Batches with missing docs: {len(results_missing)}")
print(f"Batches with stubs: {len(results_stubs)}")
