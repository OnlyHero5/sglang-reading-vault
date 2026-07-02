#!/usr/bin/env python3
"""
Obsidian vault migration for sglang_reading:
1. Rename generic batch filenames to module-prefixed unique names
2. Inject YAML frontmatter for graph filtering
3. Rewrite markdown links to Obsidian wikilinks
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1] / "sglang_reading"
TODAY = date.today().isoformat()

STANDARD_RENAMES = {
    "README.md": "{mod}-MOC.md",
    "01-核心概念.md": "{mod}-核心概念.md",
    "02-源码走读.md": "{mod}-源码走读.md",
    "03-数据流与交互.md": "{mod}-数据流与交互.md",
    "04-关键问题.md": "{mod}-关键问题.md",
    "checkpoint.md": "{mod}-checkpoint.md",
}

STAGE_MOC_DIRS = [
    "01-启动与入口",
    "02-请求调度",
]

SKIP_DIRS = {"_archive", "_TEMPLATE", ".git", ".obsidian"}

DOC_TYPE_MAP = {
    "MOC": "moc",
    "核心概念": "concept",
    "源码走读": "walkthrough",
    "数据流与交互": "dataflow",
    "关键问题": "faq",
    "checkpoint": "checkpoint",
}

MODULE_SLUG = {
    "00-方法论": "methodology",
    "02-启动链路": "launch",
    "03-HTTP-Server": "http-server",
    "04-OpenAI-API": "openai-api",
    "05-gRPC-Proto": "grpc-proto",
    "06-TokenizerManager": "tokenizer-manager",
    "07-Scheduler": "scheduler",
    "08-SchedulePolicy": "schedule-policy",
    "09-ScheduleBatch-IO": "schedule-batch-io",
    "10-Detokenizer": "detokenizer",
    "11-ModelRunner": "model-runner",
    "12-ModelLoader": "model-loader",
    "13-Models-通用": "models-common",
    "14-Models-专用": "models-specialized",
    "15-RadixAttention": "radix-attention",
    "16-KV-Cache": "kv-cache",
    "17-Attention": "attention",
    "18-MoE": "moe",
    "19-Quantization": "quantization",
    "20-Sampling": "sampling",
    "21-Speculative": "speculative",
    "22-Disaggregation": "disaggregation",
    "23-Distributed": "distributed",
    "24-Multimodal": "multimodal",
    "25-LoRA": "lora",
    "26-sgl-kernel": "sgl-kernel",
    "27-model-gateway": "model-gateway",
    "28-Frontend-lang": "frontend-lang",
    "29-multimodal_gen": "multimodal-gen",
    "31-Observability": "observability",
    "32-CheckpointEngine": "checkpoint-engine",
}

BATCH_NUM = {
    "00-方法论": "01",
    "02-启动链路": "02",
    "03-HTTP-Server": "03",
    "04-OpenAI-API": "04",
    "05-gRPC-Proto": "05",
    "06-TokenizerManager": "06",
    "07-Scheduler": "07",
    "08-SchedulePolicy": "08",
    "09-ScheduleBatch-IO": "09",
    "10-Detokenizer": "10",
    "11-ModelRunner": "11",
    "12-ModelLoader": "12",
    "13-Models-通用": "13",
    "14-Models-专用": "14",
    "15-RadixAttention": "15",
    "16-KV-Cache": "16",
    "17-Attention": "17",
    "18-MoE": "18",
    "19-Quantization": "19",
    "20-Sampling": "20",
    "21-Speculative": "21",
    "22-Disaggregation": "22",
    "23-Distributed": "23",
    "24-Multimodal": "24",
    "25-LoRA": "25",
    "26-sgl-kernel": "26",
    "27-model-gateway": "27",
    "28-Frontend-lang": "28",
    "29-multimodal_gen": "29",
    "31-Observability": "31",
    "32-CheckpointEngine": "32",
}

LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
FRONTMATTER_PATTERN = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
H1_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def batch_module_dirs() -> list[Path]:
    dirs: list[Path] = []
    for d in ROOT.rglob("*"):
        if not d.is_dir():
            continue
        if any(part in SKIP_DIRS for part in d.parts):
            continue
        if (d / "01-核心概念.md").exists():
            dirs.append(d)
    return sorted(dirs)


def stage_moc_files() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for stage in STAGE_MOC_DIRS:
        stage_dir = ROOT / stage
        readme = stage_dir / "README.md"
        if readme.exists():
            pairs.append((readme, stage_dir / f"{stage}-MOC.md"))
    return pairs


def build_rename_plan() -> dict[Path, Path]:
    plan: dict[Path, Path] = {}
    for mod_dir in batch_module_dirs():
        mod = mod_dir.name
        for old_name, pattern in STANDARD_RENAMES.items():
            old_path = mod_dir / old_name
            if old_path.exists():
                new_name = pattern.format(mod=mod)
                plan[old_path] = mod_dir / new_name
    for old_path, new_path in stage_moc_files():
        plan[old_path] = new_path
    return plan


def note_stem(path: Path) -> str:
    return path.stem


def extract_h1(content: str) -> str | None:
    m = H1_PATTERN.search(content)
    return m.group(1).strip() if m else None


def infer_doc_type(stem: str, mod: str) -> str:
    suffix = stem.removeprefix(f"{mod}-")
    return DOC_TYPE_MAP.get(suffix, "note")


def make_frontmatter(path: Path, content: str, mod: str | None = None) -> str:
    rel = path.relative_to(ROOT)
    stem = path.stem
    if mod is None:
        # stage moc or top-level
        if stem.endswith("-MOC") and "/" not in str(rel):
            return (
                f"---\ntype: stage-moc\nmodule: {stem.removesuffix('-MOC')}\n"
                f"title: \"{stem}\"\ntags:\n  - sglang/stage-moc\n"
                f"updated: {TODAY}\n---\n\n"
            )
        return ""

    batch = BATCH_NUM.get(mod, "")
    slug = MODULE_SLUG.get(mod, mod.lower())
    doc_type = infer_doc_type(stem, mod)
    h1 = extract_h1(content) or stem
    note_type = "module-moc" if doc_type == "moc" else "batch-doc"
    old_alias = {
        "concept": "01-核心概念",
        "walkthrough": "02-源码走读",
        "dataflow": "03-数据流与交互",
        "faq": "04-关键问题",
        "checkpoint": "checkpoint",
        "moc": "README",
    }.get(doc_type)

    lines = [
        "---",
        f"type: {note_type}",
        f"module: {mod}",
    ]
    if batch:
        lines.append(f'batch: "{batch}"')
    lines.append(f"doc_type: {doc_type}")
    lines.append(f'title: "{h1}"')
    lines.append("tags:")
    if batch:
        lines.append(f"  - sglang/batch/{batch}")
    lines.append(f"  - sglang/module/{slug}")
    lines.append(f"  - sglang/doc/{doc_type}")
    if old_alias:
        lines.append("aliases:")
        lines.append(f'  - "{old_alias}"')
    lines.append(f"updated: {TODAY}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def strip_frontmatter(content: str) -> str:
    return FRONTMATTER_PATTERN.sub("", content, count=1)


def module_for_dir(mod_dir: Path, planned_names: set[str]) -> str | None:
    name = mod_dir.name
    if name not in BATCH_NUM:
        return None
    if any(n.startswith(f"{name}-") for n in planned_names):
        return name
    if (mod_dir / "01-核心概念.md").exists():
        return name
    return None


def inject_frontmatter(path: Path, content: str, mod: str | None) -> str:
    body = strip_frontmatter(content)

    if mod and mod in BATCH_NUM:
        fm = make_frontmatter(path, body, mod)
    elif path.stem.endswith("-MOC") and path.parent.name in STAGE_MOC_DIRS:
        fm = make_frontmatter(path, body)
    elif path.name == "README.md" and path.parent == ROOT:
        fm = (
            f"---\ntype: index\ntitle: \"SGLang 源码阅读指南\"\n"
            f"tags:\n  - sglang/index\nupdated: {TODAY}\n---\n\n"
        )
    elif path.parent.name == "07-总结与索引" and path.name not in ("README.md",):
        fm = (
            f"---\ntype: index-doc\ntitle: \"{path.stem}\"\n"
            f"tags:\n  - sglang/index-layer\n  - sglang/batch/30\n"
            f"updated: {TODAY}\n---\n\n"
        )
    else:
        return content

    return fm + body.lstrip("\n")


def build_path_map(plan: dict[Path, Path]) -> dict[Path, Path]:
    mapping: dict[Path, Path] = {}
    for old, new in plan.items():
        mapping[old.resolve()] = new.resolve()
    # identity for untouched files
    for md in ROOT.rglob("*.md"):
        if any(part in SKIP_DIRS for part in md.parts):
            continue
        resolved = md.resolve()
        if resolved not in mapping:
            mapping[resolved] = resolved
    return mapping


def resolve_link(current: Path, target: str) -> Path | None:
    target = target.strip()
    if not target or target.startswith(("http://", "https://", "mailto:")):
        return None
    target = target.split("#", 1)[0].split("?", 1)[0]
    if not target or target.startswith("#"):
        return None
    target = unquote(target)
    if target.startswith("/"):
        return None
    return (current.parent / target).resolve()


def to_wikilink(target_path: Path, label: str, path_map: dict[Path, Path]) -> str:
    resolved = path_map.get(target_path.resolve(), target_path.resolve())
    note = note_stem(resolved)
    anchor = ""
    if label == note or not label:
        return f"[[{note}]]"
    return f"[[{note}|{label}]]"


def rewrite_links(content: str, current: Path, path_map: dict[Path, Path]) -> str:
    def repl(match: re.Match[str]) -> str:
        label, raw = match.group(1), match.group(2)
        if raw.startswith(("http://", "https://", "mailto:")):
            return match.group(0)
        parts = raw.split("#", 1)
        path_part = parts[0]
        anchor = parts[1] if len(parts) > 1 else ""
        resolved = resolve_link(current, path_part)
        if resolved is None:
            return match.group(0)
        mapped = path_map.get(resolved)
        if mapped is None:
            return match.group(0)
        note = note_stem(mapped)
        if anchor:
            link = f"[[{note}#{anchor}|{label}]]" if label else f"[[{note}#{anchor}]]"
        else:
            link = f"[[{note}|{label}]]" if label else f"[[{note}]]"
        return link

    return LINK_PATTERN.sub(repl, content)


def apply_renames(plan: dict[Path, Path]) -> None:
    # rename to temp first to avoid collisions
    temp: dict[Path, Path] = {}
    for i, (old, new) in enumerate(plan.items()):
        tmp = old.with_suffix(f".tmp{i}.md")
        old.rename(tmp)
        temp[tmp] = new
    for tmp, new in temp.items():
        new.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(new)


def migrate() -> None:
    plan = build_rename_plan()
    print(f"Planned renames: {len(plan)}")
    path_map = build_path_map(plan)
    planned_names = {p.name for p in plan.values()}

    touched = 0
    for md in sorted(ROOT.rglob("*.md")):
        if any(part in SKIP_DIRS for part in md.parts):
            continue
        original = md.read_text(encoding="utf-8")
        updated = rewrite_links(original, md, path_map)
        new_path = path_map.get(md.resolve(), md.resolve())
        mod = module_for_dir(new_path.parent, planned_names)
        if (
            mod
            or (new_path.stem.endswith("-MOC") and new_path.parent.name in STAGE_MOC_DIRS)
            or (new_path.name == "README.md" and new_path.parent == ROOT)
            or new_path.parent.name == "07-总结与索引"
        ):
            updated = inject_frontmatter(new_path, updated, mod)
        if updated != original:
            md.write_text(updated, encoding="utf-8")
            touched += 1

    apply_renames(plan)
    print(f"Updated {touched} markdown files before rename")
    print("Sample renames:")
    for old, new in sorted(plan.items(), key=lambda x: str(x[0]))[:8]:
        print(f"  {old.relative_to(ROOT)} -> {new.name}")


if __name__ == "__main__":
    migrate()
