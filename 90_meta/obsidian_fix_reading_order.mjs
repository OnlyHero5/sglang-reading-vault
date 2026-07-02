#!/usr/bin/env node
/**
 * Rename batch docs to include reading-order prefix (01-05) and rewrite wikilinks.
 * {mod}-核心概念.md -> {mod}-01-核心概念.md
 * {mod}-MOC.md -> {mod}-00-MOC.md
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const VAULT = path.resolve(__dirname, "..");
const READING = path.join(VAULT, "sglang_reading");
const SKIP = new Set(["_archive", "_TEMPLATE", ".obsidian", ".git"]);

const DOC_ORDER = [
  ["核心概念", "01"],
  ["源码走读", "02"],
  ["数据流与交互", "03"],
  ["关键问题", "04"],
  ["checkpoint", "05"],
];

/** @type {Map<string, string>} old basename -> new basename */
const renames = new Map();

function collectRenames() {
  for (const moc of walkFiles(READING, "*-MOC.md")) {
    if (moc.includes("_archive") || moc.includes("_TEMPLATE")) continue;
    const dir = path.dirname(moc);
    const mod = path.basename(moc, "-MOC.md");
    const oldMoc = `${mod}-MOC`;
    const newMoc = `${mod}-00-MOC`;
    if (oldMoc !== newMoc) renames.set(oldMoc, newMoc);

    for (const [suffix, num] of DOC_ORDER) {
      const oldBase = `${mod}-${suffix}`;
      const newBase = `${mod}-${num}-${suffix}`;
      const oldFile = path.join(dir, `${oldBase}.md`);
      if (fs.existsSync(oldFile) && oldBase !== newBase) {
        renames.set(oldBase, newBase);
      }
    }
  }
}

function walkFiles(root, pattern) {
  const out = [];
  const suffix = pattern.replace("*", "");
  function go(d) {
    for (const ent of fs.readdirSync(d, { withFileTypes: true })) {
      if (SKIP.has(ent.name)) continue;
      const full = path.join(d, ent.name);
      if (ent.isDirectory()) go(full);
      else if (ent.name.endsWith(suffix)) out.push(full);
    }
  }
  go(root);
  return out;
}

function applyFileRenames() {
  for (const moc of walkFiles(READING, "*-MOC.md")) {
    if (moc.includes("_archive") || moc.includes("_TEMPLATE")) continue;
    const dir = path.dirname(moc);
    for (const [oldBase, newBase] of renames) {
      const oldPath = path.join(dir, `${oldBase}.md`);
      const newPath = path.join(dir, `${newBase}.md`);
      if (fs.existsSync(oldPath) && !fs.existsSync(newPath)) {
        fs.renameSync(oldPath, newPath);
        console.log(`renamed ${path.relative(VAULT, oldPath)} -> ${path.basename(newPath)}`);
      }
    }
  }
}

function rewriteText(text) {
  let out = text;
  // Longest keys first to avoid partial replacements
  const sorted = [...renames.entries()].sort((a, b) => b[0].length - a[0].length);
  for (const [oldBase, newBase] of sorted) {
    // [[old|alias]] or [[path/old|alias]]
    out = out.replace(
      new RegExp(`\\[\\[([^\\]|#]*?)${escapeReg(oldBase)}(#([^\\]|]+))?(\\|([^\\]]+))?\\]\\]`, "g"),
      (_, prefix, anchorPart, anchor, aliasPart, alias) => {
        const p = prefix || "";
        if (anchor) {
          return alias ? `[[${p}${newBase}#${anchor}|${alias}]]` : `[[${p}${newBase}#${anchor}]]`;
        }
        return alias ? `[[${p}${newBase}|${alias}]]` : `[[${p}${newBase}]]`;
      }
    );
    // Fix path-style links missing sglang_reading/ -> use unique name only
    out = out.replace(
      new RegExp(
        `\\[\\[(?:sglang_reading/)?[^\\]|]+/${escapeReg(oldBase)}(#([^\\]|]+))?(\\|([^\\]]+))?\\]\\]`,
        "g"
      ),
      (_, anchorPart, anchor, aliasPart, alias) => {
        if (anchor) {
          return alias ? `[[${newBase}#${anchor}|${alias}]]` : `[[${newBase}#${anchor}]]`;
        }
        return alias ? `[[${newBase}|${alias}]]` : `[[${newBase}]]`;
      }
    );
  }
  // Strip prohibited troubleshooting block (HTTP Server legacy)
  out = out.replace(
    />\s*若 Obsidian 仍显示「未创建」[^\n]*\n\n/g,
    ""
  );
  out = out.replace(/## 文档导航（兼容短链）\n\n[\s\S]*?(?=\n## )/g, "");
  return out;
}

function escapeReg(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function rewriteAllMarkdown() {
  let count = 0;
  function go(d) {
    for (const ent of fs.readdirSync(d, { withFileTypes: true })) {
      if (SKIP.has(ent.name)) continue;
      const full = path.join(d, ent.name);
      if (ent.isDirectory()) go(full);
      else if (ent.name.endsWith(".md")) {
        const text = fs.readFileSync(full, "utf8");
        const newText = rewriteText(text);
        if (newText !== text) {
          fs.writeFileSync(full, newText, "utf8");
          count++;
        }
      }
    }
  }
  go(READING);
  // Also fix AGENTS-adjacent vault root docs if any links
  for (const f of ["AGENTS.md"]) {
    const p = path.join(VAULT, f);
    if (fs.existsSync(p)) {
      const text = fs.readFileSync(p, "utf8");
      const newText = rewriteText(text);
      if (newText !== text) {
        fs.writeFileSync(p, newText, "utf8");
        count++;
      }
    }
  }
  return count;
}

collectRenames();
console.log(`Planned renames: ${renames.size}`);
if (process.argv.includes("--dry-run")) {
  for (const [a, b] of [...renames.entries()].sort()) console.log(`  ${a} -> ${b}`);
  process.exit(0);
}

applyFileRenames();
const n = rewriteAllMarkdown();
console.log(`Updated ${n} markdown files`);
