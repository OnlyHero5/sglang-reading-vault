#!/usr/bin/env node
/**
 * Restore fenced code blocks from sglang/ using # 来源：path Lstart-Lend headers.
 * Also fix YAML list indentation damaged by space collapse.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const VAULT = path.resolve(__dirname, "..");
const READING = path.join(VAULT, "sglang_reading");
const SGLANG = path.join(VAULT, "sglang");
const SKIP = new Set(["_archive", "_TEMPLATE", "_qa", ".git", ".obsidian"]);

const SOURCE_RE =
  /^#\s*来源[：:]\s*(.+?\.\w+)\s+L(\d+)(?:\s*[-–—]\s*L?(\d+))?/;

function readSourceLines(relPath, start, end) {
  const candidates = [
    path.join(SGLANG, relPath),
    path.join(SGLANG, "python", relPath.replace(/^python\//, "")),
    path.join(VAULT, relPath),
  ];
  let file = null;
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      file = c;
      break;
    }
  }
  if (!file) return null;
  const lines = fs.readFileSync(file, "utf8").split("\n");
  const s = Math.max(1, start) - 1;
  const e = end ? end : start;
  return lines.slice(s, e).join("\n");
}

function fixYamlLists(text) {
  return text.replace(
    /^(tags:|aliases:)\n((?: - .+\n)+)/gm,
    (_, key, items) => {
      const fixed = items.replace(/^ - /gm, "  - ");
      return `${key}\n${fixed}`;
    }
  );
}

function restoreCodeBlocks(text) {
  const fence = /```(\w*)\n([\s\S]*?)```/g;
  return text.replace(fence, (full, lang, body) => {
    const lines = body.split("\n");
    const headerIdx = lines.findIndex((l) => SOURCE_RE.test(l.trim()));
    if (headerIdx === -1) return full;

    const m = lines[headerIdx].trim().match(SOURCE_RE);
    if (!m) return full;

    let rel = m[1].replace(/\\/g, "/");
    if (rel.startsWith("python/sglang/")) rel = rel;
    else if (rel.startsWith("sglang/")) rel = "python/" + rel;

    const start = parseInt(m[2], 10);
    const end = m[3] ? parseInt(m[3], 10) : start;
    const snippet = readSourceLines(rel, start, end);
    if (!snippet) return full;

    const headerLines = [];
    for (let i = 0; i <= headerIdx; i++) {
      if (lines[i].trim().startsWith("# 提交版本")) headerLines.push(lines[i]);
      else if (SOURCE_RE.test(lines[i].trim())) headerLines.push(lines[i]);
    }
    if (!headerLines.some((l) => SOURCE_RE.test(l.trim()))) {
      headerLines.push(lines[headerIdx]);
    }

    const versionLine = lines.find((l) => l.trim().startsWith("# 提交版本"));
    const prefix = [lines[headerIdx]];
    if (versionLine && versionLine !== lines[headerIdx]) prefix.push(versionLine);

    return "```" + lang + "\n" + prefix.join("\n") + "\n" + snippet + "\n```";
  });
}

function processFile(filePath) {
  let text = fs.readFileSync(filePath, "utf8");
  const orig = text;
  text = fixYamlLists(text);
  text = restoreCodeBlocks(text);
  if (text !== orig) {
    fs.writeFileSync(filePath, text, "utf8");
    return true;
  }
  return false;
}

function walk(dir) {
  let n = 0;
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP.has(ent.name)) continue;
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) n += walk(full);
    else if (ent.name.endsWith(".md") && processFile(full)) n++;
  }
  return n;
}

const n = walk(READING);
console.log(`Restored code/YAML in ${n} files`);
