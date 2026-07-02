#!/usr/bin/env node
/** Force-restore all # 来源： code blocks from sglang/ */
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

function readSourceLines(relPath, start, endIncl) {
  const rel = relPath.replace(/\\/g, "/").trim();
  const tries = [
    path.join(SGLANG, rel),
    path.join(SGLANG, rel.replace(/^python\//, "")),
    path.join(VAULT, rel),
  ];
  const file = tries.find((p) => fs.existsSync(p));
  if (!file) return { snippet: null, file: tries[0] };
  const lines = fs.readFileSync(file, "utf8").split("\n");
  const s = start - 1;
  const e = endIncl ?? start;
  return { snippet: lines.slice(s, e).join("\n"), file };
}

function restore(text) {
  let blocks = 0;
  const out = text.replace(/```(\w*)\r?\n([\s\S]*?)```/g, (full, lang, body) => {
    const headerLine = body.split("\n").find((l) => SOURCE_RE.test(l.trim()));
    if (!headerLine) return full;
    const m = headerLine.trim().match(SOURCE_RE);
    if (!m) return full;
    const end = m[3] ? parseInt(m[3], 10) : parseInt(m[2], 10);
    const { snippet, file } = readSourceLines(m[1], parseInt(m[2], 10), end);
    if (!snippet) {
      console.warn("MISSING", m[1], "->", file);
      return full;
    }
    const versionLine = body.split("\n").find((l) => l.trim().startsWith("# 提交版本"));
    const head = [headerLine.trim()];
    if (versionLine) head.push(versionLine.trim());
    blocks++;
    return "```" + lang + "\n" + head.join("\n") + "\n" + snippet + "\n```";
  });
  return { out, blocks };
}

function walk(dir) {
  let files = 0;
  let blocks = 0;
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP.has(ent.name)) continue;
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      const r = walk(full);
      files += r.files;
      blocks += r.blocks;
    } else if (ent.name.endsWith(".md")) {
      const text = fs.readFileSync(full, "utf8");
      const { out, blocks: b } = restore(text);
      if (out !== text) {
        fs.writeFileSync(full, out, "utf8");
        files++;
        blocks += b;
      }
    }
  }
  return { files, blocks };
}

const r = walk(READING);
console.log(`Restored ${r.blocks} code blocks in ${r.files} files`);
