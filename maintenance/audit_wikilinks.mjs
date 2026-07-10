#!/usr/bin/env node
/**
 * Audit Obsidian wikilinks against actual note files.
 * Vault root = parent of maintenance.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const VAULT = path.resolve(__dirname, "..");
const SGLANG_READING = path.join(VAULT, "sglang_reading");
const SLIME_READING = path.join(VAULT, "slime_reading");
const FLASH_ATTN_READING = path.join(VAULT, "flash-attn_reading");
const AI_INFRA_COURSE = path.join(VAULT, "AI-Infra课程");
const SKIP = new Set(["模板", ".obsidian", ".git", "sglang", "slime", "flash-attn"]);

/** @type {Map<string, string[]>} basename -> full paths relative to VAULT */
const notesByBase = new Map();

function addNoteName(name, rel) {
  if (!notesByBase.has(name)) notesByBase.set(name, []);
  const hits = notesByBase.get(name);
  if (!hits.includes(rel)) hits.push(rel);
}

function indexMarkdown(filePath) {
  const rel = path.relative(VAULT, filePath).replace(/\\/g, "/");
  const base = path.basename(filePath, ".md");
  addNoteName(base, rel);

  const text = fs.readFileSync(filePath, "utf8");
  const fm = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!fm) return;
  const aliases = [];
  let inAliases = false;
  for (const rawLine of fm[1].split("\n")) {
    const line = rawLine.replace(/\r$/, "");
    if (/^aliases:\s*$/.test(line)) {
      inAliases = true;
      continue;
    }
    if (inAliases) {
      const item = line.match(/^\s+-\s+(.+)$/);
      if (item) {
        aliases.push(item[1].replace(/^["']|["']$/g, ""));
        continue;
      }
      if (/^\S/.test(line)) inAliases = false;
    }
  }
  for (const alias of aliases) {
    if (!alias) continue;
    addNoteName(alias, rel);
  }
}

function indexBase(filePath) {
  const rel = path.relative(VAULT, filePath).replace(/\\/g, "/");
  addNoteName(path.basename(filePath), rel);
}

function walkIndex(dir) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP.has(ent.name)) continue;
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) walkIndex(full);
    else if (ent.name.endsWith(".md")) indexMarkdown(full);
    else if (ent.name.endsWith(".base")) indexBase(full);
  }
}

function indexVaultNotes() {
  walkIndex(SGLANG_READING);
  walkIndex(SLIME_READING);
  if (fs.existsSync(FLASH_ATTN_READING)) walkIndex(FLASH_ATTN_READING);
  if (fs.existsSync(AI_INFRA_COURSE)) walkIndex(AI_INFRA_COURSE);

  for (const name of ["index.md", "README.md", "AGENTS.md"]) {
    const full = path.join(VAULT, name);
    if (fs.existsSync(full)) indexMarkdown(full);
  }

  const dashboard = path.join(VAULT, "knowledge_maps");
  if (fs.existsSync(dashboard)) walkIndex(dashboard);

  const metaDir = path.join(VAULT, "maintenance");
  if (fs.existsSync(metaDir)) {
    for (const ent of fs.readdirSync(metaDir, { withFileTypes: true })) {
      if (ent.isFile() && ent.name.endsWith(".md")) {
        indexMarkdown(path.join(metaDir, ent.name));
      }
    }
  }
}

indexVaultNotes();

const WIKI = /\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|([^\]]+))?\]\]/g;

/** @type {Map<string, {file:string, line:number, target:string, anchor?:string}[]>} */
const broken = new Map();
let totalLinks = 0;

function resolveTarget(target, sourceRel) {
  const t = target.trim();
  // Path-style link (contains /)
  if (t.includes("/")) {
    const candidates = [
      t,
      `${t}.md`,
      `sglang_reading/${t}.md`,
      `slime_reading/${t}.md`,
      `flash-attn_reading/${t}.md`,
      `AI-Infra课程/${t}.md`,
      `knowledge_maps/${t}.md`,
      `maintenance/${t}.md`,
    ];
    for (const c of candidates) {
      const full = path.join(VAULT, c);
      if (fs.existsSync(full)) return { ok: true, path: c.replace(/\\/g, "/") };
    }
    return { ok: false, reason: "path-not-found", candidates };
  }
  const hits = notesByBase.get(t) || [];
  if (hits.length === 1) return { ok: true, path: hits[0] };
  if (hits.length > 1) {
    const sourceDir = path.posix.dirname(sourceRel.replace(/\\/g, "/"));
    const local = hits.filter((h) => path.posix.dirname(h) === sourceDir);
    if (local.length === 1) return { ok: true, path: local[0] };
    return { ok: false, reason: "ambiguous", hits };
  }
  return { ok: false, reason: "missing" };
}

/** Return substrings of line that are outside inline `code` spans. */
function segmentsOutsideInlineCode(line) {
  const segments = [];
  let i = 0;
  while (i < line.length) {
    if (line[i] === "`") {
      let j = i + 1;
      while (j < line.length && line[j] !== "`") j++;
      i = j < line.length ? j + 1 : line.length;
      continue;
    }
    let j = i;
    while (j < line.length && line[j] !== "`") j++;
    if (j > i) segments.push(line.slice(i, j));
    i = j;
  }
  return segments;
}

function scanWikilinksInText(text, rel, lineNo) {
  let m;
  WIKI.lastIndex = 0;
  while ((m = WIKI.exec(text)) !== null) {
    totalLinks++;
    const target = m[1];
    const r = resolveTarget(target, rel);
    if (!r.ok) {
      const key = `${r.reason}:${target}`;
      if (!broken.has(key)) broken.set(key, []);
      broken.get(key).push({ file: rel, line: lineNo, target, reason: r.reason, detail: r });
    }
  }
}

function scanFile(filePath) {
  const rel = path.relative(VAULT, filePath).replace(/\\/g, "/");
  const text = fs.readFileSync(filePath, "utf8");
  const lines = text.split("\n");
  let inCodeBlock = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      inCodeBlock = !inCodeBlock;
      continue;
    }
    if (inCodeBlock) continue;

    for (const segment of segmentsOutsideInlineCode(line)) {
      scanWikilinksInText(segment, rel, i + 1);
    }
  }
}

function scanDir(dir) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP.has(ent.name)) continue;
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) scanDir(full);
    else if (ent.name.endsWith(".md")) scanFile(full);
  }
}

function scanVault() {
  scanDir(SGLANG_READING);
  scanDir(SLIME_READING);
  if (fs.existsSync(FLASH_ATTN_READING)) scanDir(FLASH_ATTN_READING);
  if (fs.existsSync(AI_INFRA_COURSE)) scanDir(AI_INFRA_COURSE);

  for (const name of ["index.md", "README.md", "AGENTS.md"]) {
    const full = path.join(VAULT, name);
    if (fs.existsSync(full)) scanFile(full);
  }

  const dashboard = path.join(VAULT, "knowledge_maps");
  if (fs.existsSync(dashboard)) scanDir(dashboard);

  const metaDir = path.join(VAULT, "maintenance");
  if (fs.existsSync(metaDir)) {
    for (const ent of fs.readdirSync(metaDir, { withFileTypes: true })) {
      if (ent.isFile() && ent.name.endsWith(".md")) {
        scanFile(path.join(metaDir, ent.name));
      }
    }
  }
}

scanVault();

console.log(`Notes indexed: ${[...notesByBase.values()].reduce((a, b) => a + b.length, 0)}`);
console.log(`Wikilinks scanned: ${totalLinks}`);
console.log(`Broken unique targets: ${broken.size}\n`);

for (const [key, hits] of [...broken.entries()].sort()) {
  const [reason] = key.split(":");
  console.log(`=== ${key} (${hits.length} refs) ===`);
  for (const h of hits.slice(0, 5)) {
    console.log(`  ${h.file}:${h.line}  [[${h.target}]]`);
  }
  if (hits.length > 5) console.log(`  ... +${hits.length - 5} more`);
  if (hits[0]?.detail?.candidates) {
    console.log(`  tried: ${hits[0].detail.candidates.join(", ")}`);
  }
  if (hits[0]?.detail?.hits) {
    console.log(`  ambiguous paths: ${hits[0].detail.hits.join(", ")}`);
  }
}

const duplicateBasenames = [...notesByBase.entries()].filter(([, hits]) => hits.length > 1);
console.log(`Duplicate note names: ${duplicateBasenames.length}`);
for (const [name, hits] of duplicateBasenames.slice(0, 20)) {
  console.log(`  ${name}: ${hits.join(", ")}`);
}
