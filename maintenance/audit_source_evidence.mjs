#!/usr/bin/env node
/**
 * Audit embedded source citations in reading notes.
 *
 * This script is intentionally read-only. It checks that every "来源：..."
 * citation points to an upstream file and, when a line range is present, that
 * the range is inside the current source baseline.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const VAULT = path.resolve(__dirname, "..");

const ROOTS = [
  {
    reading: "sglang_reading",
    source: "sglang",
    tag: "sglang",
  },
  {
    reading: "slime_reading",
    source: "slime",
    tag: "slime",
  },
  {
    reading: "flash-attn_reading",
    source: path.join("flash-attn", "flash-attention"),
    tag: "flash-attn",
  },
];

const SOURCE_RE =
  /来源：\s*([^`#*<>\r\n]+?)(?:\s+L(\d+)(?:\s*[-–]\s*L?(\d+))?)?(?=\s*(?:\r?\n|$|[（(]|`|\*))/gu;

const SOURCE_CARD_RE =
  /来源：\s*`?([^`\s>]+)`?\s+L(\d+)(?:\s*[-–]\s*L?(\d+))?/u;

function walkMarkdown(dir, out = []) {
  if (!fs.existsSync(dir)) return out;
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (ent.name === "模板") continue;
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) walkMarkdown(full, out);
    else if (ent.name.endsWith(".md")) out.push(full);
  }
  return out;
}

function normalizeSourcePath(raw) {
  return raw
    .trim()
    .replace(/^["'`]|["'`]$/g, "")
    .replace(/\\/g, "/")
    .replace(/\s+$/g, "");
}

function countLines(filePath) {
  const text = fs.readFileSync(filePath, "utf8");
  if (text.length === 0) return 0;
  return text.split(/\r?\n/).length;
}

function lineOfOffset(text, offset) {
  let line = 1;
  for (let i = 0; i < offset; i++) {
    if (text.charCodeAt(i) === 10) line++;
  }
  return line;
}

function normalizeExcerpt(lines) {
  const copy = [...lines];
  while (copy.length > 0 && copy[0].trim() === "") copy.shift();
  while (copy.length > 0 && copy.at(-1).trim() === "") copy.pop();
  const indents = copy
    .filter((line) => line.trim() !== "")
    .map((line) => line.match(/^\s*/u)[0].length);
  const commonIndent = indents.length > 0 ? Math.min(...indents) : 0;
  return copy
    .map((line) => line.slice(commonIndent).replace(/[ \t]+$/u, ""))
    .join("\n");
}

function scanSourceCards(text, root, relNote) {
  const lines = text.split(/\r?\n/u);
  const cards = [];

  for (let i = 0; i < lines.length; i++) {
    const opening = lines[i].match(/^\s*(`{3,}|~{3,})/u);
    if (!opening) continue;

    const marker = opening[1][0];
    const markerLength = opening[1].length;
    const body = [];
    let end = i + 1;
    for (; end < lines.length; end++) {
      const closing = lines[end].match(/^\s*(`{3,}|~{3,})\s*$/u);
      if (
        closing &&
        closing[1][0] === marker &&
        closing[1].length >= markerLength
      ) {
        break;
      }
      body.push(lines[end]);
    }

    const firstLine = body[0]?.trim() ?? "";
    const sourceMatch = firstLine.match(SOURCE_CARD_RE);
    if (sourceMatch) {
      const sourceRel = normalizeSourcePath(sourceMatch[1]);
      const start = Number(sourceMatch[2]);
      const finish = sourceMatch[3] ? Number(sourceMatch[3]) : start;
      const resolved = resolveSource(root, sourceRel);
      const exists = fs.existsSync(resolved.abs);
      let exact = false;
      if (exists) {
        const sourceLines = fs
          .readFileSync(resolved.abs, "utf8")
          .split(/\r?\n/u)
          .slice(start - 1, finish);
        exact = normalizeExcerpt(body.slice(1)) === normalizeExcerpt(sourceLines);
      }
      cards.push({
        note: relNote,
        noteLine: i + 2,
        source: resolved.rel,
        sourceRoot: root.source.replace(/\\/g, "/"),
        start,
        end: finish,
        exists,
        exact,
      });
    }

    i = end;
  }

  return cards;
}

function getDocType(text) {
  const m = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!m) return "";
  const dt = m[1].match(/^type:\s*(.+)$/m);
  return dt ? dt[1].trim() : "";
}

function resolveSource(root, sourceRel) {
  const candidates = [sourceRel];
  if (sourceRel.startsWith(`${root.tag}/`)) {
    candidates.push(sourceRel.slice(root.tag.length + 1));
  }
  if (root.tag === "flash-attn" && sourceRel.startsWith("flash-attention/")) {
    candidates.push(sourceRel.slice("flash-attention/".length));
  }

  for (const rel of candidates) {
    const abs = path.join(VAULT, root.source, rel);
    if (fs.existsSync(abs)) return { abs, rel };
  }
  return { abs: path.join(VAULT, root.source, sourceRel), rel: sourceRel };
}

function scanNote(notePath, root) {
  const text = fs.readFileSync(notePath, "utf8");
  const relNote = path.relative(VAULT, notePath).replace(/\\/g, "/");
  const refs = [];
  let m;
  SOURCE_RE.lastIndex = 0;
  while ((m = SOURCE_RE.exec(text)) !== null) {
    const sourceRel = normalizeSourcePath(m[1]);
    if (!sourceRel || sourceRel.includes(" ")) continue;
    const start = m[2] ? Number(m[2]) : null;
    const end = m[3] ? Number(m[3]) : start;
    const resolved = resolveSource(root, sourceRel);
    const absSource = resolved.abs;
    const exists = fs.existsSync(absSource);
    let lineCount = null;
    let rangeOk = true;
    if (exists) {
      lineCount = countLines(absSource);
      if (start !== null) {
        rangeOk = start >= 1 && end >= start && end <= lineCount;
      }
    }
    refs.push({
      note: relNote,
      noteLine: lineOfOffset(text, m.index),
      source: resolved.rel,
      rawSource: sourceRel,
      sourceRoot: root.source.replace(/\\/g, "/"),
      start,
      end,
      exists,
      lineCount,
      rangeOk,
    });
  }
  return {
    note: relNote,
    docType: getDocType(text),
    refs,
    sourceCards: scanSourceCards(text, root, relNote),
    sourceFiles: [...new Set(refs.map((r) => r.source))].sort(),
  };
}

const allNotes = [];
for (const root of ROOTS) {
  const readingAbs = path.join(VAULT, root.reading);
  for (const note of walkMarkdown(readingAbs)) {
    allNotes.push({ root, ...scanNote(note, root) });
  }
}

const noteArgIndex = process.argv.indexOf("--note");
if (noteArgIndex !== -1) {
  const rawNote = process.argv[noteArgIndex + 1];
  if (!rawNote) {
    console.error("Usage: node maintenance/audit_source_evidence.mjs --note <note-path>");
    process.exit(2);
  }
  const wanted = path.normalize(rawNote).replace(/\\/g, "/");
  const hit = allNotes.find((n) => n.note === wanted || n.note.endsWith(`/${wanted}`));
  if (!hit) {
    console.error(`Note not found: ${rawNote}`);
    process.exit(1);
  }
  console.log(`=== SOURCE FILES FOR ${hit.note} ===`);
  for (const source of hit.sourceFiles) {
    const refs = hit.refs.filter((r) => r.source === source);
    const ranges = refs
      .map((r) => (r.start === null ? `note:${r.noteLine}` : `note:${r.noteLine} L${r.start}-L${r.end}`))
      .join(", ");
    console.log(`${hit.root.source.replace(/\\/g, "/")}/${source}  (${ranges})`);
  }
  console.log(`Refs: ${hit.refs.length}`);
  console.log(`Unique source files: ${hit.sourceFiles.length}`);
  process.exit(0);
}

const allRefs = allNotes.flatMap((n) => n.refs);
const missing = allRefs.filter((r) => !r.exists);
const badRanges = allRefs.filter((r) => r.exists && !r.rangeOk);
const allSourceCards = allNotes.flatMap((n) => n.sourceCards);
const badSourceCards = allSourceCards.filter((card) => !card.exists || !card.exact);
const notesWithRefs = allNotes.filter((n) => n.refs.length > 0);
const walkthroughs = allNotes.filter((n) => n.docType === "walkthrough");
const walkthroughsWithoutRefs = walkthroughs.filter((n) => n.refs.length === 0);

console.log("=== SOURCE EVIDENCE AUDIT ===");
console.log(`Notes scanned: ${allNotes.length}`);
console.log(`Notes with source refs: ${notesWithRefs.length}`);
console.log(`Source refs: ${allRefs.length}`);
console.log(`Missing source files: ${missing.length}`);
console.log(`Bad line ranges: ${badRanges.length}`);
console.log(`Formal source cards: ${allSourceCards.length}`);
console.log(`Formal source card mismatches: ${badSourceCards.length}`);
console.log(`Walkthrough notes: ${walkthroughs.length}`);
console.log(`Walkthrough notes without refs: ${walkthroughsWithoutRefs.length}`);

console.log("\n=== BY LIBRARY ===");
for (const root of ROOTS) {
  const notes = allNotes.filter((n) => n.root.tag === root.tag);
  const refs = notes.flatMap((n) => n.refs);
  const files = new Set(refs.filter((r) => r.exists).map((r) => r.source));
  const missingCount = refs.filter((r) => !r.exists).length;
  const badCount = refs.filter((r) => r.exists && !r.rangeOk).length;
  console.log(
    `${root.tag}: notes=${notes.length}, refs=${refs.length}, source_files=${files.size}, missing=${missingCount}, bad_ranges=${badCount}`,
  );
}

function printHits(title, hits) {
  if (hits.length === 0) return;
  console.log(`\n=== ${title} ===`);
  for (const h of hits.slice(0, 50)) {
    const range = h.start === null ? "" : ` L${h.start}-L${h.end}`;
    console.log(`${h.note}:${h.noteLine} -> ${h.sourceRoot}/${h.source}${range}`);
  }
  if (hits.length > 50) console.log(`... +${hits.length - 50} more`);
}

printHits("MISSING SOURCE FILES", missing);
printHits("BAD LINE RANGES", badRanges);
printHits("FORMAL SOURCE CARD MISMATCHES", badSourceCards);

console.log("\n=== WRITING GATE ===");
console.log(
  "Before rewriting any note, read every unique upstream file listed by that note's source refs, then verify the cited ranges still support the explanation.",
);

if (
  missing.length > 0 ||
  badRanges.length > 0 ||
  badSourceCards.length > 0 ||
  walkthroughsWithoutRefs.length > 0
) {
  process.exitCode = 1;
}
