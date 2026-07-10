#!/usr/bin/env node
/**
 * Audit reader-facing Markdown hygiene and reading-quality signals.
 *
 * This script is read-only. It intentionally ignores fenced code blocks for
 * heading and prose checks, so source snippets such as "# TODO" or "# comment"
 * do not become false Markdown errors.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const VAULT = path.resolve(__dirname, "..");

const ROOTS = [
  "index.md",
  "README.md",
  "AGENTS.md",
  "sglang_reading",
  "slime_reading",
  "flash-attn_reading",
  "AI-Infra课程",
  "knowledge_maps",
];

const SKIP_DIRS = new Set(["模板", ".obsidian", ".git", "sglang", "slime", "flash-attn"]);

const OLD_STRUCTURE_RE =
  /(?<![A-Za-z])(?:Explain|Code|Comment)\s*[:：]|Explain\s*→\s*Code\s*→\s*Comment|\bETC\b/u;
const TODO_PROSE_RE = /\b(?:TODO|FIXME|TBD)\b|待补|待完善|稍后补|后续补|省略号占位/u;
const LEGACY_MAINTENANCE_PROSE_RE =
  /\b(?:MOC|module-moc|stage-moc|batch-doc|phase-moc|doc_type|onboarding)\b|本阅读项目|本 vault|读者向|唯一权威|固定套件|源码段数量|覆盖专题|全链路 Hop|(?<![A-Za-z0-9])(?:0\d|[12]\d|3[0-2])-[A-Za-z\u4e00-\u9fff]|\[\[[^\]]+\|(?:FA)?\d{1,2}[-/ ][^\]]+\]\]/u;
const SOURCE_HEADING_RE = /^来源\s*[:：]/u;
const ORIENTATION_HEADING_RE =
  /读者任务|你为什么要读|读者为什么要读|你为什么要做|为什么要读|先回答为什么读|学习目标|本模块目标|读者能做什么|读者能画出来|读完能|读完应能回答|解决什么问题|如何使用|怎么读这篇|阅读路径|首次阅读路径|先建立模型|本目录定位|长文读法|读者自测|验收目标|必达能力|通过标准|症状速查|快速定位表|排障入口|症状总表|排障总表|快速排障表/u;
const VERIFICATION_HEADING_RE =
  /运行验证|可执行验证|最小验证|可观测验证|静态验证|断点验证|验证实验|验证方式|验证抓手|怎么验证|如何验证|验证建议|哪些验证|可执行检查|必跑检查|跑哪些检查|最小运行验收|学习检查|通过标准|测试矩阵|排障(?:入口|速查|总表|顺序)/u;
const EXPECTATION_RE =
  /预期|期望|通过标准|症状|判断方式|应该看到|应当看到|失败时|确认|观察(?:到|是否)|保持一致|数值一致|行为不变/u;
const ACTION_RE =
  /\bcurl\b|\bpytest\b|\bpython\b|\bnode\b|\brg\s|\bnsys\b|\bncu\b|benchmark|日志|metric|断点|单测|测试|操作|运行|执行|命令|观察|记录|对照|复现|发起请求/u;
const VALID_TYPES = new Set([
  "map",
  "guide",
  "concept",
  "walkthrough",
  "dataflow",
  "troubleshooting",
  "exercise",
  "reference",
  "template",
  "index",
  "dashboard",
  "readme",
]);
const VALID_LEARNING_ROLES = new Set(["core", "reference", "debug", "practice"]);

function rel(filePath) {
  return path.relative(VAULT, filePath).replace(/\\/g, "/");
}

function walkMarkdown(entry, out = []) {
  const abs = path.join(VAULT, entry);
  if (!fs.existsSync(abs)) return out;
  const stat = fs.statSync(abs);
  if (stat.isFile()) {
    if (entry.endsWith(".md")) out.push(abs);
    return out;
  }
  for (const ent of fs.readdirSync(abs, { withFileTypes: true })) {
    if (SKIP_DIRS.has(ent.name)) continue;
    const full = path.join(abs, ent.name);
    if (ent.isDirectory()) walkMarkdown(rel(full), out);
    else if (ent.name.endsWith(".md")) out.push(full);
  }
  return out;
}

function frontmatter(text) {
  const m = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!m) return null;
  return m[1];
}

function yamlScalar(fm, key) {
  const m = fm.match(new RegExp(`^${key}:\\s*(.*?)\\s*$`, "m"));
  return m ? m[1].trim() : "";
}

function hasTags(fm) {
  const inline = fm.match(/^tags:\s*(.+)\s*$/m);
  if (inline && inline[1].trim() !== "") return true;

  const lines = fm.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    if (!/^tags:\s*$/.test(lines[i])) continue;
    for (let j = i + 1; j < lines.length; j++) {
      if (/^\S/.test(lines[j])) return false;
      if (/^\s+-\s+\S/.test(lines[j])) return true;
    }
  }
  return false;
}

function stripInlineCode(line) {
  let out = "";
  let inCode = false;
  for (let i = 0; i < line.length; i++) {
    if (line[i] === "`") {
      inCode = !inCode;
      out += " ";
      continue;
    }
    out += inCode ? " " : line[i];
  }
  return out;
}

function scanMarkdown(text) {
  const lines = text.split(/\r?\n/);
  const headings = [];
  const proseLines = [];
  const mermaidBlocks = [];
  let inFence = false;
  let fenceLang = "";
  let fenceStart = 0;
  let fenceBody = [];
  let inFrontmatter = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineNo = i + 1;

    if (i === 0 && line === "---") {
      inFrontmatter = true;
      continue;
    }
    if (inFrontmatter) {
      if (line === "---") inFrontmatter = false;
      continue;
    }

    const fence = line.match(/^\s*```(\S*)?/);
    if (fence) {
      if (!inFence) {
        inFence = true;
        fenceLang = (fence[1] || "").toLowerCase();
        fenceStart = lineNo;
        fenceBody = [];
      } else {
        if (fenceLang === "mermaid") {
          mermaidBlocks.push({ start: fenceStart, body: fenceBody.join("\n") });
        }
        inFence = false;
        fenceLang = "";
        fenceBody = [];
      }
      continue;
    }

    if (inFence) {
      fenceBody.push(line);
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+?)\s*$/);
    if (heading) {
      headings.push({ line: lineNo, level: heading[1].length, text: heading[2] });
    }
    proseLines.push({ line: lineNo, text: stripInlineCode(line) });
  }

  return { headings, proseLines, mermaidBlocks, unclosedFence: inFence ? fenceStart : 0 };
}

function push(map, key, file, line, detail) {
  if (!map.has(key)) map.set(key, []);
  map.get(key).push({ file, line, detail });
}

function firstLinesOutsideFences(proseLines, maxLine = 80) {
  return proseLines
    .filter((l) => l.line <= maxLine)
    .map((l) => l.text)
    .join("\n");
}

const files = ROOTS.flatMap((root) => walkMarkdown(root)).sort();
const errors = new Map();
const warnings = new Map();
const stats = {
  files: 0,
  h1Ok: 0,
  mermaidBlocks: 0,
  oldStructureLines: 0,
  legacyMaintenanceLines: 0,
  sourceHeadingLines: 0,
  docsWithReaderTaskSignal: 0,
  docsWithVerificationSignal: 0,
  docsWithActionAndExpectation: 0,
  maintenanceNumberPaths: 0,
  longDocs: 0,
};

for (const file of files) {
  const text = fs.readFileSync(file, "utf8");
  const fileRel = rel(file);
  const fm = frontmatter(text);
  const parsed = scanMarkdown(text);
  stats.files++;
  stats.mermaidBlocks += parsed.mermaidBlocks.length;

  if (/(^|\/)(?:\d{2}-|FA\d{2}-)/u.test(fileRel) || /(?:-00-MOC|-05-checkpoint)\.md$/u.test(fileRel)) {
    stats.maintenanceNumberPaths++;
    push(errors, "maintenance-number-path", fileRel, 1, "reader-facing paths must use semantic names");
  }

  if (!fm) {
    push(errors, "missing-frontmatter", fileRel, 1, "frontmatter is required");
  } else {
    if (!hasTags(fm)) push(errors, "missing-tags", fileRel, 1, "frontmatter tags are required");
    if (/^(?:batch|phase):/m.test(fm) || /\/(?:batch|phase)\//u.test(fm)) {
      push(errors, "maintenance-number-frontmatter", fileRel, 1, "remove batch/phase maintenance metadata");
    }
    const type = yamlScalar(fm, "type");
    if (type && !VALID_TYPES.has(type)) push(errors, "unknown-type", fileRel, 1, `type=${type}`);
    const learningRole = yamlScalar(fm, "learning_role");
    if (learningRole && !VALID_LEARNING_ROLES.has(learningRole)) {
      push(errors, "unknown-learning-role", fileRel, 1, `learning_role=${learningRole}`);
    }
  }

  const h1s = parsed.headings.filter((h) => h.level === 1);
  if (h1s.length === 1) stats.h1Ok++;
  else push(errors, "h1-count", fileRel, h1s[0]?.line || 1, `expected 1 H1, found ${h1s.length}`);

  if (parsed.unclosedFence) {
    push(errors, "unclosed-code-fence", fileRel, parsed.unclosedFence, "code fence is not closed");
  }

  for (let i = 1; i < parsed.headings.length; i++) {
    const prev = parsed.headings[i - 1];
    const cur = parsed.headings[i];
    if (cur.level > prev.level + 1) {
      push(errors, "heading-level-jump", fileRel, cur.line, `H${prev.level} -> H${cur.level}`);
    }
  }

  for (const heading of parsed.headings) {
    if (SOURCE_HEADING_RE.test(heading.text.trim())) {
      stats.sourceHeadingLines++;
      push(
        errors,
        "source-reference-heading",
        fileRel,
        heading.line,
        "write source refs inside the code block, not as Markdown headings"
      );
    }
  }

  for (const block of parsed.mermaidBlocks) {
    if (block.body.includes("\\n")) {
      push(errors, "mermaid-literal-newline", fileRel, block.start, "use <br/> inside Mermaid labels");
    }
    if (/\b(?:concept|flow|layer):[a-z]/u.test(block.body)) {
      push(
        errors,
        "legacy-maintenance-mermaid-node",
        fileRel,
        block.start,
        "use reader-facing semantic labels instead of internal graph ids"
      );
    }
  }

  for (const line of parsed.proseLines) {
    if (OLD_STRUCTURE_RE.test(line.text)) {
      stats.oldStructureLines++;
      push(errors, "old-structure-marker", fileRel, line.line, line.text.trim().slice(0, 120));
    }
    if (fileRel !== "AGENTS.md" && LEGACY_MAINTENANCE_PROSE_RE.test(line.text)) {
      stats.legacyMaintenanceLines++;
      push(errors, "legacy-maintenance-prose", fileRel, line.line, line.text.trim().slice(0, 120));
    }
    if (TODO_PROSE_RE.test(line.text)) {
      push(warnings, "reader-prose-todo", fileRel, line.line, line.text.trim().slice(0, 120));
    }
  }

  const docType = fm ? yamlScalar(fm, "type") : "";
  const learningRole = fm ? yamlScalar(fm, "learning_role") : "";
  const earlyHeadings = parsed.headings.filter((h) => h.line <= 80 && h.level === 2).map((h) => h.text).join("\n");
  if (ORIENTATION_HEADING_RE.test(earlyHeadings)) {
    stats.docsWithReaderTaskSignal++;
  } else if (["map", "guide", "concept", "walkthrough", "dataflow", "troubleshooting", "exercise"].includes(docType)) {
    push(warnings, "missing-early-orientation-signal", fileRel, 1, `type=${docType}`);
  }

  const allProse = parsed.proseLines.map((l) => l.text).join("\n");
  const verificationText = `${allProse}\n${text}`;
  const verificationHeadings = parsed.headings.map((h) => h.text).join("\n");
  const hasVerificationHeading = VERIFICATION_HEADING_RE.test(verificationHeadings);
  const hasActionAndExpectation = ACTION_RE.test(verificationText) && EXPECTATION_RE.test(verificationText);
  if (hasVerificationHeading || (docType === "troubleshooting" && hasActionAndExpectation)) {
    stats.docsWithVerificationSignal++;
  } else if (docType === "walkthrough" || docType === "exercise" || docType === "troubleshooting") {
    push(warnings, "missing-verification-signal", fileRel, 1, `type=${docType}`);
  }
  if (hasActionAndExpectation) {
    stats.docsWithActionAndExpectation++;
  } else if (["walkthrough", "troubleshooting", "exercise"].includes(docType)) {
    push(warnings, "verification-without-action-or-expectation", fileRel, 1, `type=${docType}`);
  }

  const lineCount = text.split(/\r?\n/).length;
  const baseLimit = docType === "walkthrough" ? 800 : 500;
  const limit =
    learningRole === "reference"
      ? docType === "walkthrough"
        ? 1600
        : 800
      : learningRole === "debug"
        ? 650
        : baseLimit;
  if (lineCount > limit) {
    stats.longDocs++;
    push(warnings, "long-reader-document", fileRel, 1, `${lineCount} lines; recommended <= ${limit}`);
  } else if (
    lineCount > baseLimit &&
    learningRole === "reference" &&
    !parsed.headings.some((h) => /长文读法|怎么读这篇|阅读路径/u.test(h.text))
  ) {
    push(
      warnings,
      "long-reference-without-reading-route",
      fileRel,
      1,
      `${lineCount} lines; add a precise selective-reading route`
    );
  }
}

function printGroup(title, map, maxHits = 20) {
  const entries = [...map.entries()].sort(([a], [b]) => a.localeCompare(b));
  console.log(`\n=== ${title} (${entries.reduce((sum, [, hits]) => sum + hits.length, 0)}) ===`);
  if (entries.length === 0) {
    console.log("none");
    return;
  }
  for (const [key, hits] of entries) {
    console.log(`\n${key}: ${hits.length}`);
    for (const h of hits.slice(0, maxHits)) {
      console.log(`  ${h.file}:${h.line}  ${h.detail}`);
    }
    if (hits.length > maxHits) console.log(`  ... +${hits.length - maxHits} more`);
  }
}

function asObject(map) {
  return Object.fromEntries([...map.entries()].sort(([a], [b]) => a.localeCompare(b)));
}

if (process.argv.includes("--json")) {
  console.log(JSON.stringify({ stats, errors: asObject(errors), warnings: asObject(warnings) }, null, 2));
  const errorCount = [...errors.values()].reduce((sum, hits) => sum + hits.length, 0);
  process.exit(errorCount === 0 ? 0 : 1);
}

const maxHits = process.argv.includes("--all") ? Number.POSITIVE_INFINITY : 20;

console.log("=== MARKDOWN QUALITY AUDIT ===");
console.log(`Files scanned: ${stats.files}`);
console.log(`Files with exactly one H1: ${stats.h1Ok}`);
console.log(`Mermaid blocks: ${stats.mermaidBlocks}`);
console.log(`Old structure lines: ${stats.oldStructureLines}`);
console.log(`Legacy maintenance lines: ${stats.legacyMaintenanceLines}`);
console.log(`Source reference headings: ${stats.sourceHeadingLines}`);
console.log(`Docs with early orientation signal: ${stats.docsWithReaderTaskSignal}`);
console.log(`Docs with verification signal: ${stats.docsWithVerificationSignal}`);
console.log(`Docs with action + expectation: ${stats.docsWithActionAndExpectation}`);
console.log(`Maintenance-number paths: ${stats.maintenanceNumberPaths}`);
console.log(`Long reader documents: ${stats.longDocs}`);

printGroup("ERRORS", errors, maxHits);
printGroup("WARNINGS", warnings, maxHits);

const errorCount = [...errors.values()].reduce((sum, hits) => sum + hits.length, 0);
process.exit(errorCount === 0 ? 0 : 1);
