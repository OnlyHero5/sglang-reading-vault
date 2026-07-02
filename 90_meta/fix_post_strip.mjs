#!/usr/bin/env node
/**
 * Post-process after strip_reader_batch_wording:
 * 1. Fix YAML list indentation
 * 2. Restore fenced code from sglang/ via # 来源： headers
 * 3. Fix duplicated titles / wikilink aliases (Module：Module title)
 * 4. Remove remaining reader-facing 批次 in stage/index MOCs
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const VAULT = path.resolve(__dirname, "..");
const READING = path.join(VAULT, "sglang_reading");
const SGLANG = path.join(VAULT, "sglang");
const SKIP = new Set(["_archive", "_TEMPLATE", "_qa", ".git", ".obsidian", "progress.md", "PLAN.md"]);

const SOURCE_RE =
  /^#\s*来源[：:]\s*(.+?\.\w+)\s+L(\d+)(?:\s*[-–—]\s*L?(\d+))?/;

function readSourceLines(relPath, start, end) {
  let rel = relPath.replace(/\\/g, "/").trim();
  const candidates = [
    path.join(SGLANG, rel),
    path.join(SGLANG, rel.replace(/^python\//, "")),
    path.join(VAULT, rel),
  ];
  if (!rel.startsWith("python/") && rel.includes("sglang/")) {
    candidates.unshift(path.join(SGLANG, rel.replace(/^sglang\//, "python/sglang/")));
  }
  let file = candidates.find((c) => fs.existsSync(c));
  if (!file) return null;
  const lines = fs.readFileSync(file, "utf8").split("\n");
  const s = start - 1;
  const e = end ? end : start;
  return lines.slice(s, e).join("\n");
}

function fixYamlLists(text) {
  return text.replace(/^(tags:|aliases:)\n((?: - .+\n)+)/gm, (_, k, items) => {
    return `${k}\n${items.replace(/^ - /gm, "  - ")}`;
  });
}

function restoreCodeBlocks(text) {
  return text.replace(/```(\w*)\r?\n([\s\S]*?)```/g, (full, lang, body) => {
    const headerLine = body.split("\n").find((l) => SOURCE_RE.test(l.trim()));
    if (!headerLine) return full;
    const m = headerLine.trim().match(SOURCE_RE);
    if (!m) return full;
    const snippet = readSourceLines(m[1], parseInt(m[2], 10), m[3] ? parseInt(m[3], 10) : null);
    if (!snippet) return full;
    const versionLine = body.split("\n").find((l) => l.trim().startsWith("# 提交版本"));
    const head = [headerLine.trim()];
    if (versionLine) head.push(versionLine.trim());
    return "```" + lang + "\n" + head.join("\n") + "\n" + snippet + "\n```";
  });
}

function fixTitlesAndAliases(text) {
  let out = text;
  // title: "启动链路：启动链路与 CLI" → title: "启动链路与 CLI"
  out = out.replace(/^title: "([^："]+)：([^"]+)"/gm, (all, a, b) => {
    if (b.includes(a) || a.includes(b.split(/[（(]/)[0])) return `title: "${b}"`;
    return all;
  });
  // [[x|Module：Full Title]] → [[x|Full Title]] when redundant
  out = out.replace(/\|([^|\]]+)：([^|\]]+)\]\]/g, (all, a, b) => {
    if (b.includes(a.trim()) || a.trim().length <= 4) return `|${b}]]`;
    return all;
  });
  return out;
}

function fixStageMocs(text) {
  let out = text;
  out = out.replace(/四个批次从 CLI/g, "四个模块从 CLI");
  out = out.replace(/\| 批次 \|/g, "| 模块 |");
  out = out.replace(/## 批次导航/g, "## 模块导航");
  out = out.replace(/\| 顺序 \| 批次 \|/g, "| 顺序 | 模块 |");
  out = out.replace(/\| \*\*批次\*\* \|/g, "| **序号** |");
  out = out.replace(/（批次 [0-9–—-]+）/g, "");
  out = out.replace(/# 10 · 批次编号与目录对照/g, "# 10 · 模块与目录对照");
  out = out.replace(/title: "10-批次编号对照"/g, 'title: "10-模块与目录对照"');
  out = out.replace(/正文 `# 批次 NN` 标题以 \*\*上表批次号\*\* 为准，不以文件夹前缀为准/g,
    "正文标题以**模块名**为准；`00-`/`01-` 等前缀是阶段文件夹编号，不是阅读顺序。");
  out = out.replace(/subgraph srt\["srt 主路径 \(批次 01-23\)"\]/g,
    'subgraph srt["srt 主路径"]');
  out = out.replace(/subgraph srt\["srt 主路径 \(01-23\)"\]/g,
    'subgraph srt["srt 主路径"]');
  // mermaid node labels: 05 gRPC → gRPC/Proto
  out = out.replace(/GRPC\[05 gRPC Server\]/g, "GRPC[gRPC/Proto]");
  out = out.replace(/HTTP\[03 HTTP Server\]/g, "HTTP[HTTP Server]");
  out = out.replace(/OAI\[04 OpenAI 路由\]/g, "OAI[OpenAI API]");
  out = out.replace(/\[\[02-启动链路-00-MOC\|02 启动链路\]\]/g, "[[02-启动链路-00-MOC|启动链路]]");
  out = out.replace(/\[\[03-HTTP-Server-00-MOC\|03 HTTP Server\]\]/g, "[[03-HTTP-Server-00-MOC|HTTP Server]]");
  out = out.replace(/\[\[04-OpenAI-API-00-MOC\|04 OpenAI API\]\]/g, "[[04-OpenAI-API-00-MOC|OpenAI API]]");
  out = out.replace(/\[\[05-gRPC-Proto-00-MOC\|05 gRPC Proto\]\]/g, "[[05-gRPC-Proto-00-MOC|gRPC/Proto]]");
  out = out.replace(/\| 1 \| 02 \|/g, "| 1 | 启动链路 |");
  out = out.replace(/\| 2 \| 03 \|/g, "| 2 | HTTP Server |");
  out = out.replace(/\| 3 \| 04 \|/g, "| 3 | OpenAI API |");
  out = out.replace(/\| 4 \| 05 \|/g, "| 4 | gRPC/Proto |");
  out = out.replace(/\| 02 \|/g, "| 启动链路 |");
  out = out.replace(/\| 03 \|/g, "| HTTP Server |");
  out = out.replace(/\| 04 \|/g, "| OpenAI API |");
  out = out.replace(/\| 05 \|/g, "| gRPC/Proto |");
  return out;
}

function processFile(filePath) {
  let text = fs.readFileSync(filePath, "utf8");
  const orig = text;
  text = fixYamlLists(text);
  text = restoreCodeBlocks(text);
  text = fixTitlesAndAliases(text);
  text = fixStageMocs(text);
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
    else if (ent.name.endsWith(".md") && processFile(full)) {
      n++;
      if (n <= 5) console.log("fixed", path.relative(READING, full));
    }
  }
  return n;
}

const n = walk(READING);
console.log(`\nFixed ${n} files`);

// audit reader-facing 批次
let rem = 0;
function audit(d) {
  for (const ent of fs.readdirSync(d, { withFileTypes: true })) {
    if (SKIP.has(ent.name)) continue;
    const f = path.join(d, ent.name);
    if (ent.isDirectory()) audit(f);
    else if (ent.name.endsWith(".md")) {
      const body = fs.readFileSync(f, "utf8").replace(/^---[\s\S]*?---\n/, "");
      if (/批次\s*\d/.test(body)) {
        rem++;
        if (rem <= 10) console.log("remaining:", path.relative(READING, f));
      }
    }
  }
}
audit(READING);
console.log(`Reader-facing 批次 refs remaining: ${rem}`);
