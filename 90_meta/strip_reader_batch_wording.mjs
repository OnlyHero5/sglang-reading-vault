#!/usr/bin/env node
/**
 * Remove reader-facing "批次 NN" wording from sglang_reading.
 * Keeps frontmatter `batch:` and tags for Obsidian graph / maintainer use.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const READING = path.join(__dirname, "..", "sglang_reading");
const SKIP_DIRS = new Set(["_archive", "_TEMPLATE", "_qa", ".git", ".obsidian"]);

/** MOC-level reader titles (no batch prefix) */
const MODULE_MOC_TITLE = {
  "00-方法论": "项目总览与阅读方法论",
  "02-启动链路": "启动链路与 CLI",
  "03-HTTP-Server": "HTTP Server 入口",
  "04-OpenAI-API": "OpenAI API 兼容层",
  "05-gRPC-Proto": "gRPC 与 Proto",
  "06-TokenizerManager": "TokenizerManager",
  "07-Scheduler": "Scheduler 核心",
  "08-SchedulePolicy": "调度策略",
  "09-ScheduleBatch-IO": "Batch 与 IO 结构",
  "10-Detokenizer": "Detokenizer 与输出",
  "11-ModelRunner": "ModelRunner 与执行器",
  "12-ModelLoader": "模型加载",
  "13-Models-通用": "通用模型实现",
  "14-Models-专用": "专用模型实现",
  "15-RadixAttention": "RadixAttention 与前缀缓存",
  "16-KV-Cache": "KV Cache 分配与存储",
  "17-Attention": "Attention 后端",
  "18-MoE": "MoE 层",
  "19-Quantization": "量化",
  "20-Sampling": "Sampling 与约束解码",
  "21-Speculative": "投机解码（Speculative Decoding）",
  "22-Disaggregation": "Prefill-Decode 分离（Disaggregation）",
  "23-Distributed": "分布式并行（Distributed）",
  "24-Multimodal": "多模态 VLM（Multimodal）",
  "25-LoRA": "LoRA 多适配器服务",
  "26-sgl-kernel": "sgl-kernel",
  "27-model-gateway": "sgl-model-gateway",
  "28-Frontend-lang": "Frontend Language（SGL）",
  "29-multimodal_gen": "multimodal_gen（扩散 / 多模态生成）",
  "31-Observability": "可观测性（Observability）",
  "32-CheckpointEngine": "CheckpointEngine 运行时权重热更新",
  "01-启动与入口": "阶段 I · 启动与入口",
  "02-请求调度": "阶段 II · 请求调度",
  "03-模型执行": "阶段 III · 模型执行",
  "04-内存与Attention": "阶段 IV · 内存与 Attention",
  "05-高级特性": "阶段 V · 高级特性",
  "06-扩展组件": "阶段 VI · 扩展组件",
  "07-总结与索引": "总结与索引",
};

const DOC_SUFFIX = {
  concept: "核心概念",
  walkthrough: "源码走读",
  dataflow: "数据流与交互",
  faq: "关键问题",
  checkpoint: "验收清单",
  moc: null,
};

/** Inline prose: 批次 03 → HTTP Server */
const BATCH_SHORT = {
  "01": "阅读方法论",
  "02": "启动链路",
  "03": "HTTP Server",
  "04": "OpenAI API",
  "05": "gRPC/Proto",
  "06": "TokenizerManager",
  "07": "Scheduler",
  "08": "调度策略",
  "09": "ScheduleBatch-IO",
  "10": "Detokenizer",
  "11": "ModelRunner",
  "12": "ModelLoader",
  "13": "Models 通用",
  "14": "Models 专用",
  "15": "RadixAttention",
  "16": "KV Cache",
  "17": "Attention",
  "18": "MoE",
  "19": "Quantization",
  "20": "Sampling",
  "21": "投机解码",
  "22": "PD 分离",
  "23": "分布式并行",
  "24": "Multimodal",
  "25": "LoRA",
  "26": "sgl-kernel",
  "27": "model-gateway",
  "28": "Frontend Language",
  "29": "multimodal_gen",
  "30": "总结与索引",
  "31": "可观测性",
  "32": "CheckpointEngine",
};

const BATCH_TO_MODULE = {
  "01": "00-方法论",
  "02": "02-启动链路",
  "03": "03-HTTP-Server",
  "04": "04-OpenAI-API",
  "05": "05-gRPC-Proto",
  "06": "06-TokenizerManager",
  "07": "07-Scheduler",
  "08": "08-SchedulePolicy",
  "09": "09-ScheduleBatch-IO",
  "10": "10-Detokenizer",
  "11": "11-ModelRunner",
  "12": "12-ModelLoader",
  "13": "13-Models-通用",
  "14": "14-Models-专用",
  "15": "15-RadixAttention",
  "16": "16-KV-Cache",
  "17": "17-Attention",
  "18": "18-MoE",
  "19": "19-Quantization",
  "20": "20-Sampling",
  "21": "21-Speculative",
  "22": "22-Disaggregation",
  "23": "23-Distributed",
  "24": "24-Multimodal",
  "25": "25-LoRA",
  "26": "26-sgl-kernel",
  "27": "27-model-gateway",
  "28": "28-Frontend-lang",
  "29": "29-multimodal_gen",
  "31": "31-Observability",
  "32": "32-CheckpointEngine",
};

const SKIP_FILES = new Set([
  "progress.md",
  "PLAN.md",
]);

function readerTitle(module, docType) {
  const base = MODULE_MOC_TITLE[module] || module.replace(/^\d+-/, "");
  if (docType === "moc" || !docType) return base;
  const suffix = DOC_SUFFIX[docType];
  if (!suffix) return base;
  if (docType === "checkpoint") return `${base} · 验收清单`;
  return `${base} · ${suffix}`;
}

function parseFrontmatter(text) {
  const m = text.match(/^---\n([\s\S]*?)\n---\n/);
  if (!m) return { fm: null, body: text };
  const fm = {};
  for (const line of m[1].split("\n")) {
    const kv = line.match(/^(\w+):\s*"?([^"]*)"?$/);
    if (kv) fm[kv[1]] = kv[2];
  }
  return { fm, body: text.slice(m[0].length), raw: m[1] };
}

function updateFrontmatterTitle(raw, newTitle) {
  if (/^title:/m.test(raw)) {
    return raw.replace(/^title:.*$/m, `title: "${newTitle}"`);
  }
  return raw;
}

function replaceBody(text) {
  let out = text;

  // Section headers
  out = out.replace(/## 上一批 \/ 下一批/g, "## 阅读路径");
  out = out.replace(/## 下一批/g, "## 下一模块");
  out = out.replace(/## 上一批/g, "## 上一模块");

  // Wikilink aliases: |批次 NN：...| or |批次 NN · ...| or |批次 NN ...|
  out = out.replace(/\|批次\s*(\d{1,2})\s*[：:·]\s*([^|\]]+)\|/g, (_, b, rest) => {
    const mod = BATCH_TO_MODULE[b.padStart(2, "0")] || BATCH_TO_MODULE[b];
    const full = mod ? MODULE_MOC_TITLE[mod] : rest.trim();
    return `|${full || rest.trim()}|`;
  });
  out = out.replace(/\|批次\s*(\d{1,2})\s+([^|\]]+)\|/g, (_, b, rest) => {
    const mod = BATCH_TO_MODULE[b.padStart(2, "0")];
    return mod ? `|${MODULE_MOC_TITLE[mod]}|` : `|${rest.trim()}|`;
  });

  // (批次 NN) in mermaid / prose
  for (const [b, short] of Object.entries(BATCH_SHORT).sort((a, c) => c.length - a.length)) {
    const re = new RegExp(`[（(]批次\\s*0?${b.replace(/^0/, "")}[)）]`, "g");
    out = out.replace(re, `（${short}）`);
  }

  // 批次 NN–MM / 批次 NN-MM ranges
  out = out.replace(/批次\s*(\d{1,2})\s*[–—-]\s*(\d{1,2})/g, (_, a, b) => {
    const sa = BATCH_SHORT[a.padStart(2, "0")] || a;
    const sb = BATCH_SHORT[b.padStart(2, "0")] || b;
    return `${sa}–${sb}`;
  });

  // 批次 NN 起 / 从批次 NN
  out = out.replace(/批次\s*(\d{1,2})\s*起/g, (_, b) => {
    const s = BATCH_SHORT[b.padStart(2, "0")];
    return s ? `从 ${s} 起` : `从模块 ${b} 起`;
  });
  out = out.replace(/从批次\s*(\d{1,2})/g, (_, b) => {
    const s = BATCH_SHORT[b.padStart(2, "0")];
    return s ? `从 ${s}` : `从模块 ${b}`;
  });
  out = out.replace(/进入批次\s*(\d{1,2})/g, (_, b) => {
    const s = BATCH_SHORT[b.padStart(2, "0")];
    return s ? `进入 ${s}` : `进入模块 ${b}`;
  });
  out = out.replace(/见批次\s*(\d{1,2})/g, (_, b) => {
    const s = BATCH_SHORT[b.padStart(2, "0")];
    return s ? `见 ${s}` : `见模块 ${b}`;
  });
  out = out.replace(/待批次\s*(\d{1,2})/g, "待后续图谱更新");
  out = out.replace(/深读批次\s*(\d{1,2})/g, (_, b) => {
    const s = BATCH_SHORT[b.padStart(2, "0")];
    return s ? `深读 ${s}` : `深读模块 ${b}`;
  });
  out = out.replace(/补读批次\s*(\d{1,2})/g, (_, b) => {
    const s = BATCH_SHORT[b.padStart(2, "0")];
    return s ? `补读 ${s}` : `补读模块 ${b}`;
  });
  out = out.replace(/交叉阅读批次\s*(\d{1,2})/g, (_, b) => {
    const s = BATCH_SHORT[b.padStart(2, "0")];
    return s ? `交叉阅读 ${s}` : `交叉阅读模块 ${b}`;
  });
  out = out.replace(/对应批次\s*(\d{1,2})/g, "对应专题");
  out = out.replace(/批次\s*(\d{1,2})\s*展开/g, (_, b) => {
    const s = BATCH_SHORT[b.padStart(2, "0")];
    return s ? `${s} 展开` : `模块 ${b} 展开`;
  });
  out = out.replace(/批次\s*(\d{1,2})\s*FAQ/g, (_, b) => {
    const s = BATCH_SHORT[b.padStart(2, "0")];
    return s ? `${s} FAQ` : `模块 ${b} FAQ`;
  });

  // 批次 NN：title / 批次 NN · title / 批次 NN title (headings in body)
  out = out.replace(/^#+\s*批次\s*(\d{1,2})\s*[：:·]\s*/gm, "# ");
  out = out.replace(/^#+\s*批次\s*(\d{1,2})\s+/gm, "# ");

  // Remaining standalone 批次 NN (longest numbers first)
  const batchNums = Object.keys(BATCH_SHORT).sort((a, b) => b.localeCompare(a));
  for (const b of batchNums) {
    const short = BATCH_SHORT[b];
    const num = String(parseInt(b, 10));
    out = out.replace(new RegExp(`批次\\s*0?${num}(?!\\d)`, "g"), short);
  }

  // Stage folder annotations in tree comments
  out = out.replace(/#\s*批次\s*\d+[–—-]\d+/g, "");
  out = out.replace(/#\s*批次\s*\d+/g, "");
  out = out.replace(/（批次\s*\d+[–—-]\d+）/g, "");
  out = out.replace(/（批次\s*\d+）/g, "");

  // 本批 → 本模块 (softer internal term)
  out = out.replace(/本批(?!次)/g, "本模块");

  return out;
}

function processFile(filePath) {
  const rel = path.relative(READING, filePath).replace(/\\/g, "/");
  if (SKIP_FILES.has(path.basename(filePath))) return false;

  let text = fs.readFileSync(filePath, "utf8");
  const { fm, body, raw } = parseFrontmatter(text);
  let changed = false;

  if (fm?.module && fm?.doc_type) {
    const newTitle = readerTitle(fm.module, fm.doc_type);
    const newRaw = updateFrontmatterTitle(raw, newTitle);
    let newText = text;
    if (newRaw !== raw) {
      newText = `---\n${newRaw}\n---\n${body}`;
      changed = true;
    }
    const h1 = `# ${newTitle}`;
    const newBody = newText.replace(/^---[\s\S]*?---\n/, "").replace(/^#\s+.+\n/m, `${h1}\n`);
    if (newBody !== newText.replace(/^---[\s\S]*?---\n/, "")) {
      newText = newText.replace(/^---[\s\S]*?---\n[\s\S]*?(?=^## |\n>|$)/m, (block) => {
        return block.replace(/^#\s+.+\n/m, `${h1}\n`);
      });
      changed = true;
    }
    text = newText;
  }

  const newText = replaceBody(text);
  if (newText !== text) {
    text = newText;
    changed = true;
  }

  if (changed) {
    fs.writeFileSync(filePath, text, "utf8");
    console.log(`updated ${rel}`);
  }
  return changed;
}

/** Special pass for index / guide files without standard frontmatter */
function processSpecialFiles() {
  const specials = [
    "SGLang源码阅读指南.md",
    "07-总结与索引/10-批次编号对照.md",
    "07-总结与索引/07-总结与索引-00-MOC.md",
    "07-总结与索引/01-项目总览.md",
    "07-总结与索引/04-导读路径.md",
    "07-总结与索引/08-设计追问与框架对比.md",
    "07-总结与索引/03-关键概念.md",
    "07-总结与索引/07-总结与索引-04-关键问题.md",
    "07-总结与索引/07-总结与索引-checkpoint.md",
    "07-总结与索引/obsidian-graph-presets.md",
    "07-总结与索引/全链路请求追踪-gRPC.md",
    "07-总结与索引/00-零基础先修.md",
    "_TEMPLATE/README.md",
  ];
  for (const rel of specials) {
    const p = path.join(READING, rel);
    if (fs.existsSync(p)) processFile(p);
  }
}

function walk(dir) {
  let n = 0;
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP_DIRS.has(ent.name)) continue;
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) n += walk(full);
    else if (ent.name.endsWith(".md")) if (processFile(full)) n++;
  }
  return n;
}

let count = walk(READING);
processSpecialFiles();
console.log(`\nDone. Files with batch wording remaining:`);

// audit
let remaining = 0;
function audit(dir) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP_DIRS.has(ent.name)) continue;
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) audit(full);
    else if (ent.name.endsWith(".md") && !SKIP_FILES.has(ent.name)) {
      const t = fs.readFileSync(full, "utf8");
      const body = t.replace(/^---[\s\S]*?---\n/, "");
      if (/批次\s*\d/.test(body) || /# 批次/.test(t)) {
        remaining++;
        if (remaining <= 15) {
          console.log(`  ${path.relative(READING, full).replace(/\\/g, "/")}`);
        }
      }
    }
  }
}
audit(READING);
console.log(`  total: ${remaining} (excludes progress.md / PLAN.md / _qa / _archive)`);
