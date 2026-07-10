#!/usr/bin/env node

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const VAULT = path.resolve(__dirname, "..");
const APPLY = process.argv.includes("--apply");

const ROLE_SUFFIX = new Map([
  ["00-MOC", ""],
  ["01-核心概念", "核心概念"],
  ["02-源码走读", "源码走读"],
  ["03-数据流与交互", "数据流"],
  ["04-关键问题", "排障指南"],
  ["05-checkpoint", "学习检查"],
]);

const TYPE_MAP = new Map([
  ["moc", "map"],
  ["concept", "concept"],
  ["walkthrough", "walkthrough"],
  ["dataflow", "dataflow"],
  ["faq", "troubleshooting"],
  ["checkpoint", "exercise"],
]);

const LIBRARIES = {
  sglang_reading: {
    prefix: "SGLang",
    framework: "sglang",
    baseline: "70df09b",
    dirs: {
      "方法论": "阅读方法",
      "Models-通用": "通用模型",
      "Models-专用": "专用模型",
      "ScheduleBatch-IO": "ScheduleBatch数据结构",
      "Multimodal": "多模态",
      "Disaggregation": "PD分离",
      "Distributed": "分布式",
      "Observability": "可观测性",
      "Frontend-lang": "前端语言",
      "multimodal_gen": "多模态生成",
    },
    special: {
      "SGLang源码阅读指南": ["SGLang学习指南", "SGLang 学习指南", "guide"],
      "00-导读与总览-00-MOC": ["SGLang-导读与总览", "SGLang 导读与总览", "map"],
      "00-零基础先修": ["SGLang-零基础先修", "SGLang 零基础先修", "concept"],
      "01-项目总览": ["SGLang-项目总览", "SGLang 项目总览", "concept"],
      "02-架构分层": ["SGLang-架构分层", "SGLang 架构分层", "concept"],
      "03-关键概念": ["SGLang-关键概念", "SGLang 关键概念", "concept"],
      "04-导读路径": ["SGLang-学习路径", "SGLang 学习路径", "guide"],
      "05-文件地图": ["SGLang-源码地图", "SGLang 源码地图", "reference"],
      "07-用户故事与场景": ["SGLang-用户场景", "SGLang 用户场景", "walkthrough"],
      "模块依赖图": ["SGLang-模块依赖图", "SGLang 模块依赖图", "map"],
      "全链路请求追踪": ["SGLang-HTTP请求全链路", "SGLang HTTP 请求全链路", "walkthrough"],
      "全链路请求追踪-gRPC": ["SGLang-gRPC请求全链路", "SGLang gRPC 请求全链路", "walkthrough"],
      "术语表": ["SGLang-术语表", "SGLang 术语表", "reference"],
      "业务域流程": ["SGLang-业务流程", "SGLang 业务流程", "walkthrough"],
      "与Slime阅读对照": ["SGLang与Slime-阅读对照", "SGLang 与 Slime 阅读对照", "map"],
      "SGLang-图谱预设重定向": ["SGLang-图谱使用说明", "SGLang 图谱使用说明", "reference"],
      "90-总结复盘-00-MOC": ["SGLang-总结复盘", "SGLang 总结复盘", "map"],
      "90-总结复盘-01-复杂度热点": ["SGLang-复杂度热点", "SGLang 复杂度热点", "reference"],
      "90-总结复盘-02-设计追问与框架对比": ["SGLang-框架对比与设计决策", "SGLang 框架对比与设计决策", "reference"],
      "90-总结复盘-03-生产排障速查": ["SGLang-生产排障", "SGLang 生产排障", "troubleshooting"],
      "90-总结复盘-04-关键问题": ["SGLang-常见问题", "SGLang 常见问题", "troubleshooting"],
      "90-总结复盘-05-未独立成专题导读": ["SGLang-补充主题", "SGLang 补充主题", "reference"],
      "90-总结复盘-06-checkpoint": ["SGLang-综合学习检查", "SGLang 综合学习检查", "exercise"],
      "SGLang-模板说明": ["SGLang-专题模板", "SGLang 专题模板", "template"],
    },
  },
  slime_reading: {
    prefix: "Slime",
    framework: "slime",
    baseline: "22cdc6e1",
    dirs: {
      "方法论": "阅读方法",
      "Arguments-Ray": "Ray参数",
      "Arguments-TrainRollout": "训练与Rollout参数",
      "Tools-DataPrep": "数据准备工具",
      "EngineTopology": "引擎拓扑",
      "Sample-Contracts": "Sample数据契约",
      "DataSource": "数据源",
      "RM-FilterHub": "Reward与过滤",
      "Alt-Rollout": "其他Rollout路径",
      "External-Engines": "外部推理引擎",
      "Megatron-Actor-Init": "Megatron-Actor初始化",
      "Model-Init": "模型初始化",
      "Train-Step": "训练步骤",
      "Train-Data": "训练数据",
      "Loss-Advantages": "Advantage计算",
      "Loss-Policy": "Policy-Loss",
      "CP-RoutingReplay": "上下文并行与路由重放",
      "WeightSync-Dist": "分布式权重同步",
      "WeightSync-Disk": "磁盘权重同步",
      "Checkpoint-M2HF": "Megatron到HF转换",
      "Agent-Trajectory": "Agent轨迹",
      "Customization": "自定义扩展",
      "Plugins-Examples": "插件与示例",
    },
    special: {
      "Slime源码阅读指南": ["Slime学习指南", "Slime 学习指南", "guide"],
      "Slime-00-导读与总览-00-MOC": ["Slime-导读与总览", "Slime 导读与总览", "map"],
      "Slime-00-零基础先修": ["Slime-零基础先修", "Slime 零基础先修", "concept"],
      "Slime-01-项目总览": ["Slime-项目总览", "Slime 项目总览", "concept"],
      "Slime-02-架构分层": ["Slime-架构分层", "Slime 架构分层", "concept"],
      "Slime-03-关键概念": ["Slime-关键概念", "Slime 关键概念", "concept"],
      "Slime-04-导读路径": ["Slime-学习路径", "Slime 学习路径", "guide"],
      "Slime-05-文件地图": ["Slime-源码地图", "Slime 源码地图", "reference"],
      "Slime-模块依赖图": ["Slime-模块依赖图", "Slime 模块依赖图", "map"],
      "Slime-术语表": ["Slime-术语表", "Slime 术语表", "reference"],
      "Slime-业务域流程": ["Slime-业务流程", "Slime 业务流程", "walkthrough"],
      "全链路RL训练追踪": ["Slime-RL训练全链路", "Slime RL 训练全链路", "walkthrough"],
      "与SGLang阅读对照": ["Slime与SGLang-阅读对照", "Slime 与 SGLang 阅读对照", "map"],
      "Slime-90-总结复盘-00-MOC": ["Slime-总结复盘", "Slime 总结复盘", "map"],
      "Slime-90-总结复盘-01-复杂度热点": ["Slime-复杂度热点", "Slime 复杂度热点", "reference"],
      "Slime-90-总结复盘-02-可观测与CI": ["Slime-可观测性与CI", "Slime 可观测性与 CI", "reference"],
      "Slime-90-总结复盘-03-未独立成专题导读": ["Slime-补充主题", "Slime 补充主题", "reference"],
      "Slime-90-总结复盘-04-checkpoint": ["Slime-综合学习检查", "Slime 综合学习检查", "exercise"],
      "Slime-模板说明": ["Slime-专题模板", "Slime 专题模板", "template"],
    },
  },
  "flash-attn_reading": {
    prefix: "FlashAttention",
    framework: "flash-attn",
    baseline: "002cce0",
    dirs: {
      "方法论": "阅读方法",
      "FA3-FA4": "新架构实现",
      "Hopper-CuTe": "Hopper与CuTe",
    },
    special: {
      "FlashAttention源码阅读指南": ["FlashAttention学习指南", "FlashAttention 学习指南", "guide"],
      "FlashAttention-00-导读与总览-00-MOC": ["FlashAttention-导读与总览", "FlashAttention 导读与总览", "map"],
      "FlashAttention-00-零基础先修": ["FlashAttention-零基础先修", "FlashAttention 零基础先修", "concept"],
      "FlashAttention-01-项目总览": ["FlashAttention-项目总览", "FlashAttention 项目总览", "concept"],
      "FlashAttention-02-架构分层": ["FlashAttention-架构分层", "FlashAttention 架构分层", "concept"],
      "FlashAttention-03-关键概念": ["FlashAttention-关键概念", "FlashAttention 关键概念", "concept"],
      "FlashAttention-04-导读路径": ["FlashAttention-学习路径", "FlashAttention 学习路径", "guide"],
      "FlashAttention-05-文件地图": ["FlashAttention-源码地图", "FlashAttention 源码地图", "reference"],
      "FlashAttention-代际演进": ["FlashAttention-代际演进", "FlashAttention 代际演进", "concept"],
      "FlashAttention-全链路Attention追踪": ["FlashAttention-前向全链路", "FlashAttention 前向全链路", "walkthrough"],
      "FlashAttention-术语表": ["FlashAttention-术语表", "FlashAttention 术语表", "reference"],
      "FlashAttention-四代增量全景": ["FlashAttention-版本演进全景", "FlashAttention 版本演进全景", "concept"],
      "FlashAttention-90-总结复盘-00-MOC": ["FlashAttention-总结复盘", "FlashAttention 总结复盘", "map"],
      "FA01-FlashAttention-1-算法原点": ["FlashAttention-算法原点", "FlashAttention 算法原点", "concept"],
      "FA04-FA2-版本增量与新特性": ["FlashAttention-FA2版本演进", "FlashAttention FA2 版本演进", "concept"],
      "FA06-FA3-Hopper增量": ["FlashAttention-FA3-Hopper演进", "FlashAttention FA3 Hopper 演进", "concept"],
      "FA06-FA4-CuTeDSL增量": ["FlashAttention-FA4-CuTeDSL演进", "FlashAttention FA4 CuTeDSL 演进", "concept"],
      "FlashAttention-模板说明": ["FlashAttention-专题模板", "FlashAttention 专题模板", "template"],
    },
  },
};

const META_FILES = {
  "obsidian-graph-presets.md": "Obsidian图谱使用指南.md",
  "obsidian-syntax-rules.md": "Obsidian知识库规范.md",
  "source-reading-writing-standard.md": "源码阅读写作标准.md",
  "source-reading-migration-plan.md": "知识库重构计划.md",
  "source-reading-quality-audit-2026-07-06.md": "源码阅读质量审计-2026-07-06.md",
  "coverage-audit-2026-07-04.md": "内容覆盖审计-2026-07-04.md",
  "audit_slime_out.txt": "Slime审计输出.txt",
  "fix-checkpoints.mjs": "normalize_learning_checks.mjs",
};

const MAP_FILES = {
  "batch-stats.md": "专题统计.md",
  "cross-library-map.md": "三框架知识地图.md",
  "doc-type-map.md": "文档类型地图.md",
  "dual-library-path.md": "AI-Infra联合学习路径.md",
  "flash-attn-module-board.md": "FlashAttention专题看板.md",
  "graph-hub.md": "关系图谱指南.md",
  "home.md": "知识地图首页.md",
  "module-board.md": "SGLang专题看板.md",
  "slime-module-board.md": "Slime专题看板.md",
};

function walk(dir, out = []) {
  if (!fs.existsSync(dir)) return out;
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

function stripCode(value) {
  return value.replace(/^Slime-/, "").replace(/^FlashAttention-/, "").replace(/^FA\d+-/, "").replace(/^\d+-/, "");
}

function semanticSegment(segment, cfg) {
  if (segment === "_TEMPLATE") return "模板";
  const clean = stripCode(segment);
  return cfg.dirs[clean] || clean;
}

function existingDocType(text) {
  const fm = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!fm) return "reference";
  const m = fm[1].match(/^doc_type:\s*([^\r\n]+)$/m);
  return TYPE_MAP.get(m?.[1]?.trim()) || m?.[1]?.trim() || "reference";
}

function moduleDisplay(value, cfg) {
  const clean = stripCode(value);
  return cfg.dirs[clean] || clean;
}

function readerDestination(abs, rootName, cfg) {
  const oldRel = path.relative(path.join(VAULT, rootName), abs);
  const oldDir = path.dirname(oldRel);
  const ext = path.extname(abs);
  const oldBase = path.basename(abs, ext);
  const newDirs = oldDir === "." ? [] : oldDir.split(path.sep).map((s) => semanticSegment(s, cfg));
  const text = ext === ".md" ? fs.readFileSync(abs, "utf8") : "";

  let newBase;
  let title;
  let type = ext === ".md" ? existingDocType(text) : "asset";

  if (cfg.special[oldBase]) {
    [newBase, title, type] = cfg.special[oldBase];
  } else {
    const role = oldBase.match(/^(?:Slime-)?(.+?)-(00-MOC|01-核心概念|02-源码走读|03-数据流与交互|04-关键问题|05-checkpoint)$/);
    if (role) {
      const module = moduleDisplay(role[1], cfg);
      const suffix = ROLE_SUFFIX.get(role[2]);
      newBase = `${cfg.prefix}-${module}${suffix ? `-${suffix}` : ""}`;
      title = `${module}${suffix ? ` · ${suffix}` : ""}`;
      type = TYPE_MAP.get(role[2].split("-").slice(1).join("-")) || type;
      if (role[2] === "00-MOC") type = "map";
      if (role[2] === "01-核心概念") type = "concept";
      if (role[2] === "02-源码走读") type = "walkthrough";
      if (role[2] === "03-数据流与交互") type = "dataflow";
      if (role[2] === "04-关键问题") type = "troubleshooting";
      if (role[2] === "05-checkpoint") type = "exercise";
    } else {
      const semantic = moduleDisplay(oldBase, cfg);
      newBase = semantic.startsWith(`${cfg.prefix}-`) || semantic === cfg.prefix ? semantic : `${cfg.prefix}-${semantic}`;
      title = newBase.replaceAll("-", " ");
    }
  }

  const newName = `${newBase}${ext}`;
  const newRel = path.join(rootName, ...newDirs, newName);
  const topic = newDirs.at(-1) || "总览";
  return { oldAbs: abs, oldRel: path.join(rootName, oldRel), newRel, oldBase, newBase, title, type, topic, cfg };
}

function learningRole(type) {
  if (["guide", "map", "concept", "walkthrough"].includes(type)) return "core";
  if (type === "troubleshooting") return "debug";
  if (type === "exercise") return "practice";
  return "reference";
}

function yamlQuote(value) {
  return JSON.stringify(value);
}

function standardizeReaderNote(text, item) {
  const withoutFm = text.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, "");
  const body = withoutFm.replace(/^#\s+.*$/m, `# ${item.title}`);
  const frontmatter = [
    "---",
    `title: ${yamlQuote(item.title)}`,
    `type: ${item.type}`,
    `framework: ${item.cfg.framework}`,
    `topic: ${yamlQuote(item.topic)}`,
    `learning_role: ${learningRole(item.type)}`,
    `source_baseline: ${yamlQuote(item.cfg.baseline)}`,
    "tags:",
    `  - framework/${item.cfg.framework}`,
    `  - content/${item.type}`,
    "  - source-reading",
    "updated: 2026-07-10",
    "---",
    "",
  ].join("\n");
  return `${frontmatter}${body.replace(/^\s+/, "")}`;
}

const items = [];
for (const [rootName, cfg] of Object.entries(LIBRARIES)) {
  for (const file of walk(path.join(VAULT, rootName))) {
    items.push(readerDestination(file, rootName, cfg));
  }
}

for (const file of walk(path.join(VAULT, "90_meta"))) {
  const name = path.basename(file);
  items.push({
    oldAbs: file,
    oldRel: path.join("90_meta", name),
    newRel: path.join("maintenance", META_FILES[name] || name),
    oldBase: path.basename(name, path.extname(name)),
    newBase: path.basename(META_FILES[name] || name, path.extname(META_FILES[name] || name)),
  });
}

for (const file of walk(path.join(VAULT, "91_dashboard"))) {
  const name = path.basename(file);
  items.push({
    oldAbs: file,
    oldRel: path.join("91_dashboard", name),
    newRel: path.join("knowledge_maps", MAP_FILES[name] || name),
    oldBase: path.basename(name, path.extname(name)),
    newBase: path.basename(MAP_FILES[name] || name, path.extname(MAP_FILES[name] || name)),
  });
}

const targets = new Map();
const collisions = [];
for (const item of items) {
  const key = item.newRel.toLowerCase();
  if (targets.has(key)) collisions.push([targets.get(key).oldRel, item.oldRel, item.newRel]);
  targets.set(key, item);
}

const basenameTargets = new Map();
for (const item of items.filter((x) => path.extname(x.newRel) === ".md")) {
  const key = item.newBase.toLowerCase();
  if (basenameTargets.has(key)) collisions.push([basenameTargets.get(key).oldRel, item.oldRel, item.newBase]);
  basenameTargets.set(key, item);
}

if (collisions.length > 0) {
  console.error("Rename collisions detected:");
  for (const row of collisions) console.error(row.join(" -> "));
  process.exit(1);
}

console.log(`Planned file moves: ${items.length}`);
for (const item of items.filter((x) => x.oldRel !== x.newRel).slice(0, 40)) {
  console.log(`${item.oldRel.replaceAll("\\", "/")} -> ${item.newRel.replaceAll("\\", "/")}`);
}
if (!APPLY) {
  console.log("Dry run only. Use --apply to perform the migration.");
  process.exit(0);
}

for (const item of items) {
  const dest = path.join(VAULT, item.newRel);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  if (path.resolve(item.oldAbs) !== path.resolve(dest)) fs.renameSync(item.oldAbs, dest);
  item.newAbs = dest;
}

const fullPathMap = new Map();
const basenameMap = new Map();
for (const item of items) {
  const oldRel = item.oldRel.replaceAll("\\", "/");
  const newRel = item.newRel.replaceAll("\\", "/");
  fullPathMap.set(oldRel, newRel);
  fullPathMap.set(oldRel.replace(/\.md$/i, ""), newRel.replace(/\.md$/i, ""));
  if (item.oldBase && item.newBase && item.oldBase !== item.newBase) basenameMap.set(item.oldBase, item.newBase);
}

function replaceWikiLinks(text) {
  return text.replace(/\[\[([^\]]+)\]\]/g, (whole, inner) => {
    const pipe = inner.indexOf("|");
    const rawTarget = pipe === -1 ? inner : inner.slice(0, pipe);
    const alias = pipe === -1 ? "" : inner.slice(pipe);
    const hash = rawTarget.indexOf("#");
    const target = hash === -1 ? rawTarget : rawTarget.slice(0, hash);
    const anchor = hash === -1 ? "" : rawTarget.slice(hash);
    const normalized = target.replaceAll("\\", "/");
    const replacement = fullPathMap.get(normalized) || basenameMap.get(normalized) || basenameMap.get(path.basename(normalized));
    if (!replacement) return whole;
    return `[[${replacement}${anchor}${alias}]]`;
  });
}

const editableRoots = ["index.md", "README.md", "AGENTS.md", "sglang_reading", "slime_reading", "flash-attn_reading", "maintenance", "knowledge_maps"];
const editableFiles = [];
for (const entry of editableRoots) {
  const abs = path.join(VAULT, entry);
  if (!fs.existsSync(abs)) continue;
  if (fs.statSync(abs).isFile()) editableFiles.push(abs);
  else editableFiles.push(...walk(abs));
}

const rawPairs = [...fullPathMap.entries()].sort((a, b) => b[0].length - a[0].length);
for (const file of editableFiles) {
  if (!/\.(?:md|mjs|txt|base|json)$/i.test(file)) continue;
  let text = fs.readFileSync(file, "utf8");
  const item = items.find((x) => x.newAbs && path.resolve(x.newAbs) === path.resolve(file));
  if (item?.cfg && path.extname(file) === ".md") text = standardizeReaderNote(text, item);
  text = replaceWikiLinks(text);
  for (const [oldValue, newValue] of rawPairs) text = text.split(oldValue).join(newValue);
  text = text.split("maintenance/").join("maintenance/").split("maintenance\\").join("maintenance\\");
  text = text.split("knowledge_maps/").join("knowledge_maps/").split("knowledge_maps\\").join("knowledge_maps\\");
  fs.writeFileSync(file, text, "utf8");
}

for (const oldRoot of ["90_meta", "91_dashboard", ...Object.keys(LIBRARIES)]) {
  const root = path.join(VAULT, oldRoot);
  if (!fs.existsSync(root)) continue;
  const dirs = [root, ...walkDirectories(root)].sort((a, b) => b.length - a.length);
  for (const dir of dirs) {
    try {
      if (fs.readdirSync(dir).length === 0 && dir !== root) fs.rmdirSync(dir);
    } catch {}
  }
  if (["90_meta", "91_dashboard"].includes(oldRoot) && fs.existsSync(root) && fs.readdirSync(root).length === 0) fs.rmdirSync(root);
}

console.log("Semantic naming migration completed.");

function walkDirectories(dir, out = []) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (!ent.isDirectory()) continue;
    const full = path.join(dir, ent.name);
    out.push(full);
    walkDirectories(full, out);
  }
  return out;
}
