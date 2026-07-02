#!/usr/bin/env node
/** Second pass: reader-facing 批次 → 专题/模块 (keep ScheduleBatch etc.) */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const READING = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "sglang_reading");
const SKIP = new Set(["_archive", "_TEMPLATE", "_qa", "progress.md", "PLAN.md"]);

function fix(text) {
  let o = text;
  o = o.replace(/前置批次/g, "前置专题");
  o = o.replace(/下一批/g, "下一专题");
  o = o.replace(/上一批/g, "上一专题");
  o = o.replace(/## 与前后批次/g, "## 与相邻专题");
  o = o.replace(/与前后批次/g, "与相邻专题");
  o = o.replace(/对应批次/g, "对应专题");
  o = o.replace(/专题批次/g, "专题目录");
  o = o.replace(/批次深潜/g, "专题深读");
  o = o.replace(/建议深读批次/g, "建议深读专题");
  o = o.replace(/阅读批次/g, "阅读专题");
  o = o.replace(/\*\*批次：\*\*/g, "**专题：**");
  o = o.replace(/\*\*批次\*\*：/g, "**专题**：");
  o = o.replace(/入口批次/g, "入口专题");
  o = o.replace(/遗留问题（后续批次）/g, "遗留问题（后续专题）");
  o = o.replace(/后续批次/g, "后续专题");
  o = o.replace(/disaggregation 批次/g, "PD 分离专题");
  o = o.replace(/15\/16 批次/g, "RadixAttention / KV Cache 专题");
  o = o.replace(/09-ScheduleBatch-IO 批次/g, "[[09-ScheduleBatch-IO-00-MOC|ScheduleBatch-IO]]");
  o = o.replace(/entrypoints 批次/g, "HTTP Server 专题");
  o = o.replace(/未单列批次/g, "未单列专题");
  o = o.replace(/另开批次/g, "另开专题");
  o = o.replace(/批次外提及/g, "其他专题");
  o = o.replace(/批次相关/g, "相关专题");
  o = o.replace(/五个批次覆盖/g, "五个专题覆盖");
  o = o.replace(/严格按批次号/g, "严格按专题顺序");
  o = o.replace(/## 前序批次索引/g, "## 前序专题索引");
  o = o.replace(/批次编号与目录名不一致/g, "文件夹编号与专题名不一致");
  o = o.replace(/\| 批次编号对照 \|/g, "| 模块与目录对照 |");
  o = o.replace(/深读：批次 /g, "深读：");
  o = o.replace(/对应批次目录/g, "对应专题目录");
  o = o.replace(/跳转「对应批次目录」/g, "跳转「对应专题目录」");
  o = o.replace(/跳转专题批次/g, "跳转专题目录");
  o = o.replace(/比死记 26 步 order 更高效/g, "比死记 26 步顺序更高效");
  o = o.replace(/深读：\[\[/g, "深读 [[");
  o = o.replace(/title: "([^"]+)：源码走读"/g, (all, m) => {
    if (m.includes("·")) return all;
    const mod = MODULE_FIX[m] || m;
    return `title: "${mod} · 源码走读"`;
  });
  o = o.replace(/title: "([^"]+)：核心概念"/g, (all, m) => {
    if (m.includes("·")) return all;
    return `title: "${MODULE_FIX[m] || m} · 核心概念"`;
  });
  o = o.replace(/# 源码走读\n/g, (all, _, offset, s) => {
    // only fix if frontmatter has module - skip
    return all;
  });
  return o;
}

const MODULE_FIX = {
  "启动链路": "启动链路与 CLI",
  "HTTP Server": "HTTP Server 入口",
};

function walk(d) {
  let n = 0;
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    if (SKIP.has(e.name)) continue;
    const f = path.join(d, e.name);
    if (e.isDirectory()) n += walk(f);
    else if (e.name.endsWith(".md")) {
      const t = fs.readFileSync(f, "utf8");
      const nxt = fix(t);
      if (nxt !== t) {
        fs.writeFileSync(f, nxt, "utf8");
        n++;
      }
    }
  }
  return n;
}

console.log("Updated", walk(READING), "files");
