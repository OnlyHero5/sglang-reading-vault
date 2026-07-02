---
type: dashboard
title: "SGLang 阅读仪表盘"
tags:
  - dashboard
  - sglang/meta
cssclasses:
  - 91_dashboard
created: 2026-07-02
updated: 2026-07-02
---

# SGLang 源码阅读 · 可视化入口

> Dataview 仪表盘 + 关系图谱预设已写入 `.obsidian/graph.json`。  
> 打开 **Graph View** 即可看到按文档类型着色的节点（如 `07-Scheduler-核心概念`）。

## 快捷导航

| 视图 | 说明 |
|------|------|
| [[91_dashboard/module-board|模块总览]] | 32 批模块 MOC + 文档类型统计 |
| [[91_dashboard/batch-stats|批次统计]] | 按批次分组文档数量 |
| [[91_dashboard/doc-type-map|文档类型分布]] | concept / walkthrough / dataflow 等 |
| [[91_dashboard/graph-hub|关系图谱指南]] | 图谱过滤式与颜色图例 |
| [[progress|阅读进度]] | 人工维护的进度摘要 |
| [[obsidian-graph-presets|图谱预设（Markdown）]] | 备用过滤式参考 |

## 库内统计

```dataview
TABLE WITHOUT ID
  "模块 MOC" AS 类型,
  length(rows) AS 数量
FROM "sglang_reading"
WHERE type = "module-moc"
GROUP BY true
```

```dataview
TABLE WITHOUT ID
  "批次正文" AS 类型,
  length(rows) AS 数量
FROM "sglang_reading"
WHERE type = "batch-doc"
GROUP BY true
```

```dataview
TABLE WITHOUT ID
  doc_type AS "文档类型",
  length(rows) AS 篇数
FROM "sglang_reading"
WHERE type = "batch-doc"
GROUP BY doc_type
SORT doc_type ASC
```

## 阶段 MOC

```dataview
TABLE title AS 标题
FROM "sglang_reading"
WHERE type = "stage-moc"
SORT file.name ASC
```
