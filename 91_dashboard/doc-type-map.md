---
type: dashboard
title: "文档类型分布"
tags:
  - dashboard
  - sglang/meta
cssclasses:
  - 91_dashboard
updated: 2026-07-02
---

# 文档类型分布

> 与 Graph View 颜色分组一一对应（见 [[91_dashboard/graph-hub]]）。

## 类型计数

```dataview
TABLE WITHOUT ID
  doc_type AS "类型",
  length(rows) AS "篇数",
  choice(doc_type = "moc", "🟢 模块入口", choice(doc_type = "concept", "🩵 核心概念", choice(doc_type = "walkthrough", "🟣 源码走读", choice(doc_type = "dataflow", "🟠 数据流", choice(doc_type = "faq", "🔴 FAQ", "⚪ checkpoint"))))) AS "图谱色"
FROM "sglang_reading"
WHERE type = "batch-doc" OR type = "module-moc"
GROUP BY doc_type
SORT length(rows) DESC
```

## 核心概念层（Graph 过滤：`tag:#sglang/doc/concept`）

```dataview
TABLE batch, module, title
FROM "sglang_reading"
WHERE doc_type = "concept"
SORT number(batch) ASC
```

## 数据流层（Graph 过滤：`tag:#sglang/doc/dataflow`）

```dataview
TABLE batch, module, title
FROM "sglang_reading"
WHERE doc_type = "dataflow"
SORT number(batch) ASC
```

## 源码走读层（Graph 过滤：`tag:#sglang/doc/walkthrough`）

```dataview
TABLE batch, module, title
FROM "sglang_reading"
WHERE doc_type = "walkthrough"
SORT number(batch) ASC
```
