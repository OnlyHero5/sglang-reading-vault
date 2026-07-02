---
type: dashboard
title: "批次统计"
tags:
  - dashboard
  - sglang/meta
cssclasses:
  - 91_dashboard
updated: 2026-07-02
---

# 批次统计

## 每批文档数（含 MOC + 五件套）

```dataview
TABLE WITHOUT ID
  batch AS "批次",
  length(rows) AS "文档数",
  rows.module[0] AS "代表模块"
FROM "sglang_reading"
WHERE type = "batch-doc" OR type = "module-moc"
GROUP BY batch
SORT number(batch) ASC
```

## 批次 × 文档类型矩阵

```dataview
TABLE WITHOUT ID
  batch AS "批次",
  length(filter(rows, (r) => r.doc_type = "concept")) AS "概念",
  length(filter(rows, (r) => r.doc_type = "walkthrough")) AS "走读",
  length(filter(rows, (r) => r.doc_type = "dataflow")) AS "数据流",
  length(filter(rows, (r) => r.doc_type = "faq")) AS "FAQ",
  length(filter(rows, (r) => r.doc_type = "checkpoint")) AS "验收"
FROM "sglang_reading"
WHERE type = "batch-doc"
GROUP BY batch
SORT number(batch) ASC
```

## 最近更新

```dataview
TABLE batch, doc_type, updated
FROM "sglang_reading"
WHERE type = "batch-doc" OR type = "module-moc"
SORT updated DESC
LIMIT 15
```
