---
type: dashboard
title: "阅读进度"
tags:
  - dashboard
  - sglang/index-layer
updated: 2026-07-02
---

# 阅读进度

> 基于 frontmatter `batch` / `doc_type` 自动统计。人工进度见 [[progress]]。

## 批次覆盖

```dataview
TABLE WITHOUT ID
  batch AS "批次",
  length(rows) AS "文档数",
  length(filter(rows, (r) => r.doc_type = "moc")) AS "MOC",
  length(filter(rows, (r) => r.doc_type = "walkthrough")) AS "走读"
FROM "sglang_reading"
WHERE batch AND !contains(file.path, "_archive") AND !contains(file.path, "_TEMPLATE")
GROUP BY batch
SORT number(batch) ASC
```

## 最近更新的模块文档

```dataview
TABLE module AS "模块", doc_type AS "类型", updated AS "更新"
FROM "sglang_reading"
WHERE module AND doc_type AND updated
SORT updated DESC
LIMIT 20
```

## 索引层文档

```dataview
TABLE WITHOUT ID
  link(file.link, title) AS "文档",
  file.folder AS "目录"
FROM "sglang_reading/07-总结与索引"
WHERE !contains(file.path, "_archive")
SORT file.name ASC
```

## 阶段 MOC

```dataview
LIST
FROM "sglang_reading"
WHERE contains(tags, "sglang/stage-moc")
SORT file.name ASC
```
