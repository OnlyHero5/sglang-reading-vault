---
type: dashboard
title: "源码阅读 Vault 仪表盘"
tags:
  - dashboard
  - sglang/meta
  - slime/meta
cssclasses:
  - 91_dashboard
created: 2026-07-02
updated: 2026-07-03
---

# 源码阅读 Vault · 可视化入口

> Dataview 仪表盘。关系图谱见 [[91_dashboard/graph-hub]]。

## 快捷导航

| 视图 | 说明 |
|------|------|
| **[[91_dashboard/dual-library-path|双库联合路径]]** | 推理 + RL 阅读顺序 |
| **[[91_dashboard/cross-library-map|跨库专题对照]]** | 专题级跳转 |
| [[SGLang源码阅读指南]] | SGLang 总索引 |
| [[Slime源码阅读指南]] | Slime 总索引 |
| [[91_dashboard/module-board|SGLang 模块总览]] | 专题 MOC 列表 |
| [[91_dashboard/slime-module-board|Slime 模块总览]] | 专题 MOC 列表 |
| [[91_dashboard/graph-hub|关系图谱指南]] | 图谱过滤与颜色 |
| [[90_meta/obsidian-graph-presets|图谱预设]] | 备用过滤式 |
| [[index]] | Vault 首页 |
| [[91_dashboard/batch-stats|专题统计]] | 每个专题的文档数与模块分布 |
| [[91_dashboard/doc-type-map|文档类型分布]] | doc_type 计数与图谱色对照 |

---

## SGLang 库内统计

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
  doc_type AS "文档类型",
  length(rows) AS 篇数
FROM "sglang_reading"
WHERE type = "batch-doc"
GROUP BY doc_type
SORT doc_type ASC
```

## Slime 库内统计

```dataview
TABLE WITHOUT ID
  "专题正文" AS 类型,
  length(rows) AS 数量
FROM "slime_reading"
WHERE type = "batch-doc"
GROUP BY true
```

```dataview
TABLE WITHOUT ID
  doc_type AS "文档类型",
  length(rows) AS 篇数
FROM "slime_reading"
WHERE type = "batch-doc"
GROUP BY doc_type
SORT doc_type ASC
```

## 阶段 MOC · SGLang

```dataview
TABLE title AS 标题
FROM "sglang_reading"
WHERE type = "stage-moc"
SORT file.name ASC
```

## 阶段 MOC · Slime

```dataview
TABLE title AS 标题
FROM "slime_reading"
WHERE contains(file.name, "-00-MOC") AND (type = "stage-moc" OR type = "batch-doc" OR type = "phase-moc")
SORT file.name ASC
LIMIT 20
```
