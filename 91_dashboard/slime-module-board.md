---
type: dashboard
title: "Slime 模块总览"
tags:
  - dashboard
  - slime/meta
cssclasses:
  - 91_dashboard
updated: 2026-07-03
---

# Slime 模块总览

```dataview
TABLE batch AS "专题序号", title AS "标题", file.link AS "入口"
FROM "slime_reading"
WHERE type = "module-moc"
SORT number(batch) ASC
```

## 按阶段浏览

### 方法论 · 启动 · Ray 编排

```dataview
LIST
FROM "slime_reading"
WHERE type = "module-moc" AND number(batch) <= 7
SORT number(batch) ASC
```

### Rollout 生成

```dataview
LIST
FROM "slime_reading"
WHERE type = "module-moc" AND number(batch) >= 8 AND number(batch) <= 16
SORT number(batch) ASC
```

### 训练后端 · 权重同步 · 扩展

```dataview
LIST
FROM "slime_reading"
WHERE type = "module-moc" AND number(batch) >= 17
SORT number(batch) ASC
```

## 单模块六件套检查

```dataview
TABLE doc_type AS "类型", file.link AS "文档"
FROM "slime_reading"
WHERE module = "08-RolloutManager" AND (type = "batch-doc" OR type = "module-moc")
SORT doc_type ASC
```

> 将 `module = "08-RolloutManager"` 改为其他模块名即可抽查。

## 双库交叉

| 文档 | 说明 |
|------|------|
| [[91_dashboard/cross-library-map|跨库专题对照]] | Slime ↔ SGLang 专题映射 |
