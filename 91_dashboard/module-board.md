---
type: dashboard
title: "SGLang 模块总览"
tags:
  - dashboard
  - sglang/meta
cssclasses:
  - 91_dashboard
updated: 2026-07-02
---

# SGLang 模块总览

```dataview
TABLE batch AS "专题序号", title AS "标题", file.link AS "入口"
FROM "sglang_reading"
WHERE type = "module-moc"
SORT number(batch) ASC
```

## 按阶段浏览

### 方法论 · 启动 · 调度

```dataview
LIST
FROM "sglang_reading"
WHERE type = "module-moc" AND (number(batch) <= 10 OR batch = "01")
SORT number(batch) ASC
```

### 模型执行 · 内存 · Attention

```dataview
LIST
FROM "sglang_reading"
WHERE type = "module-moc" AND number(batch) >= 11 AND number(batch) <= 19
SORT number(batch) ASC
```

### 高级特性 · 扩展组件 · 运维

```dataview
LIST
FROM "sglang_reading"
WHERE type = "module-moc" AND number(batch) >= 20
SORT number(batch) ASC
```

## 单模块五件套检查

```dataview
TABLE doc_type AS "类型", file.link AS "文档"
FROM "sglang_reading"
WHERE module = "07-Scheduler" AND (type = "batch-doc" OR type = "module-moc")
SORT doc_type ASC
```

> 将最后一行 `module = "07-Scheduler"` 改为其他模块名即可抽查。

## 双库交叉

| 文档 | 说明 |
|------|------|
| [[91_dashboard/dual-library-path|双库联合路径]] | 推理 + RL 推荐阅读顺序 |
| [[91_dashboard/cross-library-map|跨库专题对照]] | 推理 + RL 专题映射 |
| [[91_dashboard/slime-module-board|Slime 模块总览]] | 模块 MOC |
