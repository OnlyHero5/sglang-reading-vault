---
type: dashboard
title: "关系图谱指南"
tags:
  - dashboard
  - sglang/meta
  - obsidian
  - graph
cssclasses:
  - 91_dashboard
updated: 2026-07-02
---

# 关系图谱指南

> **已写入** `.obsidian/graph.json`：默认过滤 + 颜色分组。  
> 打开 Obsidian 左侧 **关系图谱** 图标即可生效（若未刷新，请 `Ctrl+P` → Reload app）。

## 当前默认过滤（`.obsidian/graph.json`）

```text
path:sglang_reading -path:_TEMPLATE
```

> **勿用** `-path:sglang` — Obsidian 会误匹配 `sglang_reading`，导致图谱空白。

## 颜色图例（按 frontmatter 属性，非 tag）

| 查询 | 颜色 | 含义 |
|------|------|------|
| `[type:index-doc]` / `[type:index]` | 金 | 总结索引 / 总入口 |
| `[type:stage-moc]` | 蓝 | 阶段 MOC |
| `[doc_type:moc]` | 绿 | 模块 MOC |
| `[doc_type:concept]` | 青 | 核心概念 |
| `[doc_type:dataflow]` | 橙 | 数据流 |
| `[doc_type:walkthrough]` | 紫 | 源码走读 |
| `[doc_type:faq]` | 珊瑚 | FAQ |
| `[doc_type:checkpoint]` | 灰 | 验收清单 |
| `path:91_dashboard` | 浅蓝 | Dataview 仪表盘 |

> **勿在 graph.json 用** `tag:#sglang/doc/xxx` — 斜杠会被 Obsidian 解析截断，导致颜色组异常、图谱空白。

## 一键切换过滤式

复制到 Graph 搜索框：

**模块骨架（最简）**

```text
tag:#sglang/doc/moc OR tag:#sglang/stage-moc -path:_TEMPLATE
```

**概念学习**

```text
tag:#sglang/doc/concept -path:_TEMPLATE
```

**含 checkpoint 全量**

```text
path:sglang_reading -path:_TEMPLATE -path:sglang
```

## Local Graph 推荐起点

| 主题 | 笔记 |
|------|------|
| HTTP 全链路 | [[全链路请求追踪]] |
| 调度 | [[07-Scheduler-00-MOC]] |
| KV / Radix | [[15-RadixAttention-00-MOC]] |
| 总入口 | [[SGLang源码阅读指南]] |

更多预设见 [[obsidian-graph-presets]]。
