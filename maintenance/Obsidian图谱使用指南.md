---
title: "Obsidian 图谱使用指南"
type: guide
framework: cross-framework
topic: "知识库维护"
learning_role: reference
tags:
  - maintenance
  - obsidian/graph
updated: 2026-07-13
---

# Obsidian 图谱使用指南

## 维护目标

图谱用于发现关系和孤岛，不承担课程排序。知识库的稳定导航由入口页、Properties、Bases 和语义双链共同提供。

## 推荐工作方式

| 任务 | 首选工具 | 原因 |
|------|----------|------|
| 首次学习 | Bookmarks + 课程入口 | 路径稳定、认知负担低 |
| 查同类内容 | Bases | 可按属性筛选和排序 |
| 查谁依赖当前笔记 | Backlinks | 直接显示入链 |
| 查当前主题邻居 | Local Graph | 局部关系清晰 |
| 查全库孤岛或异常簇 | Global Graph | 适合维护审计 |
| 查标识符和错误文本 | Search | 比图谱更精确 |

## 过滤预设

```text
(path:AI-Infra课程 OR path:sglang_reading OR path:slime_reading OR path:flash-attn_reading OR path:knowledge_maps) -path:模板
```

```text
[learning_role:core] (path:AI-Infra课程 OR path:sglang_reading OR path:slime_reading OR path:flash-attn_reading)
```

```text
[framework:sglang] [type:walkthrough]
```

```text
[framework:slime] ([type:walkthrough] OR [type:dataflow])
```

```text
[framework:flash-attn] ([type:concept] OR [type:walkthrough])
```

```text
([type:troubleshooting] OR [type:exercise]) -path:模板
```

## 颜色策略

颜色组保持在六组以内。优先按 `framework` 区分三框架和跨框架内容，再为 `troubleshooting`、`exercise` 增加任务颜色。不要同时按文件夹、标签、类型和角色重复着色。

## 关系维护

- 入口页链接主线、实验、排障和参考页。
- 专题入口链接核心概念、源码走读、数据流、排障和学习检查中实际存在的文档。
- 深度文档链接直接前置知识、上游生产者、下游消费者和验证入口。
- aliases 只保存真实同义词或常用英文名，不保存旧文件名和排序编号。
- 文件夹表达归档归属，双链表达语义关系；不要为图谱形状制造无意义链接。

## 审计方法

全局图谱重点看三类问题：没有入链的孤岛、只连模板或维护页的伪入口、跨框架主线中断。断链由 `node maintenance/audit_wikilinks.mjs` 检查，不能只靠图谱目测。

`.obsidian/` 属于用户配置：未获得明确授权时不修改。若需要调整 Graph 配置，应先在界面中验证过滤式，再由用户明确授权修改配置文件。
