---
title: "AI Infra Reading Vault"
type: reference
framework: cross-framework
topic: "项目说明"
learning_role: reference
tags:
  - framework/cross-framework
  - content/reference
  - source-reading
updated: 2026-07-13
---

# AI Infra Reading Vault

中文 AI Infra 源码学习知识库，覆盖：

- SGLang：LLM 推理 serving runtime
- Slime：RL 后训练闭环
- FlashAttention：IO-aware attention kernel

## 推荐用法

使用 Obsidian 打开仓库后，从 [index.md](index.md) 或 [AI Infra 入门课程](AI-Infra课程/AI-Infra入门课程.md) 开始。

本项目分成三层：

| 层 | 用途 |
|----|------|
| 核心课程 | 公共基础、三条系统主线、贯穿案例和实验 |
| 框架专题 | 核心概念、源码走读、数据流、排障与学习检查 |
| 维护层 | 命名规范、写作标准和自动审计 |

读者不需要按目录顺序读完全部笔记。核心课程负责形成连续学习体验，框架专题作为深入和查阅材料。

三套源码在知识库中承担不同尺度的教学任务，不应被误读为固定运行依赖：FlashAttention 用于解释 attention kernel，并不保证某次 SGLang 运行实际命中本仓库实现；真实 backend 需由版本、配置、dispatch 和 profiler 共同确认。

## 入口

| 目标 | 文档 |
|------|------|
| AI Infra 从零入门 | [AI-Infra入门课程.md](AI-Infra课程/AI-Infra入门课程.md) |
| 三框架联合路径 | [AI-Infra联合学习路径.md](knowledge_maps/AI-Infra联合学习路径.md) |
| SGLang | [SGLang学习指南.md](sglang_reading/SGLang学习指南.md) |
| Slime | [Slime学习指南.md](slime_reading/Slime学习指南.md) |
| FlashAttention | [FlashAttention学习指南.md](flash-attn_reading/FlashAttention学习指南.md) |
| 动态知识地图 | [知识地图首页.md](knowledge_maps/知识地图首页.md) |

## Obsidian 结构

- Markdown 文件使用全库唯一的语义名称。
- Properties 使用 `framework`、`topic`、`type`、`learning_role` 描述内容。
- Bases 生成核心课程、框架内容、排障和实验视图。
- Wikilinks、Backlinks 和 Local Graph 表达知识关系。
- aliases 仅用于真实同义词或缩写，不保存旧编号。

## 仓库布局

```text
AI-Infra课程/          核心课程、贯穿案例、实验
sglang_reading/        SGLang 深度专题
slime_reading/         Slime 深度专题
flash-attn_reading/    FlashAttention 深度专题
knowledge_maps/        知识地图与 Obsidian Bases
maintenance/           规范、审计和迁移工具
sglang/ slime/ flash-attn/  upstream 只读基线
```

## 源码基线

| 框架 | commit |
|------|--------|
| SGLang | `70df09b` |
| Slime | `22cdc6e1` |
| FlashAttention | `002cce0` |

源码片段标注路径与行号。框架对比和性能结论必须同时记录版本、硬件、workload 与配置。

## 维护检查

```bash
node maintenance/audit_wikilinks.mjs
node maintenance/audit_source_evidence.mjs
node maintenance/audit_markdown_quality.mjs
git -c core.autocrlf=false diff --check
```

详细规则见 [Obsidian知识库规范.md](maintenance/Obsidian知识库规范.md) 和 [源码阅读写作标准.md](maintenance/源码阅读写作标准.md)。
