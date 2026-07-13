---
title: "AI Infra 知识库维护指南"
type: guide
framework: vault
topic: "维护"
learning_role: reference
tags:
  - maintenance
  - agent
  - source-reading
updated: 2026-07-13
---

# AI Infra 知识库维护指南

> AI 代理进入仓库后先读本页与 [[index]]。

## 你为什么要读

目录能告诉你文件放在哪里，却不能替你决定先学什么。本页把 维护 按读者任务重新组织：先建立主线，再进入源码证据、排障和实验，帮助你在宏观架构与微观实现之间来回切换。

## 目录边界

| 目录 | 用途 | 权限 |
|------|------|------|
| `AI-Infra课程/` | 公共基础、系统主线、贯穿案例、实验 | 可写 |
| `sglang_reading/` | SGLang 阅读笔记 | 可写 |
| `slime_reading/` | Slime 阅读笔记 | 可写 |
| `flash-attn_reading/` | FlashAttention 阅读笔记 | 可写 |
| `knowledge_maps/` | 知识地图与 Bases | 可写 |
| `maintenance/` | 规范、审计和迁移工具 | 可写 |
| `sglang/`、`slime/`、`flash-attn/` | upstream 源码基线 | 只读 |
| `.obsidian/` | 用户配置 | 未明确要求时不修改 |

## 启动协议

1. [[index]]
2. [[AI-Infra入门课程]]
3. 对应框架指南：[[SGLang学习指南]]、[[Slime学习指南]]、[[FlashAttention学习指南]]
4. [[maintenance/Obsidian知识库规范]]
5. [[maintenance/源码阅读写作标准]]

## 命名原则

- 文件名必须全库唯一，并使用读者能理解的语义名称。
- 禁止在读者文件名、目录、标题、aliases、tags 中使用维护批次号。
- 技术版本号可以保留，例如 FA2、FA3、FP8、Qwen3。
- 框架专题统一使用框架前缀，例如 `SGLang-Scheduler-源码走读.md`。
- aliases 只保存真实缩写、中文名或英文名，不保存旧文件编号。

## Properties

读者文档至少包含：

```yaml
title: "文档标题"
type: concept
framework: sglang
topic: "Scheduler"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
```

允许的 `type`：`guide`、`map`、`concept`、`walkthrough`、`dataflow`、`troubleshooting`、`exercise`、`reference`、`dashboard`、`template`。

`learning_role`：

- `core`：首次学习主线
- `reference`：按需查阅
- `debug`：故障定位
- `practice`：实验和学习检查

## 写作原则

- 正文中文；英文用于标识符、协议和标准术语。
- 核心主线采用：读者任务 → 心理模型 → 贯穿对象 → 源码证据 → 运行验证 → 复盘。
- 每张源码证据卡只证明一个判断，来源写在代码块第一行。
- 类比必须映射到源码对象，并说明失效边界。
- Walkthrough 不按文件顺序堆代码，必须沿请求、样本、tensor 或权重生命周期组织。
- Troubleshooting 必须包含症状、可能原因、源码入口、操作和预期。
- Exercise 必须提供可执行步骤；无法运行时给静态替代和环境限制。

## Obsidian 原则

- 文件夹负责物理归档，Wikilinks 和 Backlinks 负责语义关系。
- 核心入口使用少量 Map，不为每个概念制造额外导航页。
- 动态列表优先使用 Bases；统计可使用 Dataview，但不能依赖维护编号。
- 首次阅读优先 Bookmarks 和 Local Graph；全局图谱仅用于探索。
- Properties 保存原子结构化信息，正文保存解释和证据。

## 禁止

- 修改 upstream 源码。
- 编造源码行为、函数签名、行号和性能数字。
- 用固定文档数量或源码段数量作为完成标准。
- 在读者入口暴露迁移说明、派工信息和维护编号。
- 未给版本、硬件和 workload 就写框架性能阈值。

## 发布检查

```bash
node maintenance/audit_wikilinks.mjs
node maintenance/audit_source_evidence.mjs
node maintenance/audit_markdown_quality.mjs
git -c core.autocrlf=false diff --check
```

检查项：全局唯一文件名、零断链、源码引用有效、正式源码卡与标注行段逐行一致、单一 H1、Mermaid 合规、无维护编号、核心文档有读者任务、验证包含操作和预期、改动无空白错误或冲突标记。自动检查只是发布下限；本轮修改的文档还必须完成全文语义复读。
