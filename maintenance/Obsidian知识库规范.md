---
title: "Obsidian 知识库规范"
type: guide
framework: cross-framework
topic: "知识库维护"
learning_role: reference
tags:
  - maintenance
  - obsidian
updated: 2026-07-13
---

# Obsidian 知识库规范

## 设计原则

本 Vault 使用语义文件名、统一 Properties、任务型入口、双链和 Bases。目录负责物理归档，双链负责知识关系，Properties/Bases 负责动态汇总。

维护编号、排序前缀、旧文件名 aliases 和固定文档套件均不进入读者界面。一个主题只创建真正服务读者任务的文档。

## 文件命名

文件名必须全库唯一，并直接表达框架、主题和文档职责。

| 职责 | 命名示例 |
|------|----------|
| 专题入口 | `SGLang-Scheduler.md` |
| 核心概念 | `SGLang-Scheduler-核心概念.md` |
| 源码走读 | `SGLang-Scheduler-源码走读.md` |
| 数据流 | `SGLang-Scheduler-数据流.md` |
| 排障 | `SGLang-Scheduler-排障指南.md` |
| 学习检查 | `SGLang-Scheduler-学习检查.md` |

读者专题禁止使用排序数字前缀、泛化的 `README.md`、重复的 `核心概念.md`，也不要用历史编号作为 alias。仓库根目录用于说明项目打开方式和目录边界的 `README.md` 是唯一例外。

## Properties

读者文档使用以下字段：

```yaml
---
title: "SGLang Scheduler 源码走读"
type: walkthrough
framework: sglang
topic: "Scheduler"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/walkthrough
  - source-reading
updated: 2026-07-10
---
```

`type` 使用：`guide`、`map`、`concept`、`walkthrough`、`dataflow`、`troubleshooting`、`exercise`、`reference`、`dashboard`、`template`。

`learning_role` 使用：`core`、`reference`、`debug`、`practice`。`framework` 使用 `sglang`、`slime`、`flash-attn` 或 `cross-framework`。

Properties 必须存放可筛选事实，不存放发布批次、迁移阶段或派工状态。

## Tags

Tags 只表达低变化、可聚合的分类：

- `framework/sglang`、`framework/slime`、`framework/flash-attn`
- `content/concept`、`content/walkthrough`、`content/troubleshooting`
- `source-reading`、`ai-infra`、`obsidian/graph`

不要用 tag 重复 `topic` 的所有细粒度值，也不要创建排序编号 tag。

## 双链与 aliases

- 优先使用 `[[唯一文件名]]`；只有显示文本确实更自然时才写 alias。
- 路径式链接用于明确跨目录目标，如 `[[maintenance/源码阅读写作标准]]`。
- aliases 只保留真实同义词、缩写和常用英文名。
- 代码块和行内代码中的 `[[...]]` 不是知识链接，审计脚本会忽略。
- 新增或重命名文件后运行 `node maintenance/audit_wikilinks.mjs`。

## Bases

Bases 用于动态汇总，不手工维护重复清单。现有视图位于 `knowledge_maps/`，覆盖核心课程、源码走读、排障、实验以及三个框架。

新增 Base 时应：

- 用文件夹、`framework`、`type`、`learning_role` 过滤。
- 只展示当前任务需要的属性。
- 视图命名用读者语言，不使用维护编号。
- Base 负责发现内容，课程入口仍负责解释学习顺序和完成标准。

## Markdown 与 Mermaid

- 每篇只有一个 H1，标题层级不跳级。
- Mermaid 标签换行使用 `<br/>`，不使用字面量 `\n`。
- 单图只表达一个判断；节点过多时拆图。
- 源码来源放在代码块首行，不写成 Markdown 标题。
- 正文中文；函数、类、参数和协议名保留英文。

## 发布检查

```powershell
node maintenance/audit_wikilinks.mjs
node maintenance/audit_source_evidence.mjs
node maintenance/audit_markdown_quality.mjs
git -c core.autocrlf=false diff --check
```

同时确认：读者入口没有维护编号或派工话术，upstream 和 `.obsidian/` 未被意外修改，本轮修改文档已完成脱离编辑过程的全文语义复读。
