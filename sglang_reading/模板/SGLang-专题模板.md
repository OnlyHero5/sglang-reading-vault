---
title: "SGLang 专题模板"
type: template
framework: sglang
topic: "模板"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/template
  - source-reading
updated: 2026-07-11
---

# SGLang 专题模板

## 适用方式

先判断读者任务，再选择需要的文档。专题入口、核心概念、源码走读、数据流、排障指南和学习检查都采用语义文件名，但不要求每个主题机械创建全部类型。

| 读者任务 | 推荐文件 |
|----------|----------|
| 建立主题地图 | `SGLang-{主题}.md` |
| 理解术语和心理模型 | `SGLang-{主题}-核心概念.md` |
| 沿真实请求读源码 | `SGLang-{主题}-源码走读.md` |
| 看对象和边界变化 | `SGLang-{主题}-数据流.md` |
| 从症状定位问题 | `SGLang-{主题}-排障指南.md` |
| 验证掌握程度 | `SGLang-{主题}-学习检查.md` |

## Properties 模板

```yaml
---
title: "SGLang {主题} 源码走读"
type: walkthrough
framework: sglang
topic: "{主题}"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/walkthrough
  - source-reading
updated: 2026-07-11
---
```

`learning_role` 只使用 `core`、`reference`、`debug`、`practice`。源码走读通常是 `reference`；首次学习主线中的必读页才使用 `core`。

## 正文模板

```markdown
# SGLang {主题} 源码走读

## 读者任务

说明为什么要读、读完能排查或修改什么、本文贯穿哪类请求或状态。

## 先建立模型

用图或表说明对象、队列、进程边界、状态所有者和下游消费者。

## 贯穿场景

选择一条真实主线，例如 HTTP generate、prefill/decode、KV retract 或权重热更新。

## 主线走读

先解释系统压力和设计选择，再给逐字源码证据、执行逻辑、不变量与失败模式。

## 运行验证

- 操作：命令、日志、metric、测试或断点。
- 预期：说明正常与异常分别会看到什么。

## 复盘

总结可迁移结论，并链接直接前置知识、下游消费者和排障入口。
```

## 源码要求

- 修改解释前完整阅读本文引用的 SGLang upstream 文件。
- Python 来源行写成 `# 来源：path Lx-Ly`，并放在代码块首行。
- 不编造行为、签名、行号、动机或性能数据。
- 新增摘录后运行 `node maintenance/audit_source_evidence.mjs`。

完整规范见 [[maintenance/源码阅读写作标准]] 与 [[maintenance/Obsidian知识库规范]]。
