---
title: "Slime 专题模板"
type: template
framework: slime
topic: "模板"
learning_role: reference
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/template
  - source-reading
updated: 2026-07-11
---

# Slime 专题模板

## 适用方式

每篇文档必须回到 `generate → train → update_weights` 闭环，说明样本、Ray ObjectRef、actor 或权重在哪个边界改变形态。只创建解决实际读者任务的文档。

| 读者任务 | 推荐文件 |
|----------|----------|
| 建立主题地图 | `Slime-{主题}.md` |
| 理解术语和心理模型 | `Slime-{主题}-核心概念.md` |
| 沿闭环读源码 | `Slime-{主题}-源码走读.md` |
| 看样本、资源或权重流 | `Slime-{主题}-数据流.md` |
| 从症状定位问题 | `Slime-{主题}-排障指南.md` |
| 验证掌握程度 | `Slime-{主题}-学习检查.md` |

## Properties 模板

```yaml
---
title: "Slime {主题} 源码走读"
type: walkthrough
framework: slime
topic: "{主题}"
learning_role: reference
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/walkthrough
  - source-reading
updated: 2026-07-11
---
```

`learning_role` 只使用 `core`、`reference`、`debug`、`practice`。源码走读通常是 `reference`；首次学习主线中的必读页才使用 `core`。

## 正文模板

```markdown
# Slime {主题} 源码走读

## 读者任务

说明本文位于闭环哪一段，读完能解释或排查什么。

## 先建立模型

画出 Ray actor、Placement Group、Sample、ObjectRef、DP partition 或权重同步边界。

## 贯穿场景

选择一次 rollout、训练、reward 过滤、offload 或 update_weights。

## 主线走读

按真实调用和对象生命周期展开源码证据。

## 运行验证

- 操作：debug flag、日志、metric、测试或静态源码定位。
- 预期：说明样本数、资源状态、同步屏障或失败信号。

## 复盘

总结闭环位置、不变量、上下游和下一篇。
```

## 源码要求

- 修改解释前完整阅读本文引用的 Slime upstream 文件。
- 来源写在代码块首行，行号对应当前基线。
- 特别检查 rollout_id、样本分组、ObjectRef 生命周期和权重版本一致性。
- 新增摘录后运行 `node maintenance/audit_source_evidence.mjs`。

完整规范见 [[maintenance/源码阅读写作标准]] 与 [[maintenance/Obsidian知识库规范]]。
