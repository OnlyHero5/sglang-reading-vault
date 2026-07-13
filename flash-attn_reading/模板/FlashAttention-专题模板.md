---
title: "FlashAttention 专题模板"
type: template
framework: flash-attn
topic: "模板"
learning_role: reference
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/template
  - source-reading
updated: 2026-07-11
---

# FlashAttention 专题模板

## 适用方式

每篇文档围绕一个 Q/K/V tensor、tile、online softmax 状态或 KV cache 请求，说明它如何穿过 Python、C++、CUDA/CuTe 和硬件内存层级。只创建服务实际学习、排障或修改任务的文档。

| 读者任务 | 推荐文件 |
|----------|----------|
| 建立主题地图 | `FlashAttention-{主题}.md` |
| 理解算法和硬件模型 | `FlashAttention-{主题}-核心概念.md` |
| 沿 dispatch 与 kernel 读源码 | `FlashAttention-{主题}-源码走读.md` |
| 看 tensor、指针和状态流 | `FlashAttention-{主题}-数据流.md` |
| 从精度、性能或平台症状排查 | `FlashAttention-{主题}-排障指南.md` |
| 验证掌握程度 | `FlashAttention-{主题}-学习检查.md` |

## Properties 模板

```yaml
---
title: "FlashAttention {主题} 源码走读"
type: walkthrough
framework: flash-attn
topic: "{主题}"
learning_role: reference
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/walkthrough
  - source-reading
updated: 2026-07-11
---
```

`learning_role` 只使用 `core`、`reference`、`debug`、`practice`。源码走读通常是 `reference`；只有首次学习主线中的必读页才标为 `core`。

## 正文模板

```markdown
# FlashAttention {主题} 源码走读

## 读者任务

说明读完能判断哪些 dtype、shape、stride、mask、硬件或 dispatch 问题。

## 先建立模型

画出 HBM、共享内存、寄存器、tile 和 online softmax 状态。

## 贯穿场景

选择普通 forward、varlen、backward、KV cache decode、Hopper 或 CuTeDSL JIT。

## 主线走读

沿 Python API、C++ 参数校验、模板分派和 kernel 执行展开证据。

## 运行验证

- 操作：正确性测试、benchmark、`ncu`、`nsys` 或静态 dispatch 检查。
- 预期：说明输出误差、选路、带宽、占用率或报错信号。

## 复盘

总结 IO 账本、关键不变量、适用硬件和下一篇。
```

## 源码要求

- 修改解释前完整阅读本文引用的 FlashAttention upstream 文件。
- Python 使用 `# 来源：...`，C++/CUDA 使用 `// 来源：...`。
- 不把生成 kernel 文件列表当成学习目标；必须解释 dispatch 维度和硬件约束。
- 新增摘录后运行 `node maintenance/audit_source_evidence.mjs`。

完整规范见 [[maintenance/源码阅读写作标准]] 与 [[maintenance/Obsidian知识库规范]]。
