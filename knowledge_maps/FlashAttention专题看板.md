---
title: "FlashAttention 专题看板"
type: dashboard
framework: flash-attn
topic: "FlashAttention"
learning_role: reference
tags:
  - framework/flash-attn
  - content/dashboard
  - source-reading
updated: 2026-07-13
---

# FlashAttention 专题看板

## 阅读路径

首次阅读只走 [[Attention算子主线]]，再按问题深入：

- IO 与精确 softmax：[[FlashAttention-Attention-IO]] · [[FlashAttention-Online-Softmax]]
- Python 与 binding：[[FlashAttention-Python-API]]
- FA2 forward/backward：[[FlashAttention-FA2-Forward]] · [[FlashAttention-Backward]]
- Decode KV：[[FlashAttention-KV-Cache]]
- Hopper/CuTe：[[FlashAttention-Hopper与CuTe]]

## 动态内容

![[FlashAttention内容.base]]

## 使用建议

性能问题先确认实际 API、shape、dtype 和 GPU 架构，再查看 dispatch 和 profiler；不要直接从任意 `.cu` specialization 开始。

看板只说明知识库入口，不证明当前环境已加载 FA2/FA3/FA4。动态验收必须记录实际 extension、GPU arch 和 kernel trace。
