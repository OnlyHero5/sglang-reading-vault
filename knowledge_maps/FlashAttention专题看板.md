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
updated: 2026-07-10
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

