---
title: "KV-Cache"
type: map
framework: flash-attn
topic: "KV-Cache"
learning_role: core
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/map
  - source-reading
updated: 2026-07-10
---
# KV-Cache

> 这一组笔记只解决一个问题：一次 serving decode step 里，FlashAttention 如何把“写入新 K/V、读取历史 cache、可选 RoPE、paged addressing、SplitKV”合成一次 backend 调用。

## 你为什么要读

如果你在看 SGLang、vLLM 或自己的推理 runtime，KV cache 很容易被误解成一个普通 tensor。FA05 要建立的模型是：上层 runtime 负责分配和调度 cache，FlashAttention 负责在一次 decode attention 中按给定地址读写它。

读完后你应该能处理三类问题：

- 第一次读：能区分 prefill full attention 和 decode cache attention，知道 `q`、`k_cache`、`v_cache`、`cache_seqlens`、`block_table` 分别代表什么。
- 正在排障：能定位 paged KV、`cache_batch_idx`、`cache_leftpad`、RoPE、`num_splits` 这些开关为什么会互斥、报错或影响输出。
- 准备改代码：能沿 `flash_attn_with_kvcache` 追到 C++ params 和 splitKV kernel，知道哪些不变量必须由上层 runtime 保证。

## 主线图

```mermaid
flowchart LR
    Runtime["serving runtime<br/>分配 cache 和长度"] --> API["flash_attn_with_kvcache<br/>归一化参数"]
    API --> CPP["mha_fwd_kvcache<br/>校验地址模式"]
    CPP --> Params["Flash_fwd_params<br/>指针/stride/长度"]
    Params --> Kernel["splitKV kernel<br/>可选 append + attention"]
    Kernel --> Cache["K/V cache<br/>dense 或 paged"]
    Kernel --> Out["out / softmax_lse"]
```

这条线的关键是边界：FlashAttention 不决定哪条请求占哪个 cache slot，也不为每条请求重新分配 block；它只相信调用者传入的长度、batch index 或 block table，然后在 kernel 中完成读写。

## 源码范围

- `flash_attn_with_kvcache` 把 decode cache 语义暴露给 Python 调用者，并说明 in-place append、容量责任、RoPE 位置和 `num_splits` 语义。

```python
# 来源：flash_attn/flash_attn_interface.py L1507-L1514
If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
the previous step, and update them with the new keys/values from the current step, and do
attention with the updated cache, all in 1 kernel.

If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
```

- `mha_fwd_kvcache` 是 C++ 边界：它决定 dense cache、batch remap、paged KV、leftpad、RoPE、SplitKV 是否能组合。

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L1247-L1268
at::Tensor block_table;
const bool paged_KV = block_table_.has_value();
if (paged_KV) {
    TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
    block_table = block_table_.value();
    CHECK_DEVICE(block_table);
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
}
```

- `Flash_fwd_params` 是跨 C++/CUDA 的参数包，里面同时保存新 K/V 指针、RoPE 指针、cache remap、block table 和 page size。

```cpp
// 来源：csrc/flash_attn/src/flash.h L83-L105
void * __restrict__ knew_ptr;
void * __restrict__ vnew_ptr;
void * __restrict__ rotary_cos_ptr;
void * __restrict__ rotary_sin_ptr;
int * __restrict__ cache_batch_idx;
int * __restrict__ block_table;
index_t block_table_batch_stride;
int page_block_size;
```

## 阅读顺序

1. [[FlashAttention-KV-Cache-核心概念]]：先建立 decode step 的对象模型，重点看 `cache_seqlens`、地址模式和容量责任。
2. [[FlashAttention-KV-Cache-源码走读]]：沿一次调用从 Python 进入 C++，再落到 splitKV kernel。
3. [[FlashAttention-KV-Cache-数据流]]：把 cache update、RoPE、paged addressing、SplitKV partial buffer 串成生命周期。
4. [[FlashAttention-KV-Cache-排障指南]]：按症状找源码入口，例如 paged KV 互斥、RoPE 报错、长上下文性能下降。
5. [[FlashAttention-KV-Cache-学习检查]]：用源码定位和测试命令验收自己是否真的读通。

## 和其他专题的关系

- 从 [[FlashAttention-Attention-IO]] 继承的是 IO-aware attention 的基本约束：不要物化完整 attention 矩阵，尽量让 Q/K/V tile 在合适层级流动。
- 从 [[FlashAttention-FA2-Forward]] 继承的是 forward kernel 的参数包和 launch 分发。
- FA05 新增的是 serving decode 语义：cache 是跨 step 保存的状态，backend 每次只处理当前 step 的读写。
