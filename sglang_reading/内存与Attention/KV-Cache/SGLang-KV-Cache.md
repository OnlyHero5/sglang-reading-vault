---
title: "KV-Cache"
type: map
framework: sglang
topic: "KV-Cache"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-10
---
# KV-Cache

## 读者为什么要读

如果你只知道“KV Cache 能减少重复计算”，排查 serving 问题时会很快卡住：`KV cache pool is full` 到底是物理显存不够，还是请求行不够，还是 page 边界导致的额外分配？prefix cache 命中后，哪些 token 真的不用写 KV？decode 每步为什么还可能触发 retract？

这一组文档回答一个更具体的问题：

> 一个请求里的 token，如何拿到 KV slot、把 K/V 写进 GPU 张量、被 prefix tree 复用、在空间紧张时释放或下沉到 HiCache？

读完后，你应该能把 `req_pool_idx`、`req_to_token`、`out_cache_loc`、`token_to_kv_pool_allocator`、`KVCache` 张量池和 RadixCache 的边界分清。

## 主线图

```mermaid
flowchart LR
    MR["ModelRunner<br/>建 req/token 两级池"] --> Builder["kv_cache_builder<br/>把池交给 RadixCache"]
    Builder --> Tree["tree_cache<br/>prefix match / evict"]
    Tree --> Extend["alloc_for_extend<br/>prefill 分配"]
    Extend --> Map["req_to_token<br/>请求行到 KV slot"]
    Extend --> Loc["out_cache_loc<br/>本轮写入位置"]
    Loc --> Attn["Attention backend<br/>set_kv_buffer"]
    Attn --> Pool["KVCache 张量池<br/>K/V 物理存储"]
    Pool --> Decode["alloc_for_decode<br/>decode 追加"]
    Decode --> Retract["check_decode_mem / retract"]
    Pool --> Host["HiCache Host Pool"]
    Host --> Storage["Storage Backend<br/>file / NIXL / Mooncake 等"]
```

这条线的关键不是“有一个 allocator”，而是两级索引：

- `ReqToTokenPool`：给每个请求一行，记录这个请求每个逻辑 token 对应哪个 KV slot。
- `TokenToKVPoolAllocator`：管理 KV slot 或 page 的空闲与释放。
- `KVCache`：真正持有每层 K/V 张量，attention backend 用 `out_cache_loc` 写入。

源码入口：

```python
# 来源：python/sglang/srt/mem_cache/memory_pool.py L242-L268
class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""
    ...
    self.req_to_token = torch.zeros(
        (self._alloc_size, max_context_len), dtype=torch.int32, device=device
    )
    self.free_slots = list(range(1, self._alloc_size))
```

```python
# 来源：python/sglang/srt/mem_cache/allocator/base.py L27-L110
class BaseTokenToKVPoolAllocator(abc.ABC):
    ...
    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size
```

## 首次阅读路径

| 文件 | 读完要能回答 |
| ------ | -------------- |
| [[SGLang-KV-Cache-核心概念]] | 为什么 KV Cache 是“请求行 + KV slot + 物理张量”三层模型？ |
| [[SGLang-KV-Cache-源码走读]] | 一次 prefill/decode 如何分配 `out_cache_loc` 并写入 K/V？ |
| [[SGLang-KV-Cache-数据流]] | `Req`、`ScheduleBatch`、allocator、attention backend 之间传什么对象？ |
| [[SGLang-KV-Cache-排障指南]] | `alloc None`、page 对齐、retract、HiCache 启动失败分别查哪里？ |
| [[SGLang-KV-Cache-学习检查]] | 能否独立画出 KV slot 生命周期并做静态/运行验收？ |

## 源码范围

主要阅读：

- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/srt/mem_cache/kv_cache_builder.py`
- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/allocator/base.py`
- `python/sglang/srt/mem_cache/allocator/token.py`
- `python/sglang/srt/mem_cache/allocator/paged.py`
- `python/sglang/srt/mem_cache/pool_host/base.py`
- `python/sglang/srt/mem_cache/storage/backend_factory.py`
- `python/sglang/srt/layers/attention/triton_backend.py`

## 相邻专题

- 上游调度压力来自 [[SGLang-Scheduler]]：Scheduler 决定何时 prefill/decode，以及 decode 前何时 retract。
- prefix 复用逻辑见 [[SGLang-RadixAttention]]：RadixCache 决定哪些 prefix token 可以复用。
- Attention backend 如何消费 KV slot 见 [[SGLang-Attention]]。
- GPU forward 入口见 [[SGLang-ModelRunner]]。
