---
title: "内存与Attention"
type: map
framework: sglang
topic: "内存与Attention"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-10
---
# 内存与Attention

> **你只需阅读本目录，不必打开 `sglang/` 源码。** 
> 内嵌代码对应 sglang Git commit `70df09b`。

---

## 本目录解决什么问题

模型执行部分跑通了 model forward。本目录回答：**KV Cache 如何分配与复用？Attention 内核如何选路？MoE 与量化如何压显存、提吞吐？** 这是 SGLang 相对 vLLM 等框架的核心差异化所在。

| 模块 | 模块 | 一句话 |
|------|------|--------|
| [[SGLang-RadixAttention]] | 前缀缓存 | Radix Tree 共享 prompt 前缀 KV，LPM 调度协同 |
| [[SGLang-KV-Cache]] | 物理分配 | Token/Page allocator、HiCache、Storage Backend |
| [[SGLang-Attention]] | 算子后端 | FlashInfer / Triton / MLA，extend vs decode kernel |
| [[SGLang-MoE]] | 专家并行 | Router、TopK、DeepEP dispatch、EPLB |
| [[SGLang-Quantization]] | 量化 | FP8/GPTQ/AWQ method、Linear/MoE 量化 apply |

---

## 内存—Attention 分层

```mermaid
flowchart TB
 SCH["Scheduler<br/>PrefillAdder 预算"]
 RC["RadixCache<br/>prefix match / evict"]
 ALLOC["KV Allocator<br/>token / page slot"]
 POOL["KVCache 张量池"]
 ATTN["Attention Backend<br/>paged KV 读写"]
 MOE["MoE Layer"]
 QUANT["Quant Method"]
 SCH --> RC
 RC --> ALLOC
 ALLOC --> POOL
 POOL --> ATTN
 ATTN --> MOE
 MOE --> QUANT
```

这张图的读法是：逻辑层（Radix Tree）决定「哪些 token 可复用前缀」；物理层（Allocator + KVCache）发 slot；Attention backend 在 forward 时按 `req_to_token` 索引读写 K/V。MoE 与量化在 layer 内叠加，不改变 Scheduler 调度接口。

**源码锚点：**

```python
## 来源：python/sglang/srt/mem_cache/radix_cache.py L312-L328
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

```

读法：

- `match_prefix` 返回可复用的 KV slot indices，Scheduler 在 prefill 时跳过已缓存 前缀。
- page 对齐时截断 token 长度，与 KV Cache page allocator 一致。
- evict 叶节点时通过 `allocator.free` 归还 slot（见 RadixAttention / KV Cache 专题）。

---

## 零基础一句话

**像图书馆：** RadixCache 是书目索引，KV Cache allocator 是书架格子，Attention backend 是阅览规则，MoE 与量化是特藏分区和压缩版藏书。

---

## 推荐阅读顺序

| 顺序 | 文档 | 必读理由 |
|------|------|----------|
| 1 | [[SGLang-RadixAttention-核心概念]] | RadixKey、match/insert/evict |
| 2 | [[SGLang-KV-Cache-源码走读]] | alloc_extend / alloc_decode |
| 3 | [[SGLang-Attention-数据流]] | ForwardBatch → backend 时序 |
| 4 | [[SGLang-MoE-核心概念]] | 五阶段 MoE 流水线 |
| 5 | [[SGLang-Quantization-排障指南]] | backend 选型与 Marlin |

---

## 阶段衔接

| 方向 | 模块 | 衔接点 |
|------|------|--------|
| ← 模型执行 | ModelRunner 与模型层 | Model 层调用 RadixAttention；ModelRunner 持有 attn_backend |
| → 高级特性 | Sampling、Speculative、PD 分离 | Sampling 读取 logits；投机解码复用 KV；PD 分离传输 KV |
| → 算子扩展 | sgl-kernel | Attention、MoE、量化热点算子下沉到 CUDA custom op |

---

## 自测建议（零基础可试）

1. **前缀命中：** 同一 system prompt 发两次，第二次 TTFT 应显著降低（`--schedule-policy lpm`）。
2. **OOM：** 调低 `--mem-fraction-static`，观察 retract 与 `KV cache pool is full` 日志。
3. **量化：** `--quantization fp8` 启动，确认 Linear 走 `dispatch_w8a8_block_fp8_linear`（见 [[SGLang-Quantization-源码走读]]）。

---

## 模块导航

| 专题 | 入口 |
|------|------|
| RadixAttention | [[SGLang-RadixAttention]] |
| KV Cache | [[SGLang-KV-Cache]] |
| Attention | [[SGLang-Attention]] |
| MoE | [[SGLang-MoE]] |
| Quantization | [[SGLang-Quantization]] |

← [[SGLang-模型执行]] · → [[SGLang-高级特性]]
