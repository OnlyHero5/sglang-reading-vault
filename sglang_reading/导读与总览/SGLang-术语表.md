---
title: "SGLang 术语表"
type: reference
framework: sglang
topic: "导读与总览"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/reference
  - source-reading
updated: 2026-07-10
---
# SGLang 术语表

> SGLang 核心术语 + **首次出现**的源码锚点

按字母/拼音序排列。路径相对于 `sglang/` 仓库根。

---

## 零基础速查（生活类比版）

> 完全没接触过 LLM serving？先看 [[SGLang-零基础先修]]。下表用一句话 + 生活类比快速建立直觉；细节点「深读」链到对应文档。

| 术语 | 一句话定义 | 生活类比 | 深读 |
|------|-----------|----------|------|
| **LLM Serving** | 把大模型部署成在线服务，多用户并发、流式返回答案 | 智能餐厅：接单、做菜、上菜一条龙 | [[SGLang-零基础先修]] §1、[[SGLang-项目总览]] |
| **Prefill** | 一次性处理整段 prompt，写入 KV Cache | 备菜：把订单上所有食材先切好腌好 | [[SGLang-零基础先修]] §2、[[SGLang-通用模型-数据流]] |
| **Decode** | 每步只生成 1 个新 token，循环直到结束 | 逐道上菜：炒好一道端一道，直到结账 | [[SGLang-零基础先修]] §2、[[SGLang-Scheduler]] |
| **TTFT** | 从请求进入到收到第一个 token 的耗时 | 点单后等第一道菜的时间 | [[SGLang-零基础先修]] §3、[[SGLang-生产排障]] §2 |
| **TPOT / ITL** | 首 token 之后，每两个 token 之间的平均间隔 | 每两道菜之间的等待时间 | [[SGLang-零基础先修]] §3、[[SGLang-可观测性]] |
| **Token** | 模型处理文本的最小单位（可能是字、词或子词） | 菜单上的「份」——整句被切成可计价的小份 | [[SGLang-TokenizerManager]] |
| **KV Cache** | 存已算过的 Key/Value 向量，避免 decode 时重复计算 | 服务员记事本：翻过的页不用重抄 | [[SGLang-零基础先修]] §4、[[SGLang-KV-Cache]] |
| **Continuous Batching** | 动态合并 prefill 与 decode 请求，提高 GPU 利用率 | 拼桌：有人吃完不撤桌，新客随时加入 | [[SGLang-零基础先修]] §5、[[SGLang-Scheduler]] |
| **TokenizerManager** | 主进程：HTTP 接入、tokenize、流式响应组装 | 餐厅前台：接单、记备注、收银 | [[SGLang-零基础先修]] §6、[[SGLang-TokenizerManager]] |
| **Scheduler** | 子进程：组 batch、驱动 GPU forward、管理队列 | 调度员 + 后厨入口：决定这轮做哪几桌 | [[SGLang-零基础先修]] §6、[[SGLang-Scheduler]] |
| **DetokenizerManager** | 子进程：token id 转 UTF-8 文本 | 传菜员：把厨房编号翻译成菜名 | [[SGLang-零基础先修]] §6、[[SGLang-Detokenizer]] |
| **ZMQ IPC** | 三进程间传递结构化消息的高性能通道 | 前台 ↔ 后厨的内部对讲机 | [[SGLang-HTTP-Server-数据流]]、[[SGLang-HTTP请求全链路|全链路请求追踪]] |
| **ForwardMode** | 区分 prefill（EXTEND）与 decode（DECODE）等 forward 类型 | 备菜模式 vs 单道上菜模式 | [[SGLang-零基础先修]] §2、[[SGLang-ScheduleBatch数据结构]] |
| **Prefix Cache** | 复用相同 prompt 前缀的 KV，跳过重复 prefill | 多家分店共用同一份今日特价菜单 | [[SGLang-零基础先修]] §7、[[SGLang-RadixAttention]] |
| **RadixAttention** | 用 radix tree 组织前缀 KV 的 SGLang 核心特性 | 按菜单目录树共享相同章节 | [[SGLang-零基础先修]] §7、[[SGLang-RadixAttention]] |
| **RadixCache** | 前缀缓存的具体实现类，含 match/insert/evict | 菜单档案柜：查、存、腾位置 | [[SGLang-RadixAttention]]、[[SGLang-KV-Cache]] |
| **Streaming / SSE** | 生成一个 token 就推一条，不等全文完成 | 逐道上菜，不用等满汉全席摆齐 | [[SGLang-HTTP-Server]]、[[SGLang-HTTP请求全链路|全链路请求追踪]] |
| **ModelRunner** | 加载权重、管理 KV pool、执行模型 forward | 后厨灶台：真正炒菜的地方 | [[SGLang-ModelRunner]]、[[SGLang-关键概念]] |
| **PD Disaggregation** | Prefill 与 Decode 拆到不同机器/池 | 备菜厨房与上菜厨房分开，各自扩缩 | [[SGLang-PD分离]]、[[SGLang-用户场景]] 故事 C |
| **Speculative Decoding** | 用小模型草稿 + 大模型验证，加速 decode | 学徒先猜下一道，主厨快速验对错 | [[SGLang-Speculative]]、[[SGLang-用户场景]] 故事 B |

---

## BatchStrOutput

**定义：** Detokenizer 发给 TokenizerManager 的批量字符串输出消息，含 `output_strs`、`finished_reasons`、token 计数等。

**读法：** 与 `BatchTokenIDOutput` 成对，是 ZMQ 回程消息类型。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/detokenizer_manager.py L415-L419
        return BatchStrOutput(
            rids=recv_obj.rids,
            http_worker_ipcs=recv_obj.http_worker_ipcs,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
```

---

## BatchTokenIDOutput

**定义：** Scheduler 发给 Detokenizer 的批量 token id 输出，每个 rid 对应本 step 新采样的 token。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/detokenizer_manager.py L406-L408
    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
        # If handling idle batch, set output_strs to [].
        output_strs = (
```

---

## Continuous Batching（连续批处理）

**定义：** Scheduler 每轮 loop 动态合并 prefill 与 decode 请求，提高 GPU 利用率。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L1533-L1539
            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
```

---

## CSGMV

**定义：** Compressed Sparse GEMV，LoRA 多 adapter batch 并行的 Triton kernel 名。

**源码锚点：**

```python
## 来源：python/sglang/srt/lora/lora_manager.py L98-L99
        # LoRA backend for running sgemm kernels
        logger.info(f"Using {lora_backend} as backend of LoRA kernels.")
```

---

## DetokenizerManager

**定义：** `sglang::detokenizer` 子进程的主类，专职 token id → UTF-8 string。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/detokenizer_manager.py L161-L164
    def event_loop(self):
        """The event loop that handles requests"""
        while True:
            with self.soft_watchdog.disable():
```

---

## DisaggregationMode

**定义：** PD 分离模式枚举：NULL / PREFILL / DECODE，决定 Scheduler 队列行为。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L3199-L3201
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            for req in batch.reqs:
                self.maybe_send_cached_prefix_chunk(req)
```

---

## Engine

**定义：** SRT 引擎 facade，负责 spawn 子进程、暴露 TokenizerManager。

**源码锚点：**

```python
## 来源：python/sglang/srt/entrypoints/http_server.py L2501-L2506
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager_func,
        run_scheduler_process_func=run_scheduler_process_func,
        run_detokenizer_process_func=run_detokenizer_process_func,
    )
```

---

## EntryClass

**定义：** 模型 registry 中每个 architecture 对应的实现类（如 `LlamaForCausalLM`）。

**源码锚点：**

```python
## 来源：python/sglang/srt/models/llama.py（文件末尾惯例）
EntryClass = LlamaForCausalLM
```

---

## ForwardBatch

**定义：** GPU 侧前向 batch 数据结构，由 `ScheduleBatch` 物化而来。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/tp_worker.py L495
            forward_batch = ForwardBatch.init_new(batch, self.model_runner)
```

---

## ForwardMode

**定义：** EXTEND/DECODE/MIXED/TARGET_VERIFY/PREBUILT 等前向语义模式。

**源码锚点：**

```python
## 来源：python/sglang/srt/model_executor/forward_batch_info.py L78-L81
class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
```

---

## GenerateReqInput

**定义：** HTTP `/generate` 请求体 dataclass，含 prompt、采样参数、stream 标志等。

**源码锚点：**

```python
## 来源：python/sglang/srt/entrypoints/http_server.py L790
async def generate_request(obj: GenerateReqInput, request: Request):
```

---

## GroupCoordinator

**定义：** ProcessGroup 包装类，路由 all_reduce/all_gather 到 PyNccl/CustomAR。

**源码锚点：**

```python
## 来源：python/sglang/srt/distributed/parallel_state.py（class GroupCoordinator）
# all_reduce(tensor) → 按 backend 选 collective 实现
```

---

## HiCache

**定义：** 分层 KV 缓存，device cache + host backup，支持更大有效上下文。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/tp_worker.py L492-L493
            # update the consumer index of hicache to the running batch
            self.set_hicache_consumer(batch.hicache_consumer_index)
```

---

## LoadFormat

**定义：** 权重加载格式枚举：AUTO、GGUF、REMOTE 等，决定 Loader 实现类。

**源码锚点：**

```python
## 来源：python/sglang/srt/model_loader/loader.py（LoadFormat 枚举）
```

---

## MatchPrefixParams / MatchResult

**定义：** RadixCache 前缀匹配入参/出参；Result 含 `device_indices` 与 terminal node。

**源码锚点：**

```python
## 来源：python/sglang/srt/mem_cache/radix_cache.py L355
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
```

---

## ModelRunner

**定义：** 单 GPU rank 的模型执行引擎：forward、sample、CUDA Graph。

**源码锚点：**

```python
## 来源：python/sglang/srt/model_executor/model_runner.py L2954
    def forward(
```

---

## PD Disaggregation（Prefill-Decode 分离）

**定义：** Prefill 与 Decode 集群独立部署，KV 跨节点 RDMA 传输。

**源码锚点：**

```python
## 来源：python/sglang/srt/disaggregation/prefill.py L104-L107
class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """
```

---

## RadixAttention

**定义：** SGLang Attention 层实现，对接多种 backend 读写 KV pool。

**源码锚点：**

```python
## 来源：python/sglang/srt/layers/radix_attention.py L57-L60
class RadixAttention(nn.Module):
    """
    The attention layer implementation.
    """
```

---

## RadixCache

**定义：** 基于 Radix Tree 的前缀 KV 索引，支持 match/insert/evict。

**源码锚点：**

```python
## 来源：python/sglang/srt/mem_cache/radix_cache.py L355-L359
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
```

---

## RadixKey

**定义：** Radix Tree 的 key 类型，含 token id 序列与 optional `extra_key` 命名空间。

**源码锚点：**

```python
## 来源：python/sglang/srt/mem_cache/radix_cache.py L359-L367
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.
```

---

## ScheduleBatch

**定义：** Scheduler 侧逻辑 batch，含多个 `Req` 及 forward_mode、sampling_info。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L3178
        batch: ScheduleBatch,
```

---

## ServerArgs

**定义：** 全部 server CLI 参数 dataclass，由 `prepare_server_args` 解析生成。

**源码锚点：**

```python
## 来源：python/sglang/cli/serve.py L126
            server_args = prepare_server_args(dispatch_argv)
```

---

## SpeculativeAlgorithm

**定义：** 投机解码算法枚举：EAGLE、NGRAM、DFLASH 等。

**源码锚点：**

```python
## 来源：python/sglang/srt/speculative/spec_info.py L28-L35
class SpeculativeAlgorithm(Enum):
    """Builtin speculative decoding algorithms. Plugin-registered ones are
    ``CustomSpecAlgo`` instances; ``from_string`` returns either type, and
    both expose the same ``is_*()`` / ``create_worker`` interface so callers
    dispatch uniformly without isinstance checks.
    """

    DFLASH = auto()
```

---

## SRT（SGLang Runtime）

**定义：** `python/sglang/srt/` 包，推理运行时核心。

**源码锚点：**

```python
## 来源：python/sglang/srt/entrypoints/http_server.py L2480
    Launch SRT (SGLang Runtime) Server.
```

---

## TokenizerManager

**定义：** 主进程请求枢纽：tokenize、IPC、流式响应组装。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/tokenizer_manager.py L589
    async def generate_request(
```

---

## TpModelWorker

**定义：** Tensor Parallel Worker，Scheduler 调用的 execution 门面。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/tp_worker.py L482
    def forward_batch_generation(
```

---

## ZMQ IPC

**定义：** Scheduler/Detokenizer 与 TokenizerManager 之间的 ZeroMQ 进程间通信。

**源码锚点：**

```python
## 来源：python/sglang/srt/entrypoints/http_server.py L2491-L2492
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
```

---

## 索引

| 类别 | 术语 |
|------|------|
| 数据结构 | GenerateReqInput, ScheduleBatch, ForwardBatch, BatchTokenIDOutput, BatchStrOutput |
| 进程/类 | TokenizerManager, Scheduler, DetokenizerManager, ModelRunner, TpModelWorker |
| 缓存 | RadixCache, RadixKey, RadixAttention, HiCache, MatchPrefixParams |
| 模式 | ForwardMode, DisaggregationMode, SpeculativeAlgorithm, LoadFormat |
| 特性 | Continuous Batching, PD Disaggregation, CSGMV, EntryClass |
