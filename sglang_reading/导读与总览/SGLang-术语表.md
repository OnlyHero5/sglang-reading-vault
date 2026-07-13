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

> SGLang 核心术语、易混边界与代表性源码锚点

按字母/拼音序排列。路径相对于 `sglang/` 仓库根。

使用原则：术语表负责快速区分对象，不负责替代专题主线。每张“源码锚点”只证明紧邻定义中的一个关键边界；要判断当前运行时实际走了哪条分支，仍需查看配置、对象、日志和实验。

## 长文读法

不要从头背到尾。先用编辑器或 Obsidian Search 搜术语，再按任务跳读：

| 任务 | 先看 |
|------|------|
| 第一次建立直觉 | “零基础速查”与“五组易混概念” |
| 跟请求对象 | `GenerateReqInput`、`BatchTokenIDOutput`、`BatchStrOutput` |
| 看调度与执行 | `ScheduleBatch`、`ForwardBatch`、`ForwardMode`、`ModelRunner`、`TpModelWorker` |
| 看 prefix/KV | `RadixCache`、`RadixKey`、`MatchPrefixParams / MatchResult`、`HiCache` |
| 看启动与权重 | `Engine`、`ServerArgs`、`EntryClass`、`LoadFormat`、`GroupCoordinator` |
| 看生产特性 | `DisaggregationMode`、`SpeculativeAlgorithm`、`CSGMV`、ZMQ IPC |

如果只记住一句话：先区分“请求身份、调度状态、资源地址、一步执行视图和回程消息”，再查具体名词。

---

## 零基础速查（生活类比版）

> 完全没接触过 LLM serving？先看 [[SGLang-零基础先修]]。下表用一句话 + 生活类比快速建立直觉；细节点「深读」链到对应文档。

| 术语 | 一句话定义 | 生活类比 | 深读 |
|------|-----------|----------|------|
| **LLM Serving** | 把大模型部署成在线服务，多用户并发、流式返回答案 | 智能餐厅：接单、做菜、上菜一条龙 | [[SGLang-零基础先修]] §1、[[SGLang-项目总览]] |
| **Prefill / Extend** | 处理尚未计算 KV 的输入段；可因 chunking、session 或新增输入分多次完成 | 备菜：这轮只处理还没备好的那一段 | [[SGLang-零基础先修]] §2、[[SGLang-ScheduleBatch数据结构]] |
| **Decode** | baseline 通常每轮为活动序列产生一个 token；speculative 等路径可能一次接受多个 token | 逐轮上菜；用了草稿验证时一轮可能通过多道 | [[SGLang-零基础先修]] §2、[[SGLang-Scheduler]] |
| **TTFT** | 从请求进入到收到第一个 token 的耗时 | 点单后等第一道菜的时间 | [[SGLang-零基础先修]] §3、[[SGLang-生产排障]] §2 |
| **TPOT / ITL** | TPOT 是平均每输出 token 时间；ITL 描述相邻输出到达间隔，分布和长尾不能被平均值替代 | 平均上菜速度 vs 每两道菜实际间隔 | [[SGLang-零基础先修]] §3、[[SGLang-可观测性]] |
| **Token** | 模型处理文本的最小单位（可能是字、词或子词） | 菜单上的「份」——整句被切成可计价的小份 | [[SGLang-TokenizerManager]] |
| **KV Cache** | 存已算过的 Key/Value 向量，避免 decode 时重复计算 | 服务员记事本：翻过的页不用重抄 | [[SGLang-零基础先修]] §4、[[SGLang-KV-Cache]] |
| **Continuous Batching** | Scheduler 每轮重新选择和更新活动请求；prefill/decode 可交错调度，但不保证混在同一次 forward | 拼桌：每轮重排座位，不等整桌一起结束 | [[SGLang-零基础先修]] §5、[[SGLang-Scheduler]] |
| **TokenizerManager** | API 侧请求状态机：规范化、tokenize、IPC dispatch、等待并组装返回 | 前台调度台：接内部订单、记状态、通知取餐 | [[SGLang-零基础先修]] §6、[[SGLang-TokenizerManager]] |
| **Scheduler** | 子进程：组 batch、驱动 GPU forward、管理队列 | 调度员 + 后厨入口：决定这轮做哪几桌 | [[SGLang-零基础先修]] §6、[[SGLang-Scheduler]] |
| **DetokenizerManager** | 子进程：token id 与 decode window 转增量文本 | 传菜员：把厨房编号翻译成菜名 | [[SGLang-零基础先修]] §6、[[SGLang-Detokenizer]] |
| **ZMQ IPC** | HTTP worker/TokenizerManager 与 runtime 子进程之间的结构化消息通道之一 | 前台 ↔ 后厨的内部对讲机 | [[SGLang-HTTP-Server-数据流]]、[[SGLang-HTTP请求全链路|全链路请求追踪]] |
| **ForwardMode** | 区分 prefill（EXTEND）与 decode（DECODE）等 forward 类型 | 备菜模式 vs 单道上菜模式 | [[SGLang-零基础先修]] §2、[[SGLang-ScheduleBatch数据结构]] |
| **Prefix Cache** | 复用相同 prompt 前缀的 KV，跳过重复 prefill | 多家分店共用同一份今日特价菜单 | [[SGLang-零基础先修]] §7、[[SGLang-RadixAttention]] |
| **RadixAttention（机制名）** | SGLang 对 prefix reuse 与 KV 生命周期协同的系统设计名称；不要与 `layers/radix_attention.py` 的 attention layer 类混为一谈 | 菜单目录树负责复用，灶台仍负责真正做菜 | [[SGLang-零基础先修]] §7、[[SGLang-RadixAttention]] |
| **RadixCache** | 前缀缓存的具体实现类，含 match/insert/evict | 菜单档案柜：查、存、腾位置 | [[SGLang-RadixAttention]]、[[SGLang-KV-Cache]] |
| **Streaming / SSE** | 生成过程中增量推送 chunk；一个 chunk 可因 stream interval、coalescing 或 speculative 接受而含多个 token | 分批上菜，不等整桌完成；每盘不一定只有一道 | [[SGLang-HTTP-Server]]、[[SGLang-HTTP请求全链路|全链路请求追踪]] |
| **ModelRunner** | rank-local 模型执行核心：持有模型与执行资源，组织 graph/eager forward 和 sampling | 后厨灶台与本轮执行台 | [[SGLang-ModelRunner]]、[[SGLang-关键概念]] |
| **PD Disaggregation** | Prefill 与 Decode 由不同 worker/资源池承担，并通过选定 transport 传递 KV 与 metadata | 备菜厨房与出餐厨房分开，交接方式不只一种 | [[SGLang-PD分离]]、[[SGLang-用户场景]] 故事 C |
| **Speculative Decoding** | draft 机制先提出候选，target 侧验证并接受/拒绝；draft 不一定是“小模型”，也可能是 n-gram 或插件算法 | 助手先给候选，主厨批量验收 | [[SGLang-Speculative]]、[[SGLang-用户场景]] 故事 B |

### 最容易混淆的五组概念

| 不要混为一谈 | 正确边界 |
|--------------|----------|
| `GenerateReqInput` / `ReqState` / `Req` | 外部输入契约 / API worker 等待状态 / Scheduler 单请求权威状态 |
| `ScheduleBatch` / `ForwardBatch` | 可变调度 batch / 单个 forward step 的执行视图 |
| prefix hit / KV 地址已分配 / 数据已在 device 可读 | 逻辑匹配 / allocator 所有权 / transfer 与 event 已完成是三件事 |
| RadixAttention 机制 / `RadixAttention` layer / `RadixCache` | 系统设计名称 / attention wrapper / prefix tree 与节点生命周期实现 |
| 配置 backend / resolver 选中对象 / profiler 中实际 kernel | 用户意图 / 运行时实现对象 / 最终执行事实 |

---

## BatchStrOutput

**定义：** Detokenizer 发给 TokenizerManager 的批量字符串输出消息，含 `output_strs`、`finished_reasons`、token 计数等。

**读法：** 与 `BatchTokenIDOutput` 成对，是 ZMQ 回程消息类型。

**源码锚点：**

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py L415-L419
        return BatchStrOutput(
            rids=recv_obj.rids,
            http_worker_ipcs=recv_obj.http_worker_ipcs,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
```

---

## BatchTokenIDOutput

**定义：** Scheduler 输出的批量 token 级消息，携带 rid、增量 decode ids、offset、finish reason 和计数等。普通 tokenizer 路径由 Detokenizer 消费；`skip_tokenizer_init` 路径可直接回 TokenizerManager。speculative 路径的单个 rid 一轮可能对应多个 accepted token。

**源码锚点：**

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py L406-L408
    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
        # If handling idle batch, set output_strs to [].
        output_strs = (
```

---

## Continuous Batching（连续批处理）

**定义：** Scheduler 每轮重新接收、筛选和更新活动请求，在新 prefill、running decode、idle/sync 或特性分支之间选择下一批。它允许不同生命周期的请求交错推进，但不等于所有 prefill/decode 永远混进同一次 GPU forward。

**源码锚点：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L1533-L1539
            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
```

---

## CSGMV

**定义：** 当前 `csgmv` LoRA backend 是 **Chunked SGMV**：基于 segmented gather matrix-vector multiplication，把输入序列切成固定大小 chunk，减少 adapter 分布偏斜时的过多 kernel launch。它不是 “Compressed Sparse GEMV”。

**源码锚点：**

```python
# 来源：python/sglang/srt/lora/backend/chunked_backend.py L24-L34
class ChunkedSgmvLoRABackend(BaseLoRABackend):
    """
    Chunked LoRA backend using segmented matrix-vector multiplication.

    This backend is largely based on the SGMV (Segmented Gather Matrix-Vector multiplication) algorithm
    introduced in the Punica paper (https://arxiv.org/pdf/2310.18547). One main variation made here is to
    segment the input sequences into fixed-size chunks, which reduces excessive kernel launches especially
    when the LoRA distribution is skewed.
    """

    name = "csgmv"
```

---

## DetokenizerManager

**定义：** Detokenizer 子进程的主状态机：接收 token 级输出，按类型 dispatch，维护增量解码语义并把字符串级结果发回 TokenizerManager。它也处理 embedding 与控制消息，不应概括成“只做 UTF-8 转换”。

**源码锚点：**

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py L161-L168
    def event_loop(self):
        """The event loop that handles requests"""
        while True:
            with self.soft_watchdog.disable():
                recv_obj = sock_recv(self.recv_from_scheduler)
            output = self._request_dispatcher(recv_obj)
            if output is not None:
                sock_send(self.send_to_tokenizer, output)
```

---

## DisaggregationMode

**定义：** PD 角色枚举：`NULL` 表示统一 worker，`PREFILL` 与 `DECODE` 表示分离角色。它会影响队列、bootstrap、transfer 与结果处理，但枚举本身不指定网络传输实现。

**源码锚点：**

```python
# 来源：python/sglang/srt/disaggregation/utils.py L60-L71
class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

    @staticmethod
    def to_engine_type(mode: str) -> str:
        if mode == DisaggregationMode.PREFILL.value:
            return "prefill"
        elif mode == DisaggregationMode.DECODE.value:
            return "decode"
        return "unified"
```

---

## Engine

**定义：** SRT 引擎 facade，负责 spawn 子进程、暴露 TokenizerManager。

**源码锚点：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2501-L2506
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager_func,
        run_scheduler_process_func=run_scheduler_process_func,
        run_detokenizer_process_func=run_detokenizer_process_func,
    )
```

---

## EntryClass

**定义：** 模型模块向 registry 暴露的入口类或入口类列表。registry 按每个类的 `__name__` 注册 architecture；因此一个模块可以贡献多个 architecture，而不是“一文件固定一个类”。

**源码锚点：**

```python
# 来源：python/sglang/srt/models/llama.py L851-L856
EntryClass = [
    LlamaForCausalLM,
    Phi3ForCausalLM,
    InternLM3ForCausalLM,
    IQuestCoderForCausalLM,
]
```

---

## ForwardBatch

**定义：** 单个 forward step 的执行视图，由 `ScheduleBatch` 物化，包含 input、position、seq length、KV/attention/sampling metadata 等 CPU/GPU 状态。它不是长期请求所有者，也不是不可变快照。

**源码锚点：**

```python
# 来源：python/sglang/srt/managers/tp_worker.py L495
            forward_batch = ForwardBatch.init_new(batch, self.model_runner)
```

---

## ForwardMode

**定义：** EXTEND/DECODE/MIXED/TARGET_VERIFY/PREBUILT 等前向语义模式。

**源码锚点：**

```python
# 来源：python/sglang/srt/model_executor/forward_batch_info.py L78-L81
class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
```

---

## GenerateReqInput

**定义：** SRT generate 的内部输入 dataclass，可承载 text、input ids、embeddings、多模态输入、采样参数和单/批请求身份。HTTP `/generate` 会接收它，但 Engine 等入口也可构造同一对象，所以它不是“只属于 HTTP 的 JSON schema”。

**源码锚点：**

```python
# 来源：python/sglang/srt/managers/io_struct.py L151-L160
@dataclass
class GenerateReqInput:
    # Request ID(s). If omitted, generated during normalization. For batch
    # requests, a string is expanded to per-item IDs using it as a prefix.
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    # Stable identity shared by requests in the same session. Unlike
    # session_params, this does not alter or reconstruct the prompt.
    session_id: Optional[str] = field(default=None, kw_only=True)
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
```

---

## GroupCoordinator

**定义：** 一组进程的 ProcessGroup 协调器，记录 group/global/local rank 与 CPU/device group，并可按 tensor size、CUDA Graph mode 等条件把通信路由到特定实现；不只负责 all-reduce。

**源码锚点：**

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L216-L225
class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It can route the communication to
        a specific implementation (e.g. switch allreduce implementation
        based on the tensor size and cuda graph mode).
    """
```

---

## HiCache

**定义：** 分层 KV 缓存体系：区分 device 命中、host 命中/load-back，并可与更深 storage 层协同。命中 host 不等于数据已经在 GPU 可读，仍需追 transfer 状态和 consumer。

**源码锚点：**

```python
# 来源：python/sglang/srt/managers/tp_worker.py L492-L493
            # update the consumer index of hicache to the running batch
            self.set_hicache_consumer(batch.hicache_consumer_index)
```

---

## LoadFormat

**定义：** 权重来源/格式枚举，包含 `AUTO`、PyTorch、Safetensors、Sharded State、GGUF、BitsAndBytes、remote/RDMA 等。它参与 loader 路由，但最终 loader 还受配置与模型条件影响。

**源码锚点：**

```python
# 来源：python/sglang/srt/configs/load_config.py L17-L36
class LoadFormat(str, enum.Enum):
    AUTO = "auto"
    PT = "pt"
    SAFETENSORS = "safetensors"
    NPCACHE = "npcache"
    DUMMY = "dummy"
    SHARDED_STATE = "sharded_state"
    GGUF = "gguf"
    BITSANDBYTES = "bitsandbytes"
    MISTRAL = "mistral"
    LAYERED = "layered"
    FLASH_RL = "flash_rl"  # For RL training with quantized models
    JAX = "jax"
    REMOTE = "remote"
    REMOTE_INSTANCE = "remote_instance"
    RDMA = "rdma"
    LOCAL_CACHED = "local_cached"
    FASTSAFETENSORS = "fastsafetensors"
    PRIVATE = "private"
    RUNAI_STREAMER = "runai_streamer"
```

---

## MatchPrefixParams / MatchResult

**定义：** 各类 prefix cache 共用的匹配入参与结果契约。`MatchResult` 不只含 device indices 和末端节点，还区分 host 命中、跨 cache component 接受的 best node、SWA/Mamba 命中等状态。

**源码锚点：**

```python
# 来源：python/sglang/srt/mem_cache/base_prefix_cache.py L155-L170
class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices  :   Indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
                            Reserved for L3 storage prefetch anchoring; L2 load_back
                            uses `best_match_node` instead.
        best_match_node :   Deepest node accepted by all component validators
                            during match_prefix. Anchor for every L2 host->device
                            load_back walk (FULL / SWA / ...). For legacy caches
                            that don't run multi-component validation, set this
                            equal to `last_host_node`.
```

---

## ModelRunner

**定义：** rank-local 模型执行核心：持有模型与执行资源，准备 eager/CUDA Graph/prefill graph 路径，执行 forward、sampling 与相关结果捕获。权重来源选择属于 ModelLoader，调度准入属于 Scheduler。

**源码锚点：**

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L2954
    def forward(
```

---

## PD Disaggregation（Prefill-Decode 分离）

**定义：** Prefill 与 Decode worker/资源池分离部署，增加 bootstrap、半程请求状态、KV/metadata transfer 和失败重试。传输可由 RDMA 或其他已配置 backend 承担，不能把 PD 与某一种 transport 画等号。

**源码锚点：**

```python
# 来源：python/sglang/srt/disaggregation/prefill.py L104-L107
class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """
```

---

## RadixAttention layer 与 RadixAttention 机制名

**定义：** `layers/radix_attention.py` 中的 `RadixAttention` 是模型层里的 attention wrapper，持有 head/layout/scaling 等配置并对接 backend；文档和项目宣传中的 “RadixAttention” 还常指 prefix reuse + radix cache 的系统机制。二者相关但不是同一个对象。

**源码锚点：**

```python
# 来源：python/sglang/srt/layers/radix_attention.py L57-L60
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
# 来源：python/sglang/srt/mem_cache/radix_cache.py L355-L359
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
# 来源：python/sglang/srt/mem_cache/radix_cache.py L359-L367
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

**定义：** Scheduler 侧可变 batch 状态，保存请求列表、共享 pool/cache、forward mode、sampling 等本轮与跨轮信息。overlap 下结果队列会保存 `batch.copy()`，不能把 live `ScheduleBatch` 一概当成不可变快照。

**源码锚点：**

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L1673-L1688
@dataclasses.dataclass
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    """Store all information of a batch on the scheduler."""

    # === Core: request list (ForwardBatch derives lora_ids / rids / grammars / positions from it) ===
    reqs: List[Req]

    # === Global config and shared resources (engine-lifetime; identical across batches) ===
    # Memory pool and cache
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None
    tree_cache: BasePrefixCache = None

    # Batch configs
    model_config: ModelConfig = None
    enable_overlap: bool = False
```

---

## ServerArgs

**定义：** server-wide 配置 dataclass。大量字段通过 `Annotated` 元数据派生 CLI flag，但对象还承载校验、默认值改写和内部配置，因此不能把它只当 argparse namespace。

**源码锚点：**

```python
# 来源：python/sglang/srt/server_args.py L374-L389
@dataclasses.dataclass
class ServerArgs:
    """Server-wide configuration for SGLang.

    Adding new arguments
    --------------------
    1. **Place the field in the right section.** Arguments are grouped by
       comment blocks (``# Model and tokenizer``, ``# LoRA``, etc.).
       Add new fields to the matching section, or create a new section
       with a ``# ---`` banner when none fits.

    2. **Use the ``A[T, ...]`` annotation.**  ``A`` is an alias for
       ``typing.Annotated``.  The primary CLI flag is auto-derived from the
       field name (``tp_size`` → ``--tp-size``).  Use ``aliases`` for
       longer alternate names
       (``aliases=["--tensor-parallel-size"]``)::
```

---

## SpeculativeAlgorithm

**定义：** 内建与插件投机算法的统一 dispatch 契约。当前内建项包括 DFLASH、EAGLE/EAGLE3、FROZEN_KV_MTP、STANDALONE、NGRAM 与 NONE；未知名称还会查询插件 registry。

**源码锚点：**

```python
# 来源：python/sglang/srt/speculative/spec_info.py L28-L57
class SpeculativeAlgorithm(Enum):
    """Builtin speculative decoding algorithms. Plugin-registered ones are
    ``CustomSpecAlgo`` instances; ``from_string`` returns either type, and
    both expose the same ``is_*()`` / ``create_worker`` interface so callers
    dispatch uniformly without isinstance checks.
    """

    DFLASH = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    FROZEN_KV_MTP = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()

    @classmethod
    def from_string(
        cls, name: Optional[str]
    ) -> Union[SpeculativeAlgorithm, CustomSpecAlgo]:
        if name is None:
            return cls.NONE
        upper = name.upper()
        try:
            return cls[upper]
        except KeyError:
            pass
        spec = _get_registered_spec(upper)
        if spec is not None:
            return spec
        raise ValueError(f"Unknown speculative algorithm name: {name}")
```

---

## SRT（SGLang Runtime）

**定义：** `python/sglang/srt/` 包，推理运行时核心。

**源码锚点：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2480
    Launch SRT (SGLang Runtime) Server.
```

---

## TokenizerManager

**定义：** API worker 侧请求状态机：规范化和校验输入、创建 `ReqState`、tokenize/预处理、向 Scheduler dispatch，并把返回消息组装成调用方可消费的结果。HTTP route 是它的上游之一，而不是它本身。

**源码锚点：**

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L589
    async def generate_request(
```

---

## TpModelWorker

**定义：** Scheduler 调用的 rank-local execution facade，负责把 `ScheduleBatch` 物化为 `ForwardBatch`，调用 ModelRunner 并包装 generation/embedding result。类名中的 TP 不表示它只处理一次 collective。

**源码锚点：**

```python
# 来源：python/sglang/srt/managers/tp_worker.py L482
    def forward_batch_generation(
```

---

## ZMQ IPC

**定义：** runtime 进程间使用的 ZeroMQ IPC 通道。普通基线包括 TokenizerManager→Scheduler、Scheduler→Detokenizer、Detokenizer→TokenizerManager；多 HTTP worker、DP/PP 或控制 RPC 会增加拓扑，不能固定理解成三条永恒不变的 socket。

**源码锚点：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2491-L2492
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
```

---

## 索引

| 类别 | 术语 |
|------|------|
| 数据结构 | GenerateReqInput, ScheduleBatch, ForwardBatch, BatchTokenIDOutput, BatchStrOutput |
| 进程/类 | TokenizerManager, Scheduler, DetokenizerManager, ModelRunner, TpModelWorker |
| 缓存 | RadixCache, RadixKey, RadixAttention 机制/Layer, HiCache, MatchPrefixParams |
| 模式 | ForwardMode, DisaggregationMode, SpeculativeAlgorithm, LoadFormat |
| 特性 | Continuous Batching, PD Disaggregation, CSGMV, EntryClass |

## 静态验证

操作：在仓库根目录运行以下检查，确认最容易漂移的七个术语仍有明确源码定义：

```powershell
$checks = @(
  @{ Path = 'sglang/python/sglang/srt/lora/backend/chunked_backend.py'; Pattern = 'class ChunkedSgmvLoRABackend' },
  @{ Path = 'sglang/python/sglang/srt/disaggregation/utils.py'; Pattern = 'class DisaggregationMode' },
  @{ Path = 'sglang/python/sglang/srt/model_executor/forward_batch_info.py'; Pattern = 'class ForwardMode' },
  @{ Path = 'sglang/python/sglang/srt/configs/load_config.py'; Pattern = 'class LoadFormat' },
  @{ Path = 'sglang/python/sglang/srt/managers/schedule_batch.py'; Pattern = 'class ScheduleBatch(' },
  @{ Path = 'sglang/python/sglang/srt/speculative/spec_info.py'; Pattern = 'class SpeculativeAlgorithm' },
  @{ Path = 'sglang/python/sglang/srt/models/llama.py'; Pattern = 'EntryClass = [' }
)

foreach ($check in $checks) {
  rg -n --fixed-strings $check.Pattern $check.Path
  if ($LASTEXITCODE -ne 0) { throw "missing term anchor: $($check.Pattern)" }
}
```

预期：七组定义全部命中。若名称仍在但语义字段已经变化，应回到对应专题重新解释，而不是只更新行号。
