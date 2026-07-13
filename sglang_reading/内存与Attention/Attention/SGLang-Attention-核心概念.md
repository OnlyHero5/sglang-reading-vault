---
title: "Attention · 核心概念"
type: concept
framework: sglang
topic: "Attention"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-11
---
# Attention · 核心概念

## 你为什么要读

这篇先建立心理模型：Attention 后端不是某个单独 kernel 文件，而是连接 `ForwardBatch`、每层 Q/K/V、paged KV cache 和具体 kernel 的编译层。它先决定走哪条 backend，再把当前 batch 的 KV 索引、query 边界、Graph buffer、SWA/cross-attention 变体整理成 kernel metadata。

## 五层模型

| 层 | 对象 | 责任 |
|----|------|------|
| 解析层 | 原始 flags、device/model defaults、兼容性处理 | 得到可运行的 prefill/decode 后端名，不只是照抄用户输入 |
| 对象组合层 | registry factory、linear/hybrid/TBO/PDMux wrapper | 把一个名字变成真正执行时的对象图 |
| 调度语义层 | `ForwardMode`、`ForwardBatch` | 表示并承载本轮执行；batch 可能被 padding、分片或 inner view 改写 |
| 每层入口 | `RadixAttention.forward` | 把 Q/K/V reshape 后交给当前 attention backend |
| 后端实现层 | `AttentionBackend` 子类、metadata、KV pool | 计划索引，翻译地址，写 KV，调用选定 kernel |

读者抓手：遇到 attention 路径问题时，先问四个问题：配置解析成了什么 backend；当前 `ForwardMode` 是什么；`RadixAttention` 是否走 piecewise custom op；后端 metadata 是否和 kernel 调用一致。

## 配置不是“一个字段变两个字符串”

`get_attention_backends()` 确实返回两个名字，但它处在解析链末端。到这里之前，`ServerArgs.__post_init__` 已经可能根据设备、模型架构、确定性推理、page size、KV dtype、Unified/PageMajor、speculative decoding 等条件选择默认值、改写参数或拒绝启动。

下面这张源码卡只证明两个关键判断：显式把 prefill/decode 都设成同一个非空值时，会反向归一通用字段；通用字段为空时，会先自动选默认后端。

```python
# 来源：sglang/python/sglang/srt/server_args.py L4666-L4679
    def _handle_attention_backend_compatibility(self):
        model_config = self.get_model_config()
        use_mla_backend = self.use_mla_backend()

        if self.prefill_attention_backend is not None and (
            self.prefill_attention_backend == self.decode_attention_backend
        ):  # override the default attention backend
            self.attention_backend = self.prefill_attention_backend

        # Pick the default attention backend if not specified
        if self.attention_backend is None:
            self.attention_backend = self._get_default_attn_backend(
                use_mla_backend, model_config
            )
```

解析完成后，getter 才把未单独指定的一侧回退到已经归一过的通用字段：

```python
# 来源：sglang/python/sglang/srt/server_args.py L6922-L6933
    def get_attention_backends(self):
        prefill_attention_backend_str = (
            self.prefill_attention_backend
            if self.prefill_attention_backend
            else self.attention_backend
        )
        decode_attention_backend_str = (
            self.decode_attention_backend
            if self.decode_attention_backend
            else self.attention_backend
        )
        return prefill_attention_backend_str, decode_attention_backend_str
```

因此应记录四层证据：原始命令行、post-init 后字段、`get_attention_backends()` 返回值、`ModelRunner` 实际创建的对象。只看第一层会漏掉自动选择和兼容性改写。

当前后端也远不止 FlashInfer 与 Triton：候选集合覆盖 FA3/FA4、MLA、DSA/DSV4、TensorRT-LLM、AMD、Ascend、Intel 等实现。列表表示“可以被参数解析接受”，不等于在任意模型和硬件上都可运行；真正能力仍由 factory 和兼容性检查限定。

## `ForwardMode` 是运行时语义

`ForwardMode` 把同一层 attention 的计算形态拆开。当前枚举不止最常见的五种，还包括 draft、PD multiplexing、PD decode worker 与 dLLM 专用状态。

```python
# 来源：sglang/python/sglang/srt/model_executor/forward_batch_info.py L78-L102
class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers will be IDLE if no sequence are allocated.
    IDLE = auto()

    # Used in speculative decoding: verify a batch in the target model.
    TARGET_VERIFY = auto()
    # Used in speculative decoding: extend a batch in the draft model.
    DRAFT_EXTEND_V2 = auto()

    # Used in disaggregated decode worker
    # Represent a batch of requests having their KV cache ready to start decoding
    PREBUILT = auto()

    # Split Prefill for PD multiplexing
    SPLIT_PREFILL = auto()

    # Used in dLLM
    DLLM_EXTEND = auto()
```

```python
# 来源：sglang/python/sglang/srt/model_executor/forward_batch_info.py L107-L141
    def is_extend(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or (include_draft_extend_v2 and self == ForwardMode.DRAFT_EXTEND_V2)
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.SPLIT_PREFILL
            or self == ForwardMode.DLLM_EXTEND
        )

    def is_context_parallel_extend(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or (
                self == ForwardMode.DRAFT_EXTEND_V2
                if include_draft_extend_v2
                else False
            )
        )

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED

    def is_idle(self):
        return self == ForwardMode.IDLE

    def is_decode_or_idle(self):
        return self == ForwardMode.DECODE or self == ForwardMode.IDLE

    def is_target_verify(self):
        return self == ForwardMode.TARGET_VERIFY
```

需要特别记住三点：`is_extend()` 默认不含 `DRAFT_EXTEND_V2`；`PREBUILT` 表示 PD decode worker 已有 KV、准备进入 decode 的状态，不是普通 attention 基类随便可吃的 mode；`is_cuda_graph()` 还包含 `DLLM_EXTEND`。backend 选路应调用语义谓词，而不是拿枚举名做字符串猜测。

## `ForwardBatch` 是可变执行视图，不是不可变事实包

它从 `ScheduleBatch` 借入张量和引用，但在真正进入模型前后仍可能发生 DP padding、idle mode 转换、TBO child view、multi-step draft inner view 与 piecewise token 裁剪。源码还专门维护 `forward_metadata_ready`、计划时 shape 和 `replan_equivalent`：这说明 metadata 可能已被 graph runner 或 draft planner 提前准备，普通 forward 不能无条件再 plan 一遍。

```python
# 来源：sglang/python/sglang/srt/model_executor/forward_batch_info.py L547-L584
    def mark_forward_metadata_ready(self, replan_equivalent: bool = False):
        """Record that attention metadata was pre-planned for this batch.

        Call right next to the out-of-forward planning action
        (e.g. ``draft_attn_backend.init_forward_metadata(fb)`` or
        ``graph_runner.load_batch(fb)``). Records the batch shapes so
        staleness is detectable; pass ``replan_equivalent=True`` only when
        a forward-path re-plan is equivalent to the pre-plan (see field
        docs).
        """
        self.forward_metadata_ready = True
        self.forward_metadata_planned_bs = self.batch_size
        self.forward_metadata_planned_num_tokens = (
            self.input_ids.shape[0] if self.input_ids is not None else 0
        )
        self.forward_metadata_replan_equivalent = replan_equivalent

    def needs_forward_metadata_init(self) -> bool:
        """Single judgment point for whether the forward path must plan.

        A marked batch is treated as stale — and re-planned — when its
        shapes no longer match the plan record AND the mark site declared
        the re-plan safe (replan_equivalent). This runs after
        prepare_mlp_sync_batch in _forward_raw, so the re-plan sees the
        padded (final) shapes. Sites that cannot opt in (multi-step
        wrapper plans etc.) keep today's behavior: marked stays skipped,
        backends' defensive checks remain the backstop.
        """
        if not self.forward_metadata_ready:
            return True
        if not self.forward_metadata_replan_equivalent:
            return False
        num_tokens = self.input_ids.shape[0] if self.input_ids is not None else 0
        return (
            self.batch_size != self.forward_metadata_planned_bs
            or num_tokens != self.forward_metadata_planned_num_tokens
        )
```

这张卡证明的是“存在预计划与失效判断”，不是说所有 stale metadata 都能安全重建。multi-step wrapper 或特殊 view 若没有声明等价，重建反而会覆盖它们的专用计划。

## metadata 三阶段是为了 CUDA Graph

`AttentionBackend` 把 metadata 准备分成 eager、graph 外、graph 内。图外负责 capture/replay 前的 Python 计划与约定缓冲区刷新；图内只放 capture 时可录制的静态 shape GPU 操作。注意“在图外”不等于“可以随意换指针”：已经被 graph 捕获的 tensor 地址必须保持稳定，常见做法是对固定 buffer `copy_`，而不是重新赋一个 tensor。

```python
# 来源：sglang/python/sglang/srt/layers/attention/base_attn_backend.py L45-L51
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Eager entry point. Default = ``_out_graph(fb) + _in_graph(fb)``.

        Backends may override to keep an independent eager body.
        """
        self.init_forward_metadata_out_graph(forward_batch)
        self.init_forward_metadata_in_graph(forward_batch)
```

```python
# 来源：sglang/python/sglang/srt/layers/attention/base_attn_backend.py L75-L87
    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Graph-recordable static-shape GPU op.

        Runs inside ``with graph.capture():`` at capture time; recorded
        ops auto-execute at replay via ``graph.replay()``.

        Lint contract for overrides: body must NOT call ``.item()`` /
        ``.cpu()`` / ``.tolist()`` / dynamic-shape ``torch.empty()``.
        Such ops belong in :py:meth:`init_forward_metadata_out_graph`; they
        cannot be recorded into a cuda graph.

        Default: no-op.
        """
```

`init_forward_metadata()` 默认组合 out-graph 与 in-graph，但 FlashInfer 等后端可以覆写成独立 eager 实现；`init_forward_metadata_in_graph()` 只在 capture block 内由 Python 调一次，replay 时是录下来的 GPU op 自动执行，不会每轮再次调用 Python 方法。Graph 问题先核对生命周期和地址稳定性，再怀疑 kernel 数学。

## KV 地址有“通用语义”和“物理落点”两层

`forward_batch.out_cache_loc` 表示本轮新 token 的 generic write location。普通 pool 中它可直接对应物理位置；Unified pool 中它可能是 virtual id；SWA 还要翻译到独立 pool；cross-attention 则改用 `encoder_out_cache_loc`。`KVWriteLoc` 正是为同时携带 generic `loc`、SWA physical `swa_loc` 与 Unified full physical `full_loc` 而存在。

与之对应，`kv_indptr + kv_indices` 描述 kernel 本轮读取的 KV index stream。不要机械称为“历史 KV”：后端可能先写本轮 K/V，再让 wrapper/kernel 读取包含当前 token 的索引流，具体先后由该 backend 的 plan 与 forward 实现决定。

## `RadixAttention` 是每层进入后端的门

模型层先算 Q/K/V 和 RoPE，再调用 `RadixAttention.forward`。这里会 reshape K/V，并把当前层、`ForwardBatch`、`save_kv_cache` 交给全局 backend。

```python
# 来源：sglang/python/sglang/srt/layers/radix_attention.py L109-L153
    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)

        if (
            forward_batch.forward_mode.is_extend()
            and get_tc_piecewise_forward_context() is not None
        ):
            if self.qk_head_dim != self.v_head_dim:
                output = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
            else:
                output = torch.empty_like(q)
            if is_in_breakable_cuda_graph():
                breakable_unified_attention_with_output(
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
            else:
                unified_attention_with_output(
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
            return output
        else:
            return get_attn_backend().forward(
                q,
                k,
                v,
                self,
                forward_batch,
                save_kv_cache,
                **kwargs,
            )
```

这段源码给出两个入口：普通路径走 `get_attn_backend().forward`；piecewise CUDA Graph 的 extend 路径走 custom op，再在 op 内回到 backend。读 attention 问题时，先确认当前是否启用了 piecewise graph。

## 后端基类只做路由

`AttentionBackend.forward` 按 `forward_batch.forward_mode` 分发到 decode、mixed 或 extend。真正 kernel 在子类。

```python
# 来源：sglang/python/sglang/srt/layers/attention/base_attn_backend.py L170-L201
        if forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        elif forward_batch.forward_mode.is_mixed() and is_npu():
            return self.forward_mixed(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
```

不变量：如果某个子 backend 只实现了 decode 但没有正确处理 extend metadata，那么即使对象能创建，prefill 路径仍会失败。

## 运行验证

核心概念可以用一次检索串起来：配置层给出 backend 名称，`ForwardMode` 给出运行语义，`AttentionBackend` 定义契约，`RadixAttention` 是每层进入 backend 的门。

```powershell
rg -n 'def get_attention_backends|class ForwardMode|class CaptureHiddenMode|class AttentionBackend|def init_forward_metadata|def forward_decode|def forward_extend|class RadixAttention|def forward\(|get_attn_backend|unified_attention_with_output' sglang/python/sglang/srt/server_args.py sglang/python/sglang/srt/model_executor/forward_batch_info.py sglang/python/sglang/srt/layers/attention/base_attn_backend.py sglang/python/sglang/srt/layers/radix_attention.py
```

读输出时按四层模型对齐：`get_attention_backends` 是配置入口；`ForwardMode` 决定本轮是 extend、decode 还是 verify；`AttentionBackend` 说明 metadata 与 forward 契约；`RadixAttention.forward` 最终把每层的 Q/K/V 和 `ForwardBatch` 交给当前 backend。
