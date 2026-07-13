---
title: "Attention · 数据流"
type: dataflow
framework: sglang
topic: "Attention"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/dataflow
  - source-reading
updated: 2026-07-11
---
# Attention · 数据流

## 你为什么要读

这篇不按文件顺序，而沿五种对象的生命周期追踪：配置值、后端对象、可变 batch 视图、kernel metadata、KV 地址。它们分别回答“选谁、谁持有、这轮做什么、kernel 怎么读、显存写到哪”，混淆其中任意两层都会制造很像 kernel bug 的上游错误。

## 对象生命周期总览

```mermaid
flowchart TD
    A["raw backend flags"] --> B["post-init resolver<br/>default / rewrite / reject"]
    B --> C["resolved per-mode names"]
    C --> D["registry factory + wrappers"]
    SB["ScheduleBatch"] --> FB["ForwardBatch<br/>可变执行视图"]
    FB --> PLAN["plan 或复用预计划 metadata"]
    D --> PLAN
    L["RadixAttention layer<br/>heads / dims / layer_id"] --> RUN["backend forward"]
    PLAN --> RUN
    W["generic write loc"] --> T["full / SWA / Unified 地址翻译"] --> RUN
    R["kernel KV index stream"] --> RUN
    RUN --> O["attention output"]
```

## 配置流：原始意图先经过约束求解

不能把配置流画成 `attention_backend → 两个字符串`。原始字段先经过设备默认、模型特例和 `_handle_attention_backend_compatibility()`；例如某些 backend 会强制 page size，某些组合会禁用 Graph，某些只支持 MLA、特定 GPU 或 decode。getter 返回的是这条解析链末端的两个 resolved 名称。

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

交互边界：`ServerArgs` 负责把“不完整的偏好”收敛成“可继续实例化的配置”，但不创建对象；`ModelRunner` 再把名字交给 registry factory。配置解析成功也不证明所有运行 mode 都已覆盖，后端自己的构造断言和运行路径仍可能继续收窄能力。

## 对象流：一个名字会长成一棵 wrapper 树

真实对象图可能是：

```text
resolved per-mode name
  → ATTENTION_BACKENDS[name](ModelRunner)
  → attn_backend_wrapper（可选 HybridLinearAttnBackend）
  → 两个 per-mode 子对象不同则组成 HybridAttnBackend
  → 最外层可再组成 TboAttnBackend，或由 PDMux 创建主对象和 decode workspace group
```

`HybridLinearAttnBackend` 与 `HybridAttnBackend` 名字相近但维度不同：前者按 layer id 在 full attention 与 Mamba/linear attention 间分流；后者按 `ForwardMode` 在 prefill/decode backend 间分流。排障时必须先问“当前是哪一层 wrapper 的所有权问题”。

## 模式流：`ForwardMode` 是后端选路的运行事实

| mode | 数据形态 | 常见后端入口 |
|------|----------|--------------|
| `EXTEND` | 多个新 token，可能包含 cached prefix | `forward_extend` |
| `DECODE` | 普通 decode 通常每请求一个新 token，消费本轮计划的 KV stream | `forward_decode` |
| `MIXED` | chunked prefill 中 extend/decode 混合 | 通常按 extend 处理 |
| `IDLE` | DP attention 中本 rank 无有效请求 | decode backend 空输出 |
| `TARGET_VERIFY` | speculative target verify | Hybrid 中受 `speculative_attention_mode` 控制 |
| `DRAFT_EXTEND_V2` | draft model 的固定形状扩展 | 仅部分谓词显式纳入；linear metadata 可跳过 |
| `PREBUILT` | PD decode worker 已具备 KV、准备 decode | 过渡状态，不应机械送进普通 attention `forward` |
| `SPLIT_PREFILL` | PD multiplexing 的拆分 prefill | `is_extend()` 为真 |
| `DLLM_EXTEND` | dLLM 扩展 | 同时属于 extend 与 CUDA Graph mode |

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

不变量：`ForwardMode` 是当前执行视图的语义标签，但它本身也可能因 DP padding/idle 适配被临时改写，之后再恢复 `_original_forward_mode`。不要用“用户请求正在生成”这种业务描述替代运行时 mode，也不要假定一个请求生命周期只对应一种 mode。

## metadata 流：先判断“要不要 plan”，再比较表达形式

multi-step draft、Graph runner 或专用 planner 可能已经为当前 `ForwardBatch` 准备好 metadata，并用 `forward_metadata_ready` 记录计划时的 batch size 与 token 数。普通 forward 只有在 `needs_forward_metadata_init()` 判定需要时才可重新计划；若 shape 因 DP padding 改变，也只有 `replan_equivalent=True` 的路径能安全重建。

在“确实需要 plan”之后，FlashInfer 与 Triton 才体现出不同表达形式。

FlashInfer 后端把 batch layout 交给 wrapper updater。decode 的结果是 `DecodeMetadata`，里面保存 decode wrapper 和 SWA 写入位置。

```python
# 来源：sglang/python/sglang/srt/layers/attention/flashinfer_backend.py L739-L760
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        swa_out_cache_loc = None
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            assert self._swa_kv_pool is not None
            swa_out_cache_loc = self._swa_kv_pool.translate_loc_from_full_to_swa(
                forward_batch.out_cache_loc
            )

        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
                fixed_split_size=self.decode_split_tile_size,
                disable_split_kv=False,
            )
            self.forward_metadata = DecodeMetadata(
                self.decode_wrappers, swa_out_cache_loc=swa_out_cache_loc
```

Triton 后端把同类信息摊平成 `ForwardMetadata` 字段。

```python
# 来源：sglang/python/sglang/srt/layers/attention/triton_backend.py L81-L103
@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    # Sliding window
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    window_kv_offsets: torch.Tensor
    # Separate attn_logits for SWA layers when v_head_dim differs
    swa_attn_logits: Optional[torch.Tensor] = None
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None
    # PHYSICAL full-attn write target for the unified pool (eager: translated tensor;
    # cuda-graph: capture-stable buffer view). None for non-unified pools.
    out_cache_loc_full_physical: Optional[torch.Tensor] = None
```

同一批调度事实的两种表达：FlashInfer 把 paged/ragged 计划写进 wrapper，并按 decode、verify、draft extend、DLLM、SWA/cross-attention 选择不同 wrapper 集；Triton 更直接地持有 kernel 参数包。FlashInfer 的 eager `init_forward_metadata()` 还有独立实现，不能因为基类提供默认组合，就假定它必然先后调用 out-graph 与 in-graph。

## KV 地址流：generic 写入位置到物理 pool

decode 不是只读 KV。本步 token 的 K/V 也要写入 pool，才能参与当前或后续读取。但 `out_cache_loc` 的准确名称是 generic write location：Unified 下可为 virtual，SWA 有单独物理落点，full attention 也可能需要显式的 Unified physical 地址。

```python
# 来源：sglang/python/sglang/srt/layers/attention/triton_backend.py L1647-L1679
        if save_kv_cache:
            if self.use_mla:
                if layer.k_scale is not None:
                    # MLATokenToKVPool doesn't accept scale parameters; k is unused
                    # after this point in decode, so scale in place.
                    k.div_(layer.k_scale)
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                )
            else:
                self._set_kv_buffer(
                    forward_batch,
                    layer,
                    KVWriteLoc(
                        forward_batch.out_cache_loc,
                        self.forward_metadata.swa_out_cache_loc,
                        full_loc=self.forward_metadata.out_cache_loc_full_physical,
                    ),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
```

读者抓手：`KVWriteLoc.loc` 保留通用位置，`swa_loc` 指向 SWA physical pool，`full_loc` 指向 Unified full-attention physical pool。`kv_indices` 则是 kernel 本轮的读取索引流，不应武断地等同于“纯历史 KV”；它是否已经覆盖当前 token，取决于该 backend 的写入顺序与 wrapper plan。cross-attention 还要把写入口切到 `encoder_out_cache_loc`。

## piecewise CUDA Graph 流：先裁剪真实 token，再回 backend

piecewise graph 的 extend 路径会通过 custom op 进入 `unified_attention_with_output`。它裁掉 padded token，临时收窄 `out_cache_loc`，再调用 backend。

```python
# 来源：sglang/python/sglang/srt/layers/radix_attention.py L176-L226
    context = get_tc_piecewise_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]
    real_num_tokens = forward_batch.num_token_non_padded_cpu

    query = query[:real_num_tokens]
    if key is not None:
        key = key[:real_num_tokens]
    if value is not None:
        value = value[:real_num_tokens]

    # DeepSeek MLA has two RadixAttention instances per layer (attn_mqa and
    # attn_mha) that share the same layer_id. The attention_layers list only
    # stores attn_mqa. When the MHA path is active (save_kv_cache=False), use
    # the companion attn_mha so the backend sees correct head/dim metadata.
    if _is_hip and not save_kv_cache and hasattr(attention_layer, "_pcg_mha_companion"):
        attention_layer = attention_layer._pcg_mha_companion

    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope[:real_num_tokens]
    if k_rope is not None:
        kwargs["k_rope"] = k_rope[:real_num_tokens]
    if sinks is not None:
        kwargs["sinks"] = sinks
    if cos_sin_cache is not None:
        kwargs["cos_sin_cache"] = cos_sin_cache
    if is_neox is not None:
        kwargs["is_neox"] = is_neox
    if llama_4_scaling is not None:
        kwargs["llama_4_scaling"] = llama_4_scaling
    if topk_indices is not None:
        kwargs["topk_indices"] = topk_indices[:real_num_tokens]

    original_out_cache_loc = forward_batch.out_cache_loc
    # Keep the original ForwardBatch object and only narrow cache locations for
    # this backend call so model/backend state is still written to the same batch.
    forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]

    # Store pre-allocated output for FA backend to write directly into.
    # Must slice to real_num_tokens to match the narrowed query shape —
    # the FA kernel validates out.size(0) == q.size(0).
    forward_batch._attn_output = output[:real_num_tokens]

    ret = get_attn_backend().forward(
        query,
        key,
        value,
        attention_layer,
        forward_batch,
```

不变量：custom op 没有绕过后端，只是把 graph 友好的输出 buffer 和 token 裁剪包了一层。这里临时修改的是同一个 `ForwardBatch` 对象上的 `out_cache_loc`，调用结束必须恢复；这正是它是“可变执行视图”而非不可变事实包的直接证据。

## Graph buffer 流：图外刷新动态字段

Triton 的 graph 外 metadata 会在 capture 和 replay 前刷新 KV 索引、SWA 写入位置、物理地址翻译。capture 时可创建固定 view，replay 时必须复用 capture 已记录的对象并原地填充；“图外”并不授权替换任意 capture-stable 指针。

```python
# 来源：sglang/python/sglang/srt/layers/attention/triton_backend.py L540-L603
    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info

        if in_capture:
            assert forward_batch.encoder_lens is None, "Not supported"
            # Multi-step spec decode: kv buffers come from spec_info, not the
            # cuda-graph pool, so replay is not involved.
            if forward_mode.is_decode_or_idle() and spec_info is not None:
                self.forward_metadata = ForwardMetadata(
                    attn_logits=self.cuda_graph_attn_logits,
                    attn_lse=self.cuda_graph_attn_lse,
                    max_extend_len=None,
                    num_kv_splits=self.cuda_graph_num_kv_splits,
                    kv_indptr=spec_info.kv_indptr,
                    kv_indices=spec_info.kv_indices,
                    qo_indptr=None,
                    custom_mask=None,
                    mask_indptr=None,
                    window_kv_indptr=self.window_kv_indptr,
                    window_kv_indices=None,
                    window_num_kv_splits=None,
                    window_kv_offsets=None,
                    swa_attn_logits=self.cuda_graph_swa_attn_logits,
                )
                return

            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
            )
            out_cache_loc_full_physical = self._translate_cuda_graph_shared_pool_locs(
                forward_batch, bs
            )
            swa_out_cache_loc = self._fill_cuda_graph_swa_out_cache_loc(forward_batch)
            self.forward_metadata = self._build_cuda_graph_forward_metadata(
                bs,
                forward_mode,
                spec_info,
                swa_out_cache_loc,
                out_cache_loc_full_physical,
            )
        else:
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
            )
            # Metadata view is reused from capture; just refill the buffers.
            self._translate_cuda_graph_shared_pool_locs(forward_batch, bs)
            self._fill_cuda_graph_swa_out_cache_loc(forward_batch)
```

排障入口：如果 replay 后读到旧 KV 或写错 slot，优先看这些 graph 外 buffer 是否按当前 batch 刷新。

## 复盘迁移

- 从 vLLM 迁移概念时，可以把 block table 与这里的 `kv_indptr` / `kv_indices` 作功能类比，但不能直接等同；SGLang 还叠加 radix prefix、generic/physical 地址翻译和多层 backend wrapper。
- 从 FlashAttention 迁移概念时，kernel 内部仍是 attention 计算；本专题关注的是 kernel 之前的 paged KV 参数编译。
- 从 CUDA Graph 排障迁移时，先区分 graph 外 metadata 刷新和 graph 内静态 op，再判断是不是 kernel 自身问题。
