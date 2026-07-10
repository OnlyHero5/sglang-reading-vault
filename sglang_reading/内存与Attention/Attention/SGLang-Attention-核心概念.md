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
updated: 2026-07-10
---
# Attention · 核心概念

## 你为什么要读

这篇先建立心理模型：Attention 后端不是某个单独 kernel 文件，而是连接 `ForwardBatch`、每层 Q/K/V、paged KV cache 和具体 kernel 的编译层。它先决定走哪条 backend，再把当前 batch 的 KV 索引、query 边界、Graph buffer、SWA/cross-attention 变体整理成 kernel metadata。

## 四层模型

| 层 | 对象 | 责任 |
|----|------|------|
| 配置层 | `attention_backend`、`prefill_attention_backend`、`decode_attention_backend` | 得到 prefill/decode 的最终后端名 |
| 调度语义层 | `ForwardMode` | 表示本轮是 extend、decode、mixed、idle、target verify |
| 每层入口 | `RadixAttention.forward` | 把 Q/K/V reshape 后交给当前 attention backend |
| 后端实现层 | `AttentionBackend` 子类 | 准备 metadata，写 KV，调用 FlashInfer/Triton 等 kernel |

读者抓手：遇到 attention 路径问题时，先问四个问题：配置解析成了什么 backend；当前 `ForwardMode` 是什么；`RadixAttention` 是否走 piecewise custom op；后端 metadata 是否和 kernel 调用一致。

## 配置不是一个开关，而是两个槽位

`ServerArgs.get_attention_backends` 把显式 prefill/decode flag 和通用 `attention_backend` 合成两个最终名字。

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

这说明 `--attention-backend` 是默认值，`--prefill-attention-backend` 与 `--decode-attention-backend` 是覆盖槽位。读排障日志时，不要只看用户传了哪个 flag，要看最终 resolved backend。

## `ForwardMode` 是运行时语义

`ForwardMode` 把同一层 attention 的计算形态拆开。extend 不等于 decode；`TARGET_VERIFY` 在接口上属于 extend，但 Hybrid 后端还会按 speculative 配置决定走 prefill 还是 decode 子后端。

```python
# 来源：sglang/python/sglang/srt/model_executor/forward_batch_info.py L78-L90
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

不变量：backend 选路必须跟 `ForwardMode` 对齐。把 `TARGET_VERIFY` 当普通 prefill 或普通 decode，都可能让 speculative verify 使用错误 kernel。

## metadata 三阶段是为了 CUDA Graph

`AttentionBackend` 把 metadata 准备分成 eager、graph 外、graph 内。图外允许动态 shape、host 读写、buffer 指针替换；图内只能录制静态 GPU op。

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

读者抓手：Graph 问题不要先看 kernel 数学，先看 metadata 是否把 host sync 放进了 graph 内。

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
