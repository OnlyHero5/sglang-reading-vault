---
title: "Attention · 排障指南"
type: troubleshooting
framework: sglang
topic: "Attention"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# Attention · 排障指南

## 你为什么要读

设置 `--attention-backend` 只表达偏好，最终路径还受硬件、dtype、head 配置、forward mode 和功能兼容性约束。本文从“实际选了谁”开始，再检查 metadata、KV 布局和 kernel，避免拿配置字符串替代运行证据。

这篇按症状排障。每个问题都回到源码入口，而不是只给经验结论。

## Q1：我设置了 `--attention-backend`，为什么 prefill 和 decode 还是可能不同？

因为通用 backend 只是默认值，prefill/decode 专用 flag 会覆盖它。

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

排障动作：查启动日志里 resolved prefill/decode 名字，不要只查命令行里有没有 `--attention-backend`。

## Q2：什么时候会创建 `HybridAttnBackend`？

只有两个最终 backend 名不同才创建 Hybrid；相同则创建单后端。

```python
# 来源：sglang/python/sglang/srt/model_executor/model_runner.py L2486-L2520
        (
            self.prefill_attention_backend_str,
            self.decode_attention_backend_str,
        ) = self.server_args.get_attention_backends()

        if self.decode_attention_backend_str != self.prefill_attention_backend_str:
            from sglang.srt.layers.attention.hybrid_attn_backend import (
                HybridAttnBackend,
            )

            attn_backend = HybridAttnBackend(
                self,
                decode_backend=self._get_attention_backend_from_str(
                    self.decode_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
                prefill_backend=self._get_attention_backend_from_str(
                    self.prefill_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
            )
            logger.info(
                f"Using hybrid attention backend for decode and prefill: "
                f"decode_backend={self.decode_attention_backend_str}, "
                f"prefill_backend={self.prefill_attention_backend_str}."
            )
            logger.warning(
                "Warning: Attention backend specified by --attention-backend or default backend might be overridden."
                "The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
            )
        else:
            attn_backend = self._get_attention_backend_from_str(
                self.server_args.attention_backend,
                init_new_workspace=init_new_workspace,
            )
```

排障动作：如果你期待 Hybrid，但日志没有出现 hybrid backend，说明两个最终名字相同，或 draft worker 被 `speculative_draft_attention_backend` 覆盖。

## Q3：`TARGET_VERIFY` 到底走 prefill 还是 decode？

它不是固定答案。Hybrid 会看 `speculative_attention_mode`。

```python
# 来源：sglang/python/sglang/srt/layers/attention/hybrid_attn_backend.py L43-L52
        if forward_mode.is_decode_or_idle():
            return self.decode_backend
        elif forward_mode.is_target_verify():
            return (
                self.decode_backend
                if self.model_runner.server_args.speculative_attention_mode == "decode"
                else self.prefill_backend
            )
        else:
            return self.prefill_backend
```

排障动作：投机解码 verify 变慢或结果异常时，同时检查 `forward_batch.forward_mode` 和 `speculative_attention_mode`，不要只按 extend/decode 字面理解。

## Q4：CUDA Graph capture 报 host sync，先看哪里？

先看 backend 的 `init_forward_metadata_in_graph`，这里不能调用 `.item()`、`.cpu()`、`.tolist()` 或动态 shape `torch.empty()`。

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

排障动作：把动态 metadata 刷新移动到 `init_forward_metadata_out_graph`，再对比禁用 CUDA Graph 后是否恢复。

## Q5：为什么 FlashInfer merge state 有时会 fallback 到 Triton？

DP attention 可能让 `num_heads` 变大，FlashInfer merge state 的 CUDA block 线程数可能超过限制。源码在头数超过安全阈值时改用 Triton 实现。

```python
# 来源：sglang/python/sglang/srt/layers/attention/flashinfer_backend.py L105-L116
    def _safe_merge_state(
        v_a: torch.Tensor,
        s_a: torch.Tensor,
        v_b: torch.Tensor,
        s_b: torch.Tensor,
    ):
        num_heads = v_a.shape[1]
        head_dim = v_a.shape[2]
        max_heads = _merge_state_max_safe_num_heads(head_dim, v_a.element_size())
        if num_heads <= max_heads:
            return merge_state(v_a, s_a, v_b, s_b)
        return merge_state_triton(v_a, s_a, v_b, s_b)
```

排障动作：如果日志或 profiling 显示 merge state 走 Triton，不要立即认为 backend 配置错了；先计算 DP attention 下的 head 数是否触发安全 fallback。

## Q6：为什么 decode 也要写 KV cache？

decode 每步生成一个新 token，这个 token 的 K/V 必须写入 paged KV pool，下一步才会成为历史 KV。

```python
# 来源：sglang/python/sglang/srt/layers/attention/flashinfer_backend.py L1095-L1127
        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        # Call the wrapped function
        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)
```

排障动作：出现“第一步正常，后续 token attention 错”的问题时，同时检查本步写入 `out_cache_loc` 和 wrapper 读取的 KV buffer。

## Q7：FlashInfer 和 Triton 的差异在哪里？

文件头说明了 SGLang 的默认判断：FlashInfer 通常更快，Triton 更容易定制；两者都支持 extend 和 decode。

```python
# 来源：sglang/python/sglang/srt/layers/attention/flashinfer_backend.py L5-L10
"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""
```

排障动作：生产吞吐优先时先理解 FlashInfer wrapper 和 Graph；改 mask、新硬件、新实验 kernel 时先读 Triton metadata 和 kernel。

## Q8：新增一个 backend 最容易漏什么？

最容易只接上 `forward_decode` 或 `forward_extend`，但漏掉 metadata、Graph state、verify buffer、SWA/cross-attention wrapper 等边界。基类的契约说明这些都是 backend 的责任。

```python
# 来源：sglang/python/sglang/srt/layers/attention/base_attn_backend.py L18-L43
class AttentionBackend(ABC):
    """The base class of attention backends.

    Forward-data init contract (3 methods):

      - ``init_forward_metadata(fb)`` — eager entry point. Default is a wrapper
        that calls ``_out_graph(fb)`` then ``_in_graph(fb)``. Backends may
        override to keep an independent eager body.
      - ``init_forward_metadata_out_graph(fb, in_capture=False)`` — per-iter
        metadata prep, runs outside ``with graph.capture():``. Capture
        sites pass ``in_capture=True``; replay/eager use the default
        ``False``. Backends read ``in_capture`` only when capture / replay
        bodies diverge.
      - ``init_forward_metadata_in_graph(fb)`` — graph-recordable static-shape
        GPU op, runs inside ``with graph.capture():`` at capture time and
        is auto-replayed by ``graph.replay()``. Default is no-op.

    The legacy ``init_forward_metadata_capture_cuda_graph`` and
    ``init_forward_metadata_replay_cuda_graph`` overrides are fully
    deprecated and removed from the ABC: out-of-tree backends overriding
    those must migrate to ``init_forward_metadata_out_graph(fb, in_capture)``.
    """

    # Resolved per-mode backend names, stamped by ModelRunner.init_attention_backend
    prefill_attention_backend_str: Optional[str] = None
    decode_attention_backend_str: Optional[str] = None
```

排障动作：新 backend 先跑 eager extend/decode，再跑 decode CUDA Graph，再跑 target verify 或 SWA/cross-attention 场景。只过单步 eager 不够。

## 运行验证

Attention backend 的排障先确认三层边界：配置如何拆成 prefill/decode、`HybridAttnBackend` 如何分派、具体 backend 是否同时实现 metadata、extend、decode 和 KV 写入。

```powershell
rg -n 'def get_attention_backends|def init_attention_backend|class HybridAttnBackend|class AttentionBackend|class FlashInferAttnBackend|def init_forward_metadata|def forward_decode|def forward_extend|decode_wrapper|set_kv_buffer|KVWriteLoc' sglang/python/sglang/srt/server_args.py sglang/python/sglang/srt/model_executor/model_runner.py sglang/python/sglang/srt/layers/attention/hybrid_attn_backend.py sglang/python/sglang/srt/layers/attention/base_attn_backend.py sglang/python/sglang/srt/layers/attention/flashinfer_backend.py
```

读输出时先看 `server_args.py` 的 backend 拆分，再看 `model_runner.py` 的初始化；如果命中 `HybridAttnBackend`，就继续看它如何按 `ForwardMode` 分派。最后在 `flashinfer_backend.py` 里确认 `forward_extend` 和 `forward_decode` 都会在需要时 `set_kv_buffer`，否则 decode 侧可能读到旧 KV。
