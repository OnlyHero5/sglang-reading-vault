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
updated: 2026-07-11
---
# Attention · 排障指南

## 你为什么要读

设置 `--attention-backend` 只表达偏好，最终路径还受硬件、dtype、head 配置、forward mode 和功能兼容性约束。本文从“实际选了谁”开始，再检查 metadata、KV 布局和 kernel，避免拿配置字符串替代运行证据。

这篇按症状排障。每个问题都回到源码入口，而不是只给经验结论。

## Q1：命令行写了一个 backend，为什么实际运行不是它？

因为 flag 是输入，不是最终事实。post-init 会先做设备/模型默认选择和兼容性处理；prefill/decode 专用字段又可能覆盖通用字段，未指定的一侧才回退到归一后的通用字段。

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

排障动作：依次记录原始命令行、`ServerArgs` post-init 后三个字段、`get_attention_backends()` 返回的两个名字、最终对象类型。若 page size、Graph 配置或 backend 被改写，再回到 `_handle_attention_backend_compatibility()` 查触发条件。

## Q2：日志说 Hybrid，为什么我仍找不到真正执行 kernel 的对象？

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

这张卡只说明 per-mode `HybridAttnBackend` 的创建条件。每个子 backend 在进入它之前已经过 `attn_backend_wrapper()`，混合线性模型可能先成为 `HybridLinearAttnBackend`；外层还可能再套 TBO，PDMux 则会创建主对象和多份 decode workspace。

排障动作：打印或断点展开对象树，至少检查 `prefill_backend`、`decode_backend`、`full_attn_backend`、`linear_attn_backend`、TBO children/PDMux group。若你期待 per-mode Hybrid 但日志没有出现，才检查两个 resolved 名是否相同，或 draft worker 是否被 `speculative_draft_attention_backend` 覆盖。

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

排障动作：把 host/dynamic 计划留在 `init_forward_metadata_out_graph`，图内只保留静态 shape GPU op；同时确认 capture 已记录的 tensor 地址没有在 replay 前被重新赋值，动态数据应写回固定 buffer。禁用 Graph 只用于判断错误是否转入 eager，并不能单独证明根因就在 metadata。

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

排障动作：如果 profiling 显示 merge state 走 Triton，不要立即认为整个 backend 配置被替换；先计算 DP attention 下的 head 数是否触发安全 fallback。这个 fallback 只替换 cascade merge 这一个辅助 kernel。

## Q6：decode 写 KV 时，`out_cache_loc` 能否直接当物理 slot？

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

不能无条件这样理解。`out_cache_loc` 是 generic write location；普通 pool 中可与物理位置一致，Unified 下可为 virtual。SWA/full pool 还可能分别使用 `KVWriteLoc.swa_loc` 与 `full_loc`；cross-attention 改用 `encoder_out_cache_loc`。

排障动作：出现“第一步正常，后续 token 错”时，同时记录 pool 类型、generic 位置、翻译后的 physical 位置、实际 `set_kv_buffer` 分支与 wrapper 的读取索引流。不要只盯一个 tensor 数值。

## Q7：应该无条件优先 FlashInfer，还是优先 Triton？

没有脱离环境的正确答案。当前 SGLang 后端集合还包括 FA3/FA4、MLA、DSA/DSV4、TensorRT-LLM、AMD、Ascend、Intel 等实现；某个旧文件头里“只有两个 backend”的注释已不能代表仓库架构。backend 选择同时受模型架构、GPU/加速器、page size、KV dtype、head 形状、Graph、speculative decoding、SWA 与 cross-attention 约束。

排障动作：先确认 workload 与功能约束，再验证候选 factory 的断言和兼容性处理；最后在固定模型、硬件、batch、context、dtype 与 Graph bucket 下实测。不能把“第三方 wrapper”或“容易改 kernel”直接翻译成生产性能结论。

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

排障动作：新 backend 先跑 eager extend/decode，再按其声明能力跑 Graph capture/replay、target verify、SWA/cross-attention、Unified 地址翻译和 DP padding。若某能力不支持，应在 resolver/factory 尽早拒绝，而不是等到深层 kernel 才崩。

## Q9：为什么我明明改了 `HybridAttnBackend.forward`，运行行为却没变？

先检查是否改到了文件中第一个同名定义。当前类体里有两个 `forward`，Python 会让后定义覆盖前定义；前一个支持 `mixed_qkv/a/b` 的版本在运行时不可达。混合线性模型真正的按层能力来自 `attn_backend_wrapper()` 返回的 `HybridLinearAttnBackend`。

排障动作：不要只用 `rg 'def forward'` 后停在第一个命中。检查类定义末尾、运行时 `type(obj)`、`type(obj).__dict__['forward']`，再沿 wrapper 委托链找到实际绑定方法。

## Q10：Graph replay 或 multi-step draft 为什么会读到旧 metadata？

先判断 metadata 是否已被专用 planner 预先生成。`forward_metadata_ready` 会记录计划时 batch size 与 token 数；DP padding 后 shape 可能变化。只有标记了 `replan_equivalent=True` 的预计划，普通 forward 才能在失效时安全重建；其他路径重建会覆盖专用 wrapper metadata。

排障动作：记录“谁调用了 plan、何时 mark ready、计划 shape、进入模型时最终 shape、是否允许等价 replan”。Graph 路径还要确认 replay 前只是原地刷新 capture-stable buffer，而不是替换对象。

## 运行验证

Attention backend 的排障先确认三层边界：配置如何拆成 prefill/decode、`HybridAttnBackend` 如何分派、具体 backend 是否同时实现 metadata、extend、decode 和 KV 写入。

```powershell
rg -n '_handle_attention_backend_compatibility|def get_attention_backends|def init_attention_backend|attn_backend_wrapper|class HybridAttnBackend|class HybridLinearAttnBackend|forward_metadata_ready|needs_forward_metadata_init|def init_forward_metadata|decode_wrapper|set_kv_buffer|KVWriteLoc' sglang/python/sglang/srt/server_args.py sglang/python/sglang/srt/model_executor/model_runner.py sglang/python/sglang/srt/model_executor/forward_batch_info.py sglang/python/sglang/srt/layers/attention
```

读输出时先看 resolver，再展开 wrapper 树；然后确认 metadata 是本轮新计划还是预计划，最后才下钻具体 backend 的 KV 写入、索引更新与 kernel 调用。这个顺序能把“选错对象”“复用旧计划”“地址翻译错”和“kernel 数学错”分成四类问题。
