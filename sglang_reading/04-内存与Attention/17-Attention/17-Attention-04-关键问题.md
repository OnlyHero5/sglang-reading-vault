---
type: batch-doc
module: 17-Attention
batch: "17"
doc_type: faq
title: "Attention：关键问题"
tags:
 - sglang/batch/17
 - sglang/module/attention
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# Attention：关键问题

---

## Q1：FlashInfer vs Triton — 如何选型？

**Explain：** 二者实现同一 `AttentionBackend` 接口。FlashInfer 绑定高度优化的 paged KV CUDA kernel，**生产 decode 吞吐**通常更高；Triton 用 Python+Triton 编写 extend/decode kernel，**改 mask / 新架构**时迭代更快。默认路径：FlashInfer 可用且模型无 attention sinks → flashinfer；否则 triton。也可显式 `--attention-backend` 或分裂 `--prefill-attention-backend` / `--decode-attention-backend`。

**Code：**

```python
# 来源：python/sglang/srt/layers/attention/flashinfer_backend.py L1-L8（模块 docstring）
from __future__ import annotations

from sglang.srt.runtime_context import get_parallel

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
```

**Code：**

```python
# 来源：python/sglang/srt/server_args.py L6922-L6933
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

**Comment：**

| 场景 | 建议 |
|------|------|
| 生产 MHA、NVIDIA GPU、FlashInfer 可装 | decode 用 flashinfer |
| 调试新 mask / sinks / 自定义 metadata | triton |
| Hopper extend + Ampere decode 分裂实验 | Hybrid：`fa3` + `flashinfer` |
| MLA / DeepSeek | `trtllm_mla` 等专用 backend，见 [[17-Attention-01-核心概念|01-核心概念]] |

---

## Q2：为何 merge_state 有 Triton fallback？

**Explain：** DP attention 把多 rank 的 head 拼成更大 `num_heads`，FlashInfer 原生 `merge_state` 的 CUDA block 配置有上限。`_safe_merge_state` 检测 head 数超过 `max_heads` 时自动 fallback 到 Triton，避免 launch 失败。

**Code：**

```python
# 来源：python/sglang/srt/layers/attention/flashinfer_backend.py L105-L116
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

**Comment：** 用户无需改配置；日志中若见 triton merge 说明 head 数触达 FlashInfer 上限。

---

## Q3：CUDA Graph capture 失败 — 常见根因？

**Explain：** Graph 内只能录制**静态 shape、无 host sync** 的 GPU op。动态逻辑（`.item()`、`.cpu()`、`.tolist()`、动态 `torch.empty()`）必须放在 `init_forward_metadata_out_graph`。

| 误解 | 实际 |
|------|------|
| 任意 metadata 逻辑都可录进 graph | 仅 `init_forward_metadata_in_graph` 内 op 会被 replay |
| capture 失败应关 graph 了事 | 先查是否 host sync 误入 in_graph |
| Hybrid 与 graph 不兼容 | decode 子 backend 负责 graph state |

**Code：**

```python
# 来源：python/sglang/srt/layers/attention/base_attn_backend.py L68-L77（节选）
        diverge (e.g., snapshot metadata, swap buffer pointers, install
        temp workspace). Host op / dynamic-shape / non-graph-recordable
        logic lives here.

        Default: no-op.
        """

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Graph-recordable static-shape GPU op.

```

**Comment：** 排查：对比 `--disable-cuda-graph` 与默认配置的 decode 吞吐；失败栈若指向 in_graph 内 `.item()`，将逻辑上移到 out_graph。

---

## Q4：SGLang attention 与 vLLM PagedAttention 有何异同？

**Explain：** 二者底层都是 paged KV。vLLM block table 由 scheduler 维护，kernel 直接消费。SGLang 拆成 `ReqToTokenPool` + `TokenToKVPoolAllocator`，各 backend 在 `init_forward_metadata` 压成 `kv_indptr/kv_indices`。SGLang 额外有 Radix prefix cache，并支持 prefill/decode **分裂 backend**（HybridAttnBackend）。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/memory_pool.py L15-L20
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
```

```python
# 来源：python/sglang/srt/layers/attention/triton_backend.py L413-L415
            kv_indptr = self._fill_kv_indptr_and_indices(
                bs, seq_lens, req_pool_indices, self.cuda_graph_kv_indices
            )
```

**Comment：** 迁移读者：把 vLLM 的 block table mentally model 成 SGLang 的 `kv_indptr/kv_indices`，再加一层 Radix prefix 复用（RadixAttention / KV Cache）。

---

## Q5：EAGLE 投机时 Hybrid 如何选 backend？

**Explain：** `ForwardMode.TARGET_VERIFY` 时，`_select_backend` 读 `speculative_attention_mode`：`"decode"` 走 decode 子 backend，否则走 prefill 子 backend。CUDA Graph 初始化也据此决定是否 prefill 子 backend 需要 `init_cuda_graph_state`。

**Code：**

```python
# 来源：python/sglang/srt/layers/attention/hybrid_attn_backend.py L45-L50
        elif forward_mode.is_target_verify():
            return (
                self.decode_backend
                if self.model_runner.server_args.speculative_attention_mode == "decode"
                else self.prefill_backend
            )
```

**Comment：** 与 [[21-Speculative-00-MOC|投机解码 Speculative]] 交叉；改 spec 模式后若 verify 变慢，检查 attention 选路是否意外切到 prefill kernel。

---

## 验证建议（零基础可试）

1. **确认 backend 选型** 
 - 操作：启动后在日志搜索 `Using hybrid attention` 或 `attention backend` / `flashinfer` / `triton`。 
 - 预期：与 CLI `--attention-backend` 或 `--prefill-*` / `--decode-*` 一致。 
 - 对应：[[17-Attention-01-核心概念|01-核心概念]]

2. **Prefill vs Decode 耗时分布** 
 - 操作：长 prompt（2k token）短输出 vs 短 prompt 长输出，观察首包延迟与后续 token 间隔。 
 - 预期：长 prefill 首包慢；decode 单步较稳定。 
 - 对应：[[00-零基础先修|00-零基础先修]] §2

3. **CUDA Graph A/B** 
 - 操作：同一模型对比默认启动与 `--disable-cuda-graph` 的 decode 吞吐（可用 `--max-running-requests 1` 简化）。 
 - 预期：graph 开启时 decode 吞吐通常更高；dynamic shape 场景可能需 disable。 
 - 对应：[[17-Attention-03-数据流与交互|03-数据流与交互]] §5

4. **Hybrid 分裂是否生效** 
 - 操作：`--prefill-attention-backend triton --decode-attention-backend flashinfer`（需 FlashInfer 可用），日志应出现 hybrid 字样。 
 - 预期：extend 与 decode 阶段分别走不同 backend 实现。 
 - 对应：Q1、 [[17-Attention-03-数据流与交互|03 §2]]
