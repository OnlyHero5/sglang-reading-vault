---
type: batch-doc
module: 14-Models-专用
batch: "14"
doc_type: faq
title: "专用模型实现 · 关键问题"
tags:
  - sglang/batch/14
  - sglang/module/models-specialized
  - sglang/doc/faq
aliases:
  - "04-关键问题"
updated: 2026-07-02
---
# 专用模型实现 · 关键问题

---

## Q1：DeepseekV2 和 DeepseekV3 在同一个文件，怎么区分？

**Explain：** Registry 按 **类名** 区分；实现共用 `DeepseekV2AttentionMLA`、`DeepseekV2MoE` 等。V3 特有行为由 config 字段触发（`n_routed_experts`、`q_lora_rank`、DSA 相关字段），而非 fork 整套文件。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2966
EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]
```

---

## Q2：什么时候走 MHA 而不是 MLA absorb？

**Explain：** `dispatch_attn_forward_method` 根据 backend 能力、forward_mode、量化、是否 CPU/ROCm 等决定。不支持 FA3 MLA 或 debug 路径可能回退 MHA；chunked KV 用于超长 prefill。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L1890-L1908
        attn_forward_method = self.dispatch_attn_forward_method(forward_batch)
        if attn_forward_method == AttnForwardMethod.MHA:
            inner_state = self.forward_normal_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            inner_state = self.forward_normal_chunked_kv_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_ONE_SHOT:
            inner_state = self.forward_normal_one_shot_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MLA:
            inner_state = self.forward_absorb_prepare(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
```

**Comment：** 具体分支表在 `deepseek_common/attention_backend_handler.py`。

---

## Q3：为什么 `tp_size > n_routed_experts` 报错？

**Explain：** MoE 路由按 **expert 数** 切 EP，不是按 hidden dim TP。TP 大于 expert 数会导致 gate 权重 shard 语义错误。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L583-L587
        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )
```

---

## Q4：shared experts fusion 关闭的常见原因？

**Explain：** SBO/TBO、DeepEP 默认、非 V3/R1 architecture、GPU capability 不足、W4AFP8 量化、Kimi 等非标准 checkpoint 等。关闭后 shared expert 单独 GEMM，功能正确但慢。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2786-L2792
        if disable_reason is not None:
            server_args.disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
```

---

## Q5：DSA 的 `skip_topk` 是什么？

**Explain：** 若上一层已算 top-k index，下一层可 `skip_topk=True` 直接复用 `prev_topk_indices`，省 Indexer 开销。NextN speculative 层默认两层都 skip。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L1669-L1674
            if is_nextn:
                self.skip_topk = True
                self.next_skip_topk = True
            else:
                self.skip_topk = dsa_layer_skips_topk(config, layer_id)
                self.next_skip_topk = dsa_layer_skips_topk(config, layer_id + 1)
```

---

## Q6：DeepEP fusion 如何改变 expert 数？

**Explain：** 每个 EP rank 持有一个 **shared expert 槽**，总 expert 数 = `n_routed + ep_size`；top-k = `num_experts_per_tok + 1`（多选 1 个 shared 槽）。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L564-L567
        if _is_deepep_fusion:
            # 256 routed + EP_size shared slots = 272 experts total (for EP=16)
            num_experts_for_moe = config.n_routed_experts + self.moe_ep_size
            top_k_for_moe = config.num_experts_per_tok + 1  # 8 routed + 1 shared
```

---

## Q7：`get_attn_tp_context()` 解决什么问题？

**Explain：** DeepSeek QLoRA 路径下，Attention 输入可能在 attn_tp group 内 scattered；MLP 可能在另一 parallel layout。context 在 model forward 外统一 `maybe_input_scattered`，层内 `clear_attn_inputs` 防泄漏。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2219
        get_attn_tp_context().clear_attn_inputs()
```

---

## Q8：Hash MoE 与 learned gate 如何共存？

**Explain：** 仅 `layer_id < num_hash_layers` 用 `HashTopK`；更深稀疏层用 `MoEGate` + 常规模型 TopK。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L637-L647
        if self.is_hash and not (is_nextn and is_deepseek_v4):
            self.topk = HashTopK(
                topk=config.num_experts_per_tok + self.num_fused_shared_experts,
                num_experts=config.n_routed_experts,
                num_fused_shared_experts=self.num_fused_shared_experts,
                vocab_size=config.vocab_size,
                scoring_func=config.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
                layer_id=self.layer_id,
            )
```

---

## Q9：与 Llama 相比 load_weights 额外处理什么？

**Explain：** Expert 权重 remap（EP rank 本地 expert id）、`fused_qkv_a_proj_with_mqa` packed 映射、shared expert fusion 槽位、`is_nextn` 独立权重集。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2857-L2858
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        self.do_load_weights(weights, is_nextn)
```

---

## Q10：Prefill CP 与 Decode CP 区别？

**Explain：** 本文件入口主要设置 **prefill CP**（`dsa_enable_prefill_cp` / `mla_enable_prefill_cp`）。Decode CP 走 `prepare_decode_context_parallel_metadata`（`dcp_utils`），在 attention mixin 内触发。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2713-L2716
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.mla_enable_prefill_cp = (
            is_prefill_context_parallel_enabled() and not is_deepseek_dsa(config)
        )
```

---

## 验证建议（零基础可试）

1. **同文件三版本 EntryClass 如何共存** 
 - 操作：打开 `python/sglang/srt/models/deepseek_v2.py` 文件末尾，读 `EntryClass = [...]` 列表；再打开 DeepSeek 模型 `config.json` 的 `architectures[0]`。 
 - 预期：Registry 按**类名字符串**分别注册 V2/V3/V3.2；实现共用 MLA/MoE 模块，版本差异由 config 字段驱动而非拆文件。 
 - 对应：Q1、[[14-Models-专用-01-核心概念|01-核心概念 §EntryClass]]、[[14-Models-专用-05-checkpoint]]

2. **`tp_size > n_routed_experts` 为何必报错** 
 - 操作：读模型 `config.json` 的 `n_routed_experts`（或 `num_experts`），对比计划使用的 `--tp-size`；在源码定位 Q3 的 `raise ValueError`。 
 - 预期：`tp_size` 大于 routed expert 数时初始化即失败——MoE 按 expert 数做 EP，不是按 hidden dim TP。 
 - 对应：Q3、[[14-Models-专用-03-数据流与交互|03-数据流与交互 §MoE 边界]]

3. **Shared experts fusion 被 disable 的原因** 
 - 操作：启动 DeepSeek 模型时在 rank0 日志搜索 `Shared experts fusion optimization is disabled`；对照 Q4 列举的 disable 条件（SBO/TBO、DeepEP、非 V3 arch、量化等）。 
 - 预期：命中任一条件则 `num_fused_shared_experts=0`，功能仍正确但 shared expert 走独立 GEMM（更慢）。 
 - 对应：Q4、[[14-Models-专用-02-源码走读|02-源码走读 §determine_num_fused_shared_experts]]

4. **MLA 与 MHA 路径如何分派（读代码 + 可选日志）** 
 - 操作：在 `deepseek_v2.py` grep `dispatch_attn_forward_method`，列出返回 `AttnForwardMethod.MHA` vs `MLA` 的分支；若有 GPU，extend 阶段日志可对照 `forward_absorb_prepare` 是否被调用。 
 - 预期：backend 不支持 FA3 MLA、debug/CPU 路径等会回退 MHA；正常 V3 + 支持 backend 走 MLA absorb。 
 - 对应：Q2、[[14-Models-专用-01-核心概念|01-核心概念 §MLA dispatch]]、`deepseek_common/attention_backend_handler.py`
