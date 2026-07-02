---
type: batch-doc
module: 14-Models-专用
batch: "14"
doc_type: concept
title: "Models 专用 · 核心概念"
tags:
 - sglang/batch/14
 - sglang/module/models-specialized
 - sglang/doc/concept
aliases:
 - "01-核心概念"
updated: 2026-07-02
---
# Models 专用 · 核心概念

> 本节介绍 DeepSeek 专用架构在 SGLang 中的扩展点与核心术语。

---

## 用户故事：DeepSeek V3 上线 — MLA absorb 路径选错

### Persona

**程远**，推理平台工程师。团队要把 DeepSeek-V3 从 vLLM 迁到 SGLang，压测发现 **首包延迟正常、decode 带宽却掉 30%**。日志里 attention backend 落在 `trtllm_mla`，但某层仍走了未 absorb 的 MHA 路径——他需要搞清 MLA / DSA / MoE 在 SGLang 里各由哪个 Mixin 与 EntryClass 负责。

### 时间线

| 时刻 | 事件 |
|------|------|
| T0 | `ModelRegistry` 按 `architectures` 选中 `DeepseekV3ForCausalLM` |
| T1 | 各层 `DeepseekV2Attention` 挂载 MLA 投影 + 可选 `Indexer`（DSA） |
| T2 | MoE 层 `DeepseekV2MoE` 路由 top-k expert → EP/TP dispatch |
| T3 | `ForwardMixin` 按 backend 切换 absorb / 非 absorb forward |
| T4 | 压测对比 `--attention-backend trtllm_mla` 与错误 fallback |

### 如果…会怎样

| 现象 | 可能原因 | 排查 |
|------|----------|------|
| decode 带宽低 | MLA absorb 未生效 | 查 `DeepseekMLAForwardMixin` 是否被选中 |
| DSA 层 OOM | prefill CP 未开或 topk 过大 | `dsa_enable_prefill_cp`、`Indexer` 配置 |
| MoE 负载倾斜 | EPLB 未启用 | 见 [[18-MoE-00-MOC]] |

---

## 1. MLA（Multi-head Latent Attention）

**Explain：** DeepSeek V2/V3 将 KV 投影到低秩 latent（`kv_lora_rank`），再通过 `kv_b_proj` 展开；Q 也可经 `q_lora_rank` 压缩。推理时「absorb」路径把部分矩阵乘合并进 attention kernel，减少带宽。SGLang 用多个 **ForwardMixin**（`DeepseekMLAForwardMixin` 等）按硬件/backend 切换实现。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L1607-L1625
        # For tensor parallel attention
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=self._get_q_b_proj_quant_config(quant_config),
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
```

**Comment：** 无 `q_lora_rank` 时退化为分离 `q_proj` + `kv_a_proj_with_mqa`（仍可为 MLA 布局）。

---

## 2. DSA（DeepSeek Sparse Attention，V3.2+）

**Explain：** `is_deepseek_dsa(config)` 为真时，Attention 层挂载 `Indexer` 子模块，prefill 阶段选 top-k token 索引，decode 可 skip 层间重复 topk 计算（`skip_topk` / `next_skip_topk`）。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L1646-L1674
        if self.use_dsa:
            is_neox_style = not getattr(config, "indexer_rope_interleave", False)
            self.indexer = Indexer(
                hidden_size=hidden_size,
                index_n_heads=get_dsa_index_n_heads(config),
                index_head_dim=get_dsa_index_head_dim(config),
                rope_head_dim=qk_rope_head_dim,
                index_topk=get_dsa_index_topk(config),
                q_lora_rank=q_lora_rank,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                scale_fmt="ue8m0",
                block_size=128,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                prefix=add_prefix("indexer", prefix),
                quant_config=quant_config,
                layer_id=layer_id,
                alt_stream=alt_stream,
            )
            # Refer: https://arxiv.org/abs/2603.12201 for more details.
            # skip_topk: when True, this layer will skip computation and reuse previous layer's topk indices.
            # next_skip_topk: when True, the next layer will skip computation and reuse this layer's topk indices.
            if is_nextn:
                self.skip_topk = True
                self.next_skip_topk = True
            else:
                self.skip_topk = dsa_layer_skips_topk(config, layer_id)
                self.next_skip_topk = dsa_layer_skips_topk(config, layer_id + 1)
```

**Comment：** DSA 与 Context Parallel（`dsa_enable_prefill_cp`）绑定；CP 目前仅支持 DSA 模型。

---

## 3. MoE 层：routed + shared experts

**Explain：** 稀疏层用 `DeepseekV2MoE` 替代 dense MLP。Gate 产生 expert 路由；`n_shared_experts` 可 fusion 进 MoE kernel（减少单独 GEMM）。DeepEP 模式下 expert 布局扩展为 `n_routed + ep_size` 槽位。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L564-L573
        if _is_deepep_fusion:
            # 256 routed + EP_size shared slots = 272 experts total (for EP=16)
            num_experts_for_moe = config.n_routed_experts + self.moe_ep_size
            top_k_for_moe = config.num_experts_per_tok + 1  # 8 routed + 1 shared
            # Interleaving for DeepEP dispatch is handled by TopK internally.
        else:
            num_experts_for_moe = (
                config.n_routed_experts + self.num_fused_shared_experts
            )
            top_k_for_moe = config.num_experts_per_tok + self.num_fused_shared_experts
```

**Comment：** `tp_size > n_routed_experts` 会直接 `ValueError`——MoE 常用 EP 而非大 TP。

---

## 4. DecoderLayer：Attention + MoE/MLP 分支

**Explain：** `DeepseekV2DecoderLayer` 根据 `_is_layer_sparse` 构造 `DeepseekV2MoE` 或 `DeepseekV2MLP`；`layer_communicator` 在 CP 启用时用 `DSACPLayerCommunicator`，并向 attention 传入 `prepare_qkv_latent` 回调。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2102-L2127
        if self.is_layer_sparse:
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=moe_quant_config_override or quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
                dsa_enable_prefill_cp=dsa_enable_prefill_cp,
                mla_enable_prefill_cp=mla_enable_prefill_cp,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
                swiglu_limit=getattr(config, "swiglu_limit", None),
            )
```

---

## 5. AttnForwardMethod 枚举

**Explain：** `dispatch_attn_forward_method` 综合 forward_mode、量化格式、是否 DSA、是否 piecewise CUDA graph、CPU/ROCm 等，返回 `AttnForwardMethod` 枚举值，决定走哪套 prepare/core。

**Code（类定义位置）：**

```python
# 来源：python/sglang/srt/models/deepseek_common/attention_forward_methods.py
# （deepseek_v2.py L153-L159 导入）
from sglang.srt.models.deepseek_common.attention_forward_methods import (
 AttnForwardMethod,
 DeepseekMHAForwardMixin,
 DeepseekMLAForwardMixin,
 ...
)
```

**Comment：** Mixin 模式把 3000+ 行 deepseek_v2.py 按 backend 切开；读代码时先找 `dispatch_attn_forward_method` 再进对应 mixin。

---

## 6. Context Parallel（Prefill CP）

**Explain：** DeepSeek ForCausalLM 在 forward 入口根据 token 数与 CP 配置设置 `forward_batch.attn_cp_metadata`，供 attention 在 prefill 时切 sequence 维。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2813-L2832
        if self.dsa_enable_prefill_cp:
            if can_dsa_cp_split(
                len_input_ids, self.cp_size, self.use_dsa, forward_batch
            ):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len_input_ids,
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
        elif self.mla_enable_prefill_cp:
            if can_cp_split(len_input_ids, self.cp_size, forward_batch):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len_input_ids,
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
```

---

## 7. Shared experts fusion 自动决策

**Explain：** `determine_num_fused_shared_experts` 在模型 init 时根据 GPU capability、量化格式、EP/SBO/TBO、architecture 白名单等决定是否 fusion；不满足条件会强制 `disable_shared_experts_fusion` 并打 log。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2736-L2745
        if server_args.disable_shared_experts_fusion:
            return

        disable_reason = None
        if server_args.enforce_shared_experts_fusion:
            pass
        elif is_sbo_enabled() or is_tbo_enabled():
            disable_reason = "SBO/TBO enabled: incompatible with fusing shared expert into MoE kernel."
        elif is_deepep_class_backend():
            disable_reason = "DeepEP: fusion off by default (use --enforce-shared-experts-fusion to enable)."
```

**Comment：** Kimi-K2.5 等变体需匹配 `n_routed_experts` 与 quant 白名单，否则 fusion 会 silent 错误加载。

---

## 8. RadixAttention 在 MLA 中的双实例

**Explain：** DeepSeek MLA 层可能注册两个 `RadixAttention`（`attn_mqa` 与 `attn_mha`），共享 `layer_id`；piecewise CUDA graph 在 MHA 路径用 companion 实例修正 head 元数据（见 RadixAttention `radix_attention.py` L188-L193）。

**Comment：** 模型开发者添加新 MLA 变体时需保持 `layer_id` 与 backend 注册一致。

---

## 9. Weight loader：fused QKV A proj

**Explain：** Quark 等量化 checkpoint 将 `q_a_proj` 与 `kv_a_proj_with_mqa` 存为分离权重；SGLang 在 `fuse_qkv_a_proj` 为真时通过 `packed_modules_mapping` 告知 quant config 做 unfuse 映射。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2663-L2676
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        # Quant configs like Quark may rely on the model to provide fused-module
        # mappings so exclusion checks can unfuse derived names back to the
        # checkpoint's source layer names.
        if quant_config is not None:
            quant_config.update_packed_modules_mapping(self.packed_modules_mapping)
```

---

## 10. 与通用 Llama 模型的接口对齐

**Explain：** 对外仍暴露相同 `forward(input_ids, positions, forward_batch, ...)`；差异在 `DeepseekV2Model` 内部传递 `zero_allocator`、MoE buffer context、TBO 双 batch 等。Scheduler / ModelRunner 无 DeepSeek 专用分支。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2800-L2847
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        # Minor fix for multi-modal model: input_ids is None
        len_input_ids = (
            input_ids.shape[0] if input_ids is not None else input_embeds.shape[0]
        )
        if self.dsa_enable_prefill_cp:
            if can_dsa_cp_split(
                len_input_ids, self.cp_size, self.use_dsa, forward_batch
            ):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len_input_ids,
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
        elif self.mla_enable_prefill_cp:
            if can_cp_split(len_input_ids, self.cp_size, forward_batch):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len_input_ids,
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states
```
