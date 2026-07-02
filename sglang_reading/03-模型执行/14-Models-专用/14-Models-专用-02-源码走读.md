---
type: batch-doc
module: 14-Models-专用
batch: "14"
doc_type: walkthrough
title: "Models 专用 · 源码走读"
tags:
 - sglang/batch/14
 - sglang/module/models-specialized
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# Models 专用 · 源码走读

> 走读顺序：`DeepseekV2AttentionMLA` → `DeepseekV2MoE` → `DeepseekV2DecoderLayer` → `DeepseekV2ForCausalLM`

---

## 1. DeepseekV2AttentionMLA

### 1.1 类继承与 Mixin 组合

**Explain：** 主类 `nn.Module` + 四个 ForwardMixin，分别实现 CUDA MLA absorb、ROCm、CPU、标准 MHA。运行时只调用一套 prepare/core。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L1541-L1547
class DeepseekV2AttentionMLA(
    nn.Module,
    DeepseekMHAForwardMixin,
    DeepseekMLAForwardMixin,
    DeepseekMLARocmForwardMixin,
    DeepseekMLACpuForwardMixin,
):
```

### 1.2 `forward` — prepare / core 两阶段

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L1836-L1855
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        layer_scatter_modes: LayerScatterModes = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
            layer_scatter_modes=layer_scatter_modes,
            llama_4_scaling=llama_4_scaling,
            prev_topk_indices=prev_topk_indices,
        )
        return self.forward_core(s)
```

**Comment：** `zero_allocator` 供 fused GEMM 临时 buffer 复用，降低 decode 分配开销。

### 1.3 `kv_b_proj` 与 MHA 路径共享

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L1867-L1868
        if self.attn_mha.kv_b_proj is None:
            self.attn_mha.kv_b_proj = self.kv_b_proj
```

**Comment：** MHA fallback 与 MLA 共用同一份 `kv_b_proj` 权重引用。

### 1.4 Indexer（DSA）挂载条件

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L1585-L1589
        self.use_dsa = is_deepseek_dsa(config)
        self.dsa_enable_prefill_cp = dsa_enable_prefill_cp
        self.mla_enable_prefill_cp = mla_enable_prefill_cp
        if self.dsa_enable_prefill_cp:
            assert self.use_dsa, "CP currently only supports deepseek v3.2 model"
```

---

## 2. DeepseekV2MoE

### 2.1 Gate + Experts 构造

**Explain：** `MoEGate` 计算 routing logits；`get_moe_impl_class(quant_config)` 返回 Triton/DeepEP/FlashInfer 等具体 `FusedMoE` 实现。

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L595-L635
        self.gate = MoEGate(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("gate", prefix),
            is_nextn=is_nextn,
            is_hash_moe=self.is_hash,
            is_deepseek_v4=is_deepseek_v4,
            dsa_enable_prefill_cp=dsa_enable_prefill_cp,
            mla_enable_prefill_cp=mla_enable_prefill_cp,
        )

        # scaling factor for fused shared experts on AMD-platform.
        # DeepEP doesn't need this: shared expert is only computed on home rank
        # (not all-reduced), so no 1/ep_size correction is needed.
        fused_shared_experts_scaling_factor = None
        if (
            self.moe_ep_size > 1
            and self.num_fused_shared_experts > 0
            and not _is_deepep_fusion
        ):
            # if enable_ep_moe tp_szie == ep_size, every gpu get shared experts gemm output
            # so we scale with 1 / self.moe_ep_size in ep mode which will make it equalation as in tp mode
            # with fused_shared_experts
            fused_shared_experts_scaling_factor = 1.0 / float(self.moe_ep_size)

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=num_experts_for_moe
            + get_global_server_args().ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=top_k_for_moe,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_method_type=getattr(
                config, "routing_method_type", RoutingMethodType.DeepSeekV3
            ),
            swiglu_limit=getattr(config, "swiglu_limit", None),
            prefix=add_prefix("experts", prefix),
        )
```

### 2.2 HashTopK（前几层 hash routing）

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L580-L581
        n_hash_layers = getattr(config, "num_hash_layers", 0)
        self.is_hash = layer_id < n_hash_layers and not (is_deepseek_v4 and is_nextn)
```

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

**Comment：** Hash MoE 不用 learned gate，按 token hash 选 expert；V4 NextN 层除外。

### 2.3 `forward` — dispatch / combine

**Explain：** MoE forward 经 `BaseDispatcher` 做 token→expert dispatch，expert GEMM 后 combine；DeepEP 走 `MaybeTboDeepEPDispatcher` 支持 TBO。细节在 `layers/moe/`，本模块只记入口在 `DeepseekV2MoE.forward`。

**Code（forward 签名）：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L853-L860
    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        input_ids: Optional[torch.Tensor] = None,
```

---

## 3. DeepseekV2DecoderLayer

### 3.1 Attention → Communicator → MLP 主链

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2195-L2254
        hidden_states_orig = hidden_states
        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
                quant_format=getattr(self, "_gfx95_quant_format", ""),
            )
        )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
            llama_4_scaling=llama_4_scaling,
            layer_scatter_modes=self.layer_scatter_modes,
            prev_topk_indices=prev_topk_indices,
        )
        if isinstance(hidden_states, tuple):
            hidden_states, topk_indices = hidden_states
        else:
            topk_indices = None
        get_attn_tp_context().clear_attn_inputs()

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        if isinstance(self.mlp, DeepseekV2MLP):
            gemm_output_zero_allocator = None

        if (
            isinstance(self.mlp, DeepseekV2MoE)
            and not self.mlp.experts.moe_runner_config.inplace
            and not torch.compiler.is_compiling()
        ):
            from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx

            _mlp_ctx = moe_output_buffer_ctx(hidden_states_orig)
        else:
            _mlp_ctx = nullcontext()

        with _mlp_ctx:
            hidden_states = self.mlp(
                hidden_states,
                forward_batch,
                should_allreduce_fusion,
```

**Comment：** DSA 层返回 `(hidden, topk_indices)` 供下一层 `prev_topk_indices` 复用。

### 3.2 DSACPLayerCommunicator vs LayerCommunicator

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2136-L2160
        if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:
            # DSACPLayerCommunicator is flavor-agnostic; its internal gates
            # read both dsa_use_prefill_cp and mla_use_prefill_cp. The rename
            # to CPLayerCommunicator is deferred to a cleanup PR.
            self.layer_communicator = DSACPLayerCommunicator(
                layer_scatter_modes=self.layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=(
                    is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
                ),
                qkv_latent_func=self.self_attn.prepare_qkv_latent,
            )
        else:
            self.layer_communicator = LayerCommunicator(
                layer_scatter_modes=self.layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=(
                    is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
                ),
                qkv_latent_func=self.self_attn.prepare_qkv_latent,
            )
```

---

## 4. DeepseekV2Model / ForCausalLM

### 4.1 `DeepseekV2Model.forward` — TBO 包装

**Explain：** 启用 Two-Batch Overlap 时，`model_forward_maybe_tbo` 包装层循环，使 prefill/decode 或 dual batch 重叠。

**Code（概念入口，文件内调用）：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L37-L40（导入）
from sglang.srt.batch_overlap.two_batch_overlap import (
    MaybeTboDeepEPDispatcher,
    model_forward_maybe_tbo,
)
```

**Comment：** 具体 `DeepseekV2Model.forward` 体较长；读时搜索 `model_forward_maybe_tbo(self,` 定位。

### 4.2 `get_attn_tp_context().init_context`

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2723-L2724
        q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None
        get_attn_tp_context().init_context(q_lora_rank, is_deepseek_dsa(config))
```

**Comment：** 控制 QLoRA 路径下 input scatter 与 all-gather 行为。

### 4.3 Expert location（EPLB）

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2871-L2877
    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )
```

**Comment：** EPLB 运行时迁移 expert 权重；与 MoE dispatch 协同，非本模块主线。

### 4.4 Weight loader 入口

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2857-L2858
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        self.do_load_weights(weights, is_nextn)
```

**Comment：** `DeepseekV2WeightLoaderMixin.do_load_weights` 处理 expert 名 remap、shared expert fusion 槽位等。

---

## 5. DeepseekV2MLP（dense 层）

**Code：**

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L285-L294
    def forward(
        self,
        x,
        forward_batch=None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x
```

**Comment：** 空 batch 短路避免 EP/TP 下无效 collective。
