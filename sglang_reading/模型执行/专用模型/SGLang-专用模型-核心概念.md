---
title: "专用模型 · 核心概念"
type: concept
framework: sglang
topic: "专用模型"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-11
---
# 专用模型 · 核心概念

## 读者任务

读本篇不是为了背 `deepseek_v2.py` 的类名，而是为了在通用模型层之外建立一个更准确的判断框架：DeepSeek 专用模型把复杂度集中在 attention 形态、expert 形态、并行通信形态和权重形态四个地方。只要能把这四个地方拆开，线上排障时就不会把 MLA 回退、MoE expert 数、CP metadata、weight loader remap 混在一起。

读完后应能回答：

- DeepSeek V2/V3/V3.2 为什么共用 `DeepseekV2ForCausalLM` 实现。
- MLA、MHA、chunked KV、DSA/NPU 路径是谁选出来的。
- 哪些层是 dense MLP，哪些层是 sparse MoE。
- 独立 shared expert、普通 fusion、强制 DeepEP fusion 与 redundant experts 如何分别影响运行时 slot 数。
- 权重加载为什么比 Llama 多出 shared expert remap、fused qkv_a、indexer fusion 和 `kv_b_proj` 后处理。

## 四个改造点

| 改造点 | 通用模型里是什么 | DeepSeek 改成什么 | 排障入口 |
|--------|------------------|-------------------|----------|
| Attention | QKV projection 后交给 `RadixAttention` | `DeepseekV2AttentionMLA` 先选 MHA/MLA/DSA，再走 prepare/core | `dispatch_attn_forward_method` |
| MLP | 每层 dense MLP | 配置控制 dense 或 `DeepseekV2MoE` | `_is_layer_sparse` |
| 通信 | 普通 `LayerCommunicator` | DSA/MLA prefill CP 时换 `DSACPLayerCommunicator`，并传 `qkv_latent_func` | `DeepseekV2DecoderLayer.__init__` |
| 权重 | `load_weights` 做 QKV/gate-up mapping | `DeepseekV2WeightLoaderMixin` 处理 expert、NextN、fused qkv_a、indexer、post load | `do_load_weights` |

这四个点彼此独立。MLA 没走不一定是 MoE 问题；shared expert 没 fusion 不一定影响 `EntryClass`；权重名找不到也不应该先去看 Scheduler。

## 类账：一文件多版本，但 Registry 仍按类名命中

DeepSeek V3 和 V3.2 没有 fork 出两套完整模型类。它们继承同一个实现，由 `EntryClass` 暴露给 Registry：

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2937-L2942
class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    pass
```

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2966-L2966
EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]
```

所以 architecture 命中是类账问题：HF config 里的 `architectures` 决定 Registry 命中哪个类名；但 V3/V3.2 子类没有覆写构造或 forward。后续是否启用 DSA、shared expert fusion、CP，主要由传入 config 字段、architecture 字符串和 process-wide server args 决定，而不是 Python 子类中的专属实现。

## Attention 账：先选方法，再跑 prepare/core

DeepSeek attention 类用多个 mixin 组合出不同后端能力。读者不需要先钻进每个 kernel，而要先看它暴露了哪些方法枚举：

```python
# 来源：python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py L4-L32
class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    MHA = auto()

    # Use absorbed multi-latent attention
    MLA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()

    # Use multi-head attention, execute the MHA for prefix and extended kv in a single kernel
    # when the sequence lengths are below the threshold.
    MHA_ONE_SHOT = auto()

    # Use MLA but with fused RoPE
    MLA_FUSED_ROPE_ROCM = auto()

    # Use MLA with fused RoPE kernel for CPU
    MLA_FUSED_ROPE_CPU = auto()

    # Use multi-head attention for NPU
    MHA_NPU = auto()

    # Use absorbed multi-latent attention for NPU
    MLA_NPU = auto()

    # Use Deepseek V3.2 sparse multi-latent attention for NPU
    DSA_NPU = auto()
```

`DeepseekV2AttentionMLA` 构造 latent projection 和 MQA/MHA 两套 `RadixAttention` 适配器，然后在 forward 时按 method 走不同 prepare/core。method 不是 backend 名的同义词：同一个 handler 会根据图模式、CP、prefix 和 forward mode 改选方法；不同 handler 的规则也不相同。

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

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L1716-L1748
        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )
        # use num_local_heads * dcp_world_size because q_nope, q_rope is all gathered from dcp ranks
        if dcp_enabled():
            self.attn_mqa_for_dcp_decode = RadixAttention(
                self.num_local_heads * get_attention_dcp_world_size(),
                self.kv_lora_rank + self.qk_rope_head_dim,
                self.scaling,
                num_kv_heads=1,
                layer_id=layer_id,
                v_head_dim=self.kv_lora_rank,
                quant_config=quant_config,
                prefix=add_prefix("attn_mqa", prefix),
            )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )
```

这里的心理模型是：MLA 和 MHA 都还是 attention adapter，只是投影、KV cache 物理形态和 head/dim 元数据不同。prefix tree、调度 batch、采样输出都不在这个类里决定。

### handler registry 还有一个不显眼的 fallback

`AttentionBackendRegistry.get_handler()` 对未知 key 不抛异常，而是返回 `triton` handler。于是配置拼写错误可能表现为“服务能跑，但路径为何变成 Triton 规则”，而不是启动失败：

```python
# 来源：python/sglang/srt/models/deepseek_common/attention_backend_handler.py L20-L29
class AttentionBackendRegistry:
    _handlers = {}

    @classmethod
    def register(cls, backend_name, handler_func):
        cls._handlers[backend_name] = handler_func

    @classmethod
    def get_handler(cls, backend_name):
        return cls._handlers.get(backend_name, cls._handlers.get("triton"))
```

generic handler 也不能简化成“长 prefix 才 MHA”：纯 extend 且 prefix 总长为 0 时同样进入 MHA one-shot/chunked 分支；TC piecewise graph 与 MLA prefill CP 又会优先强制 MLA。FA4、Aiter、TRT-LLM、DSA、Ascend 和 Triton 还有各自的规则。因此诊断必须记录“backend key + handler + method”，不能只记 method。

## Expert 账：sparse 层不是每层都有

DeepSeek decoder 层先判断当前层是否 sparse。满足配置条件才构造 `DeepseekV2MoE`，否则仍是 dense `DeepseekV2MLP`：

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2090-L2127
        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

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

`DeepseekV2MoE` 的第一层判断是 expert 数和 shared expert 形态。若 DeepEP fusion 已被显式允许，它会把 shared expert 表现成 EP-size 个额外 slot；普通 fusion 则把 routed expert 数加上 fused shared expert 数：

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L548-L587
        n_shared_experts = (
            0 if config.n_shared_experts is None else int(config.n_shared_experts)
        )
        _fusion_disabled = get_global_server_args().disable_shared_experts_fusion

        # num_fused_shared_experts drives weight remapping in deepseek_weight_loader:
        # mlp.shared_experts → mlp.experts.256 when > 0.
        self.num_fused_shared_experts = 0 if _fusion_disabled else n_shared_experts

        # DeepEP shared expert fusion: shared expert is fused into the same MoE kernel
        # as a local expert at the home EP rank. Expert layout is expanded from 256
        # routed to 256+EP_size (e.g. 272 for EP=16). TopK handles interleaving.
        _is_deepep_fusion = (
            is_deepep_class_backend() and self.num_fused_shared_experts > 0
        )

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

        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.is_nextn = is_nextn

        n_hash_layers = getattr(config, "num_hash_layers", 0)
        self.is_hash = layer_id < n_hash_layers and not (is_deepseek_v4 and is_nextn)

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )
```

因此看到 `num_experts` 多出来，不要马上判断 checkpoint 错。运行时 `experts.num_experts` 还可能叠加 `ep_num_redundant_experts`。同时要注意：`determine_num_fused_shared_experts` 在 DeepEP 下默认关闭 fusion；只有显式强制等情况下，后续构造才会进入 `n_routed + moe_ep_size` 的 DeepEP fused slot 形态。

shared-expert fusion 的禁用不是局部布尔值。某个实例命中安全门禁后，会把全局 `server_args.disable_shared_experts_fusion` 改成 `True`，后续 `DeepseekV2MoE` 都读取这个 process-wide 状态。反过来，`enforce_shared_experts_fusion` 会绕过 architecture、硬件、量化、SBO/TBO 和 DeepEP 默认禁用检查；它是承担正确性风险的强制开关，不是普通性能开关。

## 通信账：LayerCommunicator 是 attention 与 MLP 之间的关节

Decoder layer 不直接手写 scatter、allreduce、reduce-scatter 的所有细节。它根据 CP 开关选择 communicator，并把 `self_attn.prepare_qkv_latent` 传进去：

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

这解释了一个常见误区：DeepSeek 层内顺序看起来仍是 norm、attention、norm、MLP，但真正影响 layout 的判断已经交给 communicator 和 `LayerScatterModes`。CP communicator 在实际 gather/reduce-scatter helper 中还断言 `attn_dp_size == 1` 且 `attn_tp_size == 1`；启用 CP 不等于任意 DP/TP 拓扑都可组合。

## 权重账：专用模型的 load_weights 是翻译层

DeepSeek 顶层 `load_weights` 保持和通用模型一样的入口，但实际工作交给 mixin：

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2649-L2652
class DeepseekV2ForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin):
    # for quark model load
    packed_modules_mapping = {}
```

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2857-L2858
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        self.do_load_weights(weights, is_nextn)
```

权重账的核心不是“找到文件”，而是把 checkpoint 的名字和运行时参数形态对齐：PP stage 外权重要跳过，shared expert 可能 remap 到 `mlp.experts.<n_routed_experts>`，`q_a_proj` 和 `kv_a_proj_with_mqa` 可能成对合成 `fused_qkv_a_proj_with_mqa`，DSA indexer 的 `wk` 与 `weights_proj` 也可能写入 fused 参数。

加载器对 CPU tensor 可把 parameter write 提交到线程池，但会在离开线程池前对所有 future 调 `result()`，然后才执行 `post_load_weights`。所以 post-load 不会与已提交写入竞态；真正需要警惕的是“成对缓存不完整”：`cached_a_proj` 与 FP8 `pending_indexer_wk` 在循环结束时没有显式非空断言。若 checkpoint 只提供一半 pair，可能没有清晰的缺权重异常，却留下未完成的 fused 参数。

`post_load_weights` 也不只是 reshape。它按量化格式完成 AWQ dequant、FP8/INT8 转换或 requant，再拆出 `w_kc/w_vc`；部分热更新时只处理本次 `weight_names` 中出现 `kv_b_proj` 的 layer。

## 复盘

本专题的第一原则是：DeepSeek 专用逻辑仍然服务于通用的 CausalLM 接口。外部看仍是 `input_ids/positions/ForwardBatch → model.forward → logits 或 PPProxyTensors`，内部才分成 MLA、MoE、CP、专用 weight loader。排障时先判断落在哪本账，再进对应源码，不要从 3000 行模型文件头读到尾。
