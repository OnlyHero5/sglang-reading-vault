---
title: "通用模型 · 核心概念"
type: concept
framework: sglang
topic: "通用模型"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-11
---
# 通用模型 · 核心概念

Models 通用层不是“模型文件清单”。它承接两条上游输入：`ModelConfig.hf_config.architectures` 和 ModelLoader 产出的 `(name, tensor)`；同时向下游交付一个可被 ModelRunner 调用的 `nn.Module`。

## 读者任务

读本篇是为了建立三个判断：

1. architecture 字符串如何变成 Python 模型类。
2. `input_ids/positions/ForwardBatch` 如何穿过通用 decoder 骨架。
3. checkpoint 名字如何在 `load_weights` 中映射到 fused 参数。

## 三本账

| 账本 | 问题 | 关键对象 | 常见失败 |
|------|------|----------|----------|
| 类账 | 这次启动该实例化哪个模型类 | `get_model_architecture`、`ModelRegistry`、`EntryClass` | architecture 不支持、fallback 到 Transformers |
| 执行账 | 一个 batch 如何得到 logits 或 PP 中间态 | `*Model.forward`、`*DecoderLayer.forward`、`RadixAttention` | PP 边界错、attention TP 错、KV cache 写入错 |
| 权重账 | 一个 checkpoint name 如何写入参数 | `load_weights`、`stacked_params_mapping`、`param.weight_loader` | 参数名不匹配、shape mismatch、stage 外权重误写 |

把这三本账分开后，很多问题会变清楚：Registry 不管 QKV 怎么 fused；`forward` 不管 checkpoint 文件怎么找；`load_weights` 不决定 attention backend。

## 类账：Registry 只回答“谁来实现这个 architecture”

Registry 的表项来自模型模块的 `EntryClass`。导入 `sglang.srt.models.registry` 时，`pkgutil.iter_modules` 会扫描内置模型包并逐模块 import；默认 `strict=False`，某个模块依赖缺失或导入报错时只记 warning，然后整个模块的 `EntryClass` 都不会进入表。若设置外部模型包，则再以 `overwrite=True` 覆盖同名 architecture key。

```python
# 来源：python/sglang/srt/models/registry.py L94-L134
@lru_cache()
def import_model_classes(package_name: str, strict: bool = False):
    model_arch_name_to_cls = {}
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            if name.split(".")[-1] in envs.SGLANG_DISABLED_MODEL_ARCHS.get():
                logger.debug(f"Skip loading {name} due to SGLANG_DISABLED_MODEL_ARCHS")
                continue

            try:
                module = importlib.import_module(name)
            except Exception as e:
                if strict:
                    raise
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(
                    entry, list
                ):  # To support multiple model classes in one module
                    for tmp in entry:
                        assert (
                            tmp.__name__ not in model_arch_name_to_cls
                        ), f"Duplicated model implementation for {tmp.__name__}"
                        model_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    assert (
                        entry.__name__ not in model_arch_name_to_cls
                    ), f"Duplicated model implementation for {entry.__name__}"
                    model_arch_name_to_cls[entry.__name__] = entry

    return model_arch_name_to_cls


ModelRegistry = _ModelRegistry()
ModelRegistry.register("sglang.srt.models")

if external_pkg := envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():
    ModelRegistry.register(external_pkg, overwrite=True)
```

这里的 key 是类名，例如 `Qwen3ForCausalLM`。所以接新模型时，最小契约不是“文件名叫 qwen3”，而是模块能成功导入，且 HF `architectures` 里的字符串能命中 `EntryClass.__name__`。`SGLANG_DISABLED_MODEL_ARCHS` 检查的也是模块 basename，不是 architecture 类名。

## 类账到实例：ModelLoader 调 get_model_architecture

`get_model_architecture` 是 ModelLoader 初始化空模型前的入口。它先处理特殊实现，再把 architecture 交给 Registry。

```python
# 来源：python/sglang/srt/model_loader/utils.py L195-L230
def get_model_architecture(model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    from sglang.srt.models.registry import ModelRegistry

    architectures = getattr(model_config.hf_config, "architectures", [])
    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    mixtral_supported = [
        "fp8",
        "compressed-tensors",
        "gptq_marlin",
        "awq_marlin",
        "quark_int4fp8_moe",
    ]

    if (
        model_config.quantization is not None
        and model_config.quantization not in mixtral_supported
        and "MixtralForCausalLM" in architectures
    ):
        architectures = ["QuantMixtralForCausalLM"]

    supported_archs = ModelRegistry.get_supported_archs()
    is_native_supported = any(arch in supported_archs for arch in architectures)

    if model_config.model_impl == ModelImpl.MINDSPORE:
        architectures = ["MindSporeForCausalLM"]
    elif not is_native_supported or model_config.model_impl == ModelImpl.TRANSFORMERS:
        architectures = resolve_transformers_arch(model_config, architectures)
    model_cls, resolved_arch = ModelRegistry.resolve_model_cls(architectures)
    setattr(model_config, "_resolved_model_arch", resolved_arch)
    setattr(
        model_config,
        "_resolved_model_impl",
        _model_impl_from_architecture(resolved_arch),
    )
    return model_cls, resolved_arch
```

这解释了 native 与 Transformers fallback 的边界：只要候选中存在 native key，`AUTO` 就保留原候选；没有 native key 或用户显式指定 `TRANSFORMERS` 时，`resolve_transformers_arch` 还会根据 generation/pooling、多模态与 MoE 选择具体 wrapper，并检查 Transformers 类或 remote `auto_map` 的 backend compatibility。显式强制 Transformers 会把不兼容从硬错误降成 warning，因此 `_resolved_model_arch/_impl` 是功能排查入口，不是单纯的性能标签。

## 执行账：通用 decoder 是 hidden state 生产线

以 Qwen 系为例，`Qwen3Model` 复用 `Qwen2Model` 骨架，只替换 decoder layer 类型。模型骨架负责 embedding、PP 切层、层循环和 final norm。

```python
# 来源：python/sglang/srt/models/qwen2.py L267-L318
class Qwen2Model(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
                params_dtype=(
                    torch.float32
                    if get_global_server_args().rl_on_policy_target is not None
                    else None
                ),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
        pp_start_layer, _ = get_pp_indices(
            config.num_hidden_layers,
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                start_layer=pp_start_layer,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
```

`make_layers` 会为不属于当前 PP rank 的 layer 放 `PPMissingLayer`，同时返回本 rank 的 `[start_layer, end_layer)`：

```python
# 来源：python/sglang/srt/utils/common.py L731-L757
    assert not pp_size or num_hidden_layers >= pp_size
    start_layer, end_layer = (
        get_pp_indices(
            num_hidden_layers,
            pp_rank,
            pp_size,
        )
        if pp_rank is not None and pp_size is not None
        else (0, num_hidden_layers)
    )
    modules = torch.nn.ModuleList(
        [PPMissingLayer(return_tuple=return_tuple) for _ in range(start_layer)]
        + get_offloader().wrap_modules(
            (
                layer_fn(idx=idx, prefix=add_prefix(idx, prefix))
                for idx in range(start_layer, end_layer)
            ),
            **(offloader_kwargs or {}),
        )
        + [
            PPMissingLayer(return_tuple=return_tuple)
            for _ in range(end_layer, num_hidden_layers)
        ]
    )
    if pp_rank is None or pp_size is None:
        return modules
    return modules, start_layer, end_layer
```

因此，PP rank 不持有所有真实 decoder layer。它保持完整 `ModuleList` 下标空间，但 stage 外位置是 `PPMissingLayer`；stage 内模块还会经过 offloader 的 `wrap_modules`，所以“被当前 rank 实例化”不等于“始终常驻设备”。

## 执行账：CausalLM wrapper 只负责最后一跳

`*ForCausalLM.forward` 统一接收 `input_ids`、`positions`、`forward_batch`。底层 model 产出 hidden states 后，只有最后一个 PP rank 才做 logits 或 pooling。

```python
# 来源：python/sglang/srt/models/llama.py L528-L562
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states
```

如果你在中间 PP rank 查不到 logits，这是正常边界，不是模型漏了 lm_head。

## Attention 不是直接调用 FlashAttention

模型 attention 子模块只准备 Q/K/V、RoPE 和模型特有的 norm，然后把张量交给 `RadixAttention`。

```python
# 来源：python/sglang/srt/models/llama.py L207-L252
    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn.layer_id == self.start_layer:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            is_neox_style=self.rotary_emb.is_neox_style,
        )
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if (
            not _is_npu
            or not hasattr(self.rotary_emb, "get_cos_sin_with_position")
            or forward_batch.forward_mode.is_extend()
        ):
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            q, k, v = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output
```

真正的 backend 选择在 [[SGLang-Attention]]。本专题只需要记住：模型层把 Q/K/V 和 `ForwardBatch` 交给 attention adapter。

## Qwen3 的特化点：attention TP 与模型级通信要成对理解

Qwen3 和 Llama 的骨架相近，但 Qwen3 的 QKV/O projection 显式使用 `attn_tp_rank/attn_tp_size`，其中 O projection 还设置 `reduce_results=False`；复用的 `Qwen2MLP` 则没有显式传这组参数。两条布局如何重新接合，由 decoder layer 的 `LayerCommunicator` 负责，不能只用“Qwen3 改成 attention TP”一句话覆盖整个 block。

```python
# 来源：python/sglang/srt/models/qwen3.py L85-L119
        self.tp_size = get_parallel().tp_size
        self.total_num_heads = num_heads
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_parallel().tp_rank

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
```

```python
# 来源：python/sglang/srt/models/qwen3.py L132-L141
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )
```

`apply_qk_norm` 必须发生在 RoPE 前；这是数值语义，不是可任意移动的融合优化：

```python
# 来源：python/sglang/srt/models/qwen3.py L173-L185
    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v
```

## 权重账：load_weights 是 checkpoint 命名和内部参数布局的翻译层

Llama 类模型把 HF 常见的 `q_proj/k_proj/v_proj` 合并到 `qkv_proj`，把 `gate_proj/up_proj` 合并到 `gate_up_proj`。

```python
# 来源：python/sglang/srt/models/llama.py L629-L645
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if name.endswith(".activation_scale"):
                name = name.replace(".activation_scale", ".input_scale")
            if name.endswith(".weight_scale_inv"):
                name = name.replace(".weight_scale_inv", ".weight_scale")
```

stage 外的 layer 权重会被跳过，命中 stacked mapping 的权重会调用参数自己的 `weight_loader`：

```python
# 来源：python/sglang/srt/models/llama.py L647-L685
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            # Handle FP8 kv-scale remapping
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
```

因此，ModelLoader 只把 `(name, tensor)` 送进来；模型类决定名字如何翻译；参数对象决定如何按当前线性层的并行组与 shard 写入。不要把它固定说成“全局 TP rank”：Qwen3 attention 的线性层显式使用 attention TP，而普通 MLP 仍走其自身默认并行上下文。

## tied embedding 不是一句“共享权重”就能讲完

单卡/无 PP 时，Qwen2/Qwen3 在 `tie_word_embeddings=True` 时可直接令 `lm_head = embed_tokens`。PP world size 大于 1 时，first rank 持有 embedding、last rank 持有输出头，二者无法是同一个 Python 模块；源码会在 last rank 创建独立 `ParallelLMHead`，再由 `load_weights` 遇到 `model.embed_tokens.weight` 时把同一 checkpoint tensor 补写到 `lm_head.weight`。

这一区分很实用：

- “参数语义 tied”不必等于“运行时对象 alias”。
- PP 下检查 tied embedding，既要看构造期模块，也要看加载期是否发生补写。
- Llama、Qwen2、Qwen3 的构造细节并不完全一致，不能从一个模型类外推全库。

## 复盘

读通用模型层时，按这三个问题走：

- 这次启动最终 resolve 到哪个 `*ForCausalLM` 类？
- 这个类的 forward 是否保持 `input_ids/positions/ForwardBatch` 的统一入口？
- 这个类的 `load_weights` 是否把 checkpoint 名字翻译成内部参数名？
