---
title: "通用模型 · 排障指南"
type: troubleshooting
framework: sglang
topic: "通用模型"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-11
---
# 通用模型 · 排障指南

本篇按症状排障。先判断问题属于类账、执行账还是权重账，再进入对应源码。

## 症状速查

| 现象 | 优先看 | 判断方式 |
|------|--------|----------|
| native 模型没命中 | `get_model_architecture`、`ModelRegistry` | `architectures` 是否注册，是否被强制 Transformers |
| `Model architectures <archs> are not supported` | Registry 支持列表 | architecture 字符串与 `EntryClass.__name__` 是否一致 |
| PP 中间 rank 没有 logits | `pp_group.is_last_rank` | 非 last rank 只返回 hidden 或 `PPProxyTensors` |
| Qwen3 attention head shape 错 | `attn_tp_rank/attn_tp_size` | attention TP 与普通 TP 是否混淆 |
| Qwen3 decode KV cache 异常 | fused 能力门禁、mRoPE 与 `save_kv_cache` | 先证明实际进入 ROCm/Aiter/MRoPE fused 路径；该路径已写 cache，后续不能重复写 |
| `Parameter <name> not found in params_dict` | 模型类 `load_weights` | 前缀、stage skip、stacked mapping 是否命中 |
| split prefill 结果断裂 | `forward_split_prefill` | `ForwardBatch` 中间态是否跨 split 保留，interval 是否落在本 rank 的真实层区间 |
| PP + tied embedding logits 异常 | 构造期 `lm_head` + 加载期 embedding 补写 | last rank 是对象 alias 还是独立输出头，权重是否真正写入 |

## Q1：为什么常见模型应该优先走 native？

Native 模型类直接集成 fused QKV/gate-up 参数、`RadixAttention`、paged KV cache、PP/TP 和图执行边界。Transformers fallback 的源码 warning 明确说“部分能力可能不支持、性能可能非最优”，但这不是无硬件与 workload 前提的性能判决；排障时应先确认功能兼容与实际 resolved wrapper。

源码上的分界在 `get_model_architecture`：如果没有 native 支持，或用户显式指定 Transformers，才调用 `resolve_transformers_arch`。

```python
# 来源：python/sglang/srt/model_loader/utils.py L216-L230
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

验证方法：启动后检查 `_resolved_model_arch` 和 `_resolved_model_impl`。如果 Qwen3/Llama 落到 Transformers wrapper，先查 architecture key 是否注册；再看 wrapper 是否因 pooling、多模态或 MoE 被选择成别的 `Transformers*` 类，而不是固定假设只有 `TransformersForCausalLM`。

## Q2：为什么 `EntryClass` 必须和 HF architecture 对齐？

Registry 的 key 来自 `EntryClass.__name__`。这意味着 HF `config.json` 里的 `architectures` 字符串要能匹配 Python 类名。

```python
# 来源：python/sglang/srt/models/registry.py L111-L125
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
```

一个文件可以暴露多个类，但它们是多个 Registry key：

```python
# 来源：python/sglang/srt/models/llama.py L851-L856
EntryClass = [
    LlamaForCausalLM,
    Phi3ForCausalLM,
    InternLM3ForCausalLM,
    IQuestCoderForCausalLM,
]
```

验证方法：拿 `config.json` 的 `architectures[0]` 去搜模型文件里的 `EntryClass`。如果类名不一致，Registry 不会因为文件名相似而自动命中。

## Q3：为什么不能把 “failed to be inspected” 当成当前 Registry 的主要分支？

错误函数仍保留“候选 key 存在但类取值失败”和“候选完全不支持”两种文案；但当前 `_try_load_model_cls` 只做字典取值，不再执行动态 inspect。正常注册表的 value 来自已成功导入的 `EntryClass`，所以实战里更常见的根因发生在更早的模块扫描阶段：import warning 导致 key 根本没注册。

```python
# 来源：python/sglang/srt/models/registry.py L55-L59
    def _try_load_model_cls(self, model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch not in self.models:
            return None

        return self.models[model_arch]
```

```python
# 来源：python/sglang/srt/models/registry.py L41-L53
    def _raise_for_unsupported(self, architectures: List[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed "
                "to be inspected. Please check the logs for more details."
            )

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )
```

验证方法：无论最终报哪条文案，都先向前找 `Ignore import error when loading ...`。若存在，修复模块依赖/导入异常；若不存在，再打印 `ModelRegistry.get_supported_archs()`、候选列表和外部包覆盖结果。不要声称 resolve 阶段会再次 inspect 类。

## Q4：Llama 和 Qwen3 的 TP 切分差异在哪里？

Llama attention 使用全局 TP size：

```python
# 来源：python/sglang/srt/models/llama.py L146-L159
        tp_size = get_parallel().tp_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

Qwen3 attention 使用 attention TP size：

```python
# 来源：python/sglang/srt/models/qwen3.py L87-L101
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
```

判断方法：这两张卡只证明 head partition 的分母不同。Qwen3 attention shape 错时，不要只看全局 `tp_size`；继续检查 QKV/O projection 的显式 attention TP 参数、O projection 的 `reduce_results=False`，以及 `LayerCommunicator` 如何把布局接回 MLP。

## Q5：Qwen3 的 fused mRoPE 何时存在，为什么要关掉后续 KV 写入？

这不是所有 Qwen3 decode 的公共路径。构造时必须同时满足：HIP/ROCm、`SGLANG_USE_AITER` 为真、Aiter kernel 导入成功、rotary 实例是 `MRotaryEmbedding` 且具有 `mrope_section`；运行时还必须是 decode 且非 RL on-policy。只有这条 fused 路径会提前把 KV 写入 paged cache，返回 `(q, None, None)`，调用方才必须传 `save_kv_cache=False`。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/models/qwen3.py L43-L56
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

_has_fused_qk_norm_mrope = False
if _use_aiter:
    try:
        from aiter import fused_qk_norm_mrope_3d_cache_pts_quant_shuffle

        _has_fused_qk_norm_mrope = True
    except ImportError:
        pass
```

```python
# 来源：python/sglang/srt/models/qwen3.py L160-L164
        self.use_fused_qk_norm_mrope = (
            _has_fused_qk_norm_mrope
            and isinstance(self.rotary_emb, MRotaryEmbedding)
            and getattr(self.rotary_emb, "mrope_section", None) is not None
        )
```

```python
# 来源：python/sglang/srt/models/qwen3.py L207-L215
    def forward_prepare_aiter_fused_mrope(
        self, positions, hidden_states, forward_batch
    ):
        """Fused QK-norm + 3D mRoPE + KV cache write for decode (ROCm/aiter).

        The fused HIP kernel replaces split → QK norm → mRoPE → cache write,
        so KV is already in the paged cache when this returns.
        Returns (q, None, None); caller must pass save_kv_cache=False to attn.
        """
```

调用方在 fused 分支后把标志关掉：

```python
# 来源：python/sglang/srt/models/qwen3.py L278-L306
        save_kv_cache = True
        use_aiter_fused = (
            self.use_fused_qk_norm_mrope
            and forward_batch.forward_mode.is_decode()
            and get_global_server_args().rl_on_policy_target is None
        )

        if use_aiter_fused:
            q, k, v = self.forward_prepare_aiter_fused_mrope(
                positions, hidden_states, forward_batch
            )
            save_kv_cache = False
        elif not _is_npu:
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

        if get_global_server_args().rl_on_policy_target is not None:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=save_kv_cache)
```

验证方法：先记录 `_is_hip`、`SGLANG_USE_AITER`、`_has_fused_qk_norm_mrope`、`type(rotary_emb)`、`mrope_section` 与 forward mode，再断点 `use_aiter_fused/save_kv_cache`。若构造期门禁未满足，就应回到 native/NPU 路径排查，不能套用 fused-cache 结论。

## Q6：为什么有些 checkpoint tensor 会被跳过？

模型类会跳过 stage 外层、冗余 RoPE cache、tie embedding 下的 `lm_head.weight`、旧 KV scale 等。以 Llama 为例：

```python
# 来源：python/sglang/srt/models/llama.py L647-L671
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
```

判断方法：看到 tensor 没写入前，先区分它是被有意 skip，还是名字 remap 后仍找不到参数。

## Q7：为什么 `q_proj.weight` 不在参数表里？

因为运行时常使用 fused `qkv_proj`。`load_weights` 会把 `q_proj/k_proj/v_proj` 映射到 `qkv_proj`，再调用参数自己的 `weight_loader`。

```python
# 来源：python/sglang/srt/models/llama.py L673-L685
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

验证方法：把原始 checkpoint name 手动套 mapping。若 mapping 后参数存在，再继续看参数 `weight_loader` 的 shape 和 shard_id。

## Q8：split prefill 的状态放在哪里？

`forward_split_prefill` 把中间 hidden 和 residual 存在同一个可变 `ForwardBatch` 上。每次调用直接按传入的全局 `[start, end)` 索引 `self.model.layers`，最后一段才做 norm 和 logits；函数本身不接收 `PPProxyTensors`，也没有把 interval 裁到当前 PP stage 的保护。

```python
# 来源：python/sglang/srt/models/llama.py L565-L603
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ) -> Optional[LogitsProcessorOutput]:
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result
```

验证方法：chunked prefill 异常时，除检查 split 间 `forward_batch.hidden_states/residual` 是否保留，还要核对调用方生成的 interval 是否只命中当前 rank 的真实 layer。若 PP>1 却索引到 `PPMissingLayer`，问题在调用契约而不在最后一次 norm。

## Q9：为什么 PP 下 `tie_word_embeddings=True` 仍可能有独立 `lm_head`？

Qwen2/Qwen3 只有在 PP world size 为 1 时才把 `lm_head` 直接指向 embedding。PP>1 时 embedding 在 first rank，输出头在 last rank，不可能共享同一个模块对象；last rank 会创建 `ParallelLMHead`，随后 `load_weights` 在读到 `model.embed_tokens.weight` 时额外写入 `lm_head.weight`。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/models/qwen3.py L487-L501
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()
```

```python
# 来源：python/sglang/srt/models/qwen3.py L616-L623
            if name == "model.embed_tokens.weight":
                if self.pp_group.is_last_rank and self.config.tie_word_embeddings:
                    if "lm_head.weight" in params_dict:
                        param = params_dict["lm_head.weight"]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
```

判断方法：同时检查构造期 `pp_group.world_size/is_last_rank` 与加载期补写是否执行。只看到 config 的 `tie_word_embeddings=True`，不能证明运行时两个参数对象已经共享或数值已经一致。

## 小结

排查 Models 通用层时按这个顺序问：

1. 类是否 resolve 到预期 native 实现。
2. 当前 PP rank 是否应该产出 logits。
3. Attention 使用的是哪个 prepare 路径和哪个 TP 组。
4. 权重名是否被模型类 remap 到真实参数。
5. 状态是否跨 PP stage 或 split prefill 正确传递，split interval 是否命中真实层。
6. tied embedding 是对象 alias，还是由加载期双写维持数值一致。
