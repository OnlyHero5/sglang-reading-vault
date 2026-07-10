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
updated: 2026-07-10
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
| Qwen3 decode KV cache 异常 | fused mRoPE 与 `save_kv_cache` | fused 路径已写 cache，后续不能重复写 |
| `Parameter <name> not found in params_dict` | 模型类 `load_weights` | 前缀、stage skip、stacked mapping 是否命中 |
| split prefill 结果断裂 | `forward_split_prefill` | `forward_batch.hidden_states/residual` 是否跨 split 保留 |

## Q1：为什么常见模型应该优先走 native？

Native 模型类直接集成 fused QKV/gate-up 参数、`RadixAttention`、paged KV cache、PP/TP 和 CUDA graph 边界。Transformers fallback 是兼容路径，不是吞吐优先路径。

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

验证方法：启动后检查 `_resolved_model_arch` 和 `_resolved_model_impl`。如果 Qwen3/Llama 落到 `TransformersForCausalLM`，先查 architecture 字符串是否命中 Registry。

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

## Q3：为什么 Registry 报错分两类？

Registry 区分两类失败：候选里有支持项但检查失败，或候选完全不支持。前者看 import/inspect 日志，后者看 architecture 字符串。

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

验证方法：如果报 “failed to be inspected”，先查前面的 import warning；如果报 “are not supported”，先查 `EntryClass` 和 fallback 分支。

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

判断方法：Qwen3 attention shape 错时，不要只看全局 `tp_size`。先看 attention TP 组是否和你预期一致。

## Q5：Qwen3 的 fused mRoPE 为什么要关掉后续 KV 写入？

fused mRoPE 路径已经把 KV 写入 paged cache。源码注释和返回值都明确提示：返回 `(q, None, None)`，调用方必须传 `save_kv_cache=False`。

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

验证方法：decode cache 异常时，打印或断点 `use_aiter_fused` 和 `save_kv_cache`。如果 fused 为真但 `save_kv_cache` 没关，边界就错了。

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

`forward_split_prefill` 把中间 hidden 和 residual 存在 `forward_batch` 上。每次 split 只执行一段 layer，最后一段才做 norm 和 logits。

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

验证方法：chunked prefill 异常时，检查 split 间 `forward_batch.hidden_states` 和 `forward_batch.residual` 是否被保留，而不是只看最后一次 forward。

## 小结

排查 Models 通用层时按这个顺序问：

1. 类是否 resolve 到预期 native 实现。
2. 当前 PP rank 是否应该产出 logits。
3. Attention 使用的是哪个 prepare 路径和哪个 TP 组。
4. 权重名是否被模型类 remap 到真实参数。
5. 状态是否跨 PP stage 或 split prefill 正确传递。
