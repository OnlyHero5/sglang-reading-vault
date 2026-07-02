---
type: batch-doc
module: 13-Models-通用
batch: "13"
doc_type: walkthrough
title: "Models 通用 · 源码走读"
tags:
 - sglang/batch/13
 - sglang/module/models-common
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# Models 通用 · 源码走读

> 走读顺序：`registry.py` → `model_loader/utils.py`（resolve 调用点）→ `llama.py`（Attention / Model / ForCausalLM）→ `qwen3.py`（QK-Norm / Communicator）

---

## 1. registry.py

### 1.1 `import_model_classes` — 扫描 EntryClass

**Explain：** 遍历 `sglang.srt.models` 包下所有非 package 模块，import 后读取 `EntryClass`。list 形式支持一文件多 architecture；单类直接注册。

**Code：**

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

**Comment：** 重复类名 assert 失败，避免两个文件 export 同名 arch。

### 1.2 `_raise_for_unsupported` — 错误信息

**Code：**

```python
# 来源：python/sglang/srt/models/registry.py L50-L53
        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )
```

**Comment：** 启动加载失败时此列表极长（100+ arch）；排查时先确认 HF config 的 `architectures[0]` 是否在 Registry。

---

## 2. llama.py — Attention 栈

### 2.1 `LlamaAttention.__init__` — TP 切分 KV head

**Explain：** 根据 `tp_size` 与 `num_kv_heads` 关系决定 partition 或 replicate KV heads。`QKVParallelLinear` 一次投影 Q/K/V；`get_rope` 构造 RoPE；`RadixAttention` 接管 attention 计算。

**Code：**

```python
# 来源：python/sglang/srt/models/llama.py L146-L187
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
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
```

**Comment：** GQA/MQA 场景 `num_kv_heads < num_heads` 是常态；TP 大于 KV head 数时会 replicate。

### 2.2 `LlamaAttention.forward` — QKV + RoPE + RadixAttention

**Code：**

```python
# 来源：python/sglang/srt/models/llama.py L228-L252
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

**Comment：** NPU decode 可走 fused `split_qkv_rmsnorm_rope`；CUDA 默认 native 路径。

### 2.3 `LlamaModel.forward` — PP 与 aux hidden

**Code：**

```python
# 来源：python/sglang/srt/models/llama.py L385-L431
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            # FIXME(@ying): reduce the number of proxy tensors by not fusing layer norms
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            deferred_norm = None

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states + residual)
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states
```

**Comment：** EAGLE3 等 speculative 路径会设置 `layers_to_capture` 收集 aux hidden。

### 2.4 `LlamaForCausalLM` — lm_head 与 tie embeddings

**Code：**

```python
# 来源：python/sglang/srt/models/llama.py L497-L507
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)
```

**Comment：** `enable_dp_lm_head` 时 lm_head 走 attention TP group，与 DP 布局一致。

---

## 3. qwen3.py — Qwen3 特化

### 3.1 `Qwen3Attention` — attn_tp 与 QK norm

**Code：**

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

**Comment：** 与 Llama 用 `tp_size` 不同；DP-Attention 模式下 attention 与 MLP 的 TP 组可分离。

### 3.2 `Qwen3Attention.forward` — aiter fused decode 路径

**Explain：** decode 且启用 aiter fused mRoPE 时，kernel 直接把 K/V 写入 paged cache，返回 `(q, None, None)`，须设 `save_kv_cache=False` 避免重复写 cache。

**Code：**

```python
# 来源：python/sglang/srt/models/qwen3.py L269-L307
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if get_global_server_args().rl_on_policy_target is not None:
            hidden_states = hidden_states.bfloat16()

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
        output, _ = self.o_proj(attn_output)
```

**Comment：** RL on-policy 训练强制 bf16 路径，禁用 fused kernel。

### 3.3 `Qwen3DecoderLayer` — LayerCommunicator

**Code：**

```python
# 来源：python/sglang/srt/models/qwen3.py L376-L387
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )
```

**Comment：** `LayerScatterModes` 描述 attention 输入/输出是 replicated 还是 scattered；MoE 层在Models 专用 展开。

### 3.4 `Qwen3ForCausalLM.forward` — 与 Llama 对齐

**Code：**

```python
# 来源：python/sglang/srt/models/qwen3.py L512-L546
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
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

**Comment：** 返回值类型注解为 `Tensor`，实际 last rank 返回 `LogitsProcessorOutput` dataclass。

---

## 4. load_weights 走读（Llama 代表）

### 4.1 PP 层过滤与 stacked mapping

**Code：**

```python
# 来源：python/sglang/srt/models/llama.py L647-L656
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
```

**Comment：** 多 PP rank 各加载自己 stage 的层；rank0 不会加载 layer 24+ 等于另一 stage 的参数。

### 4.2 默认 weight_loader fallback

**Code：**

```python
# 来源：python/sglang/srt/models/llama.py L686-L698
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip loading kv_scale from ckpts towards new design.
                if name.endswith(".kv_scale") and name not in params_dict:
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
```

**Comment：** 量化 checkpoint 的 `.weight_scale` / `.activation_scale` 在前面的 loop 开头已 remap 名称。
