---
type: batch-doc
module: 13-Models-通用
batch: "13"
doc_type: faq
title: "Models 通用：关键问题"
tags:
 - sglang/batch/13
 - sglang/module/models-common
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# Models 通用：关键问题

> 常见问题、易错点与设计取舍。

---

## Q1：Registry 和 HuggingFace `AutoModel` 什么关系？

**Explain：** SGLang **优先 native 实现**（`llama.py`、`qwen3.py` 等），性能与 KV cache 集成更好。仅当 arch 不在 Registry 或用户指定 `--model-impl transformers` 时，才 fallback 到 `TransformersForCausalLM`（仍通过 Registry 注册）。

**Code：**

```python
# 来源：python/sglang/srt/model_loader/utils.py L221-L222
    elif not is_native_supported or model_config.model_impl == ModelImpl.TRANSFORMERS:
        architectures = resolve_transformers_arch(model_config, architectures)
```

**Comment：** 生产环境 Qwen3/Llama 应确认 log 里 `_resolved_model_arch` 是 native 类名而非 Transformers wrapper。

---

## Q2：为什么 EntryClass 用类名而不是字符串？

**Explain：** `import_model_classes` 在 import 时绑定真实 `Type[nn.Module]`，ModelRunner 直接 `model_cls(config)`。字符串 indirection 会延迟 import 错误到首次 forward。

**Code：**

```python
# 来源：python/sglang/srt/models/registry.py L120
                        model_arch_name_to_cls[tmp.__name__] = tmp
```

**Comment：** 键 = `tmp.__name__`，必须与 HF JSON 完全一致。

---

## Q3：Llama 与 Qwen3 的 TP 切分有何不同？

**Explain：** Llama Attention 用全局 `get_parallel().tp_size`。Qwen3 用 `attn_tp_rank` / `attn_tp_size`（attention 专用 TP 组），支持 **DP Attention**（MLP 与 Attention 不同并行组）。

**Code（Llama）：**

```python
# 来源：python/sglang/srt/models/llama.py L146-L149
        tp_size = get_parallel().tp_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
```

**Code（Qwen3）：**

```python
# 来源：python/sglang/srt/models/qwen3.py L87-L91
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
```

---

## Q4：`forward_batch` 里模型层最关心哪些字段？

**Explain：** Attention 层主要读 `forward_mode`（extend/decode/speculative）、`out_cache_loc`（KV 写入位置）。PP 路径还可能有 piecewise CUDA graph 上下文。Embedding 路径读 pooling 相关字段。

**Code（RadixAttention 分支示例，模型层传入同一 batch）：**

```python
# 来源：python/sglang/srt/layers/radix_attention.py L127-L153
        if (
            forward_batch.forward_mode.is_extend()
            and get_tc_piecewise_forward_context() is not None
        ):
            if self.qk_head_dim != self.v_head_dim:
                output = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
            else:
                output = torch.empty_like(q)
            if is_in_breakable_cuda_graph():
                breakable_unified_attention_with_output(
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
            else:
                unified_attention_with_output(
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
            return output
        else:
            return get_attn_backend().forward(
                q,
                k,
                v,
                self,
                forward_batch,
                save_kv_cache,
                **kwargs,
            )
```

**Comment：** RadixAttention 详述；模型开发者只需保证 `forward_batch` 原样下传。

---

## Q5：load_weights 为什么 skip 某些 tensor？

**Explain：** 常见 skip：PP 非本 stage 层、HF 冗余 `rotary_emb.cos_cached`、tie embeddings 的 `lm_head.weight`、vision tower 占位、旧版 `kv_scale` 命名。

**Code：**

```python
# 来源：python/sglang/srt/models/llama.py L657-L666
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
```

---

## Q6：Qwen3 的 QK-Norm 在 checkpoint 里怎么对应？

**Explain：** HF Qwen3 checkpoint 含独立 `q_norm.weight` / `k_norm.weight`；SGLang `Qwen3Attention` 有对应 `RMSNorm` 子模块，走默认 `weight_loader` 分支（非 stacked mapping）。

**Code：**

```python
# 来源：python/sglang/srt/models/qwen3.py L118-L119
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
```

**Comment：** `apply_qk_norm` 在 forward 中对 head_dim 最后一维做 norm，再 RoPE。

---

## Q7：一个文件多个 EntryClass 会共享权重吗？

**Explain：** **不会自动共享**。`Phi3ForCausalLM(LlamaForCausalLM)` 是继承关系，共用代码；Registry 里是两个独立键，各自实例化。加载时由 `architectures` 决定实例化哪一个类。

**Code：**

```python
# 来源：python/sglang/srt/models/llama.py L843-L844
class InternLM3ForCausalLM(LlamaForCausalLM):
    pass
```

---

## Q8：native vs Transformers 性能差异根因？

**Explain：** Native 路径：(1) fused QKV/gate-up linear + TP shard；(2) `RadixAttention` 对接 paged KV 与 FlashInfer/FA backend；(3) CUDA graph / piecewise graph 钩子。Transformers wrapper 缺少这些集成，适合兼容而非吞吐。

**Comment：** 排查慢模型先确认 `_resolved_model_impl` 是否为 `NATIVE`。

---

## Q9：`SGLANG_DISABLED_MODEL_ARCHS` 何时使用？

**Explain：** 某些 arch 模块 import 会拉重型可选依赖（如特定 multimodal），环境变量可跳过注册避免 import 失败拖垮整个 Registry 扫描。

**Code：**

```python
# 来源：python/sglang/srt/models/registry.py L100-L102
            if name.split(".")[-1] in envs.SGLANG_DISABLED_MODEL_ARCHS.get():
                logger.debug(f"Skip loading {name} due to SGLANG_DISABLED_MODEL_ARCHS")
                continue
```

---

## Q10：forward_split_prefill 谁调用？

**Explain：** Chunked prefill / 超长 prompt 时 ModelRunner 按层区间多次调用，避免单次 forward OOM。与 Scheduler `PrefillAdder` 的 chunk 策略配合。

**Code：**

```python
# 来源：python/sglang/srt/models/llama.py L565-L588
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
```

**Comment：** `forward_batch.hidden_states` / `residual` 跨 split 调用持久化中间态。

---

## 验证建议（零基础可试）

1. **HF architecture 能否命中 Registry** 
 - 操作：打开模型 `config.json` 的 `architectures[0]`，在 sglang 源码 `python/sglang/srt/models/` 下 grep 该类名是否出现在某文件的 `EntryClass = [...]` 中。 
 - 预期：命中则走 native 实现；未命中或 `--model-impl transformers` 时 fallback 到 Transformers wrapper（见 Q1）。 
 - 对应：Q1、Q2、[[13-Models-通用-01-核心概念|01-核心概念 §Registry]]、[[13-Models-通用-02-源码走读|02-源码走读 §registry.py]]

2. **启动日志里的 `_resolved_model_arch`** 
 - 操作：用小模型启动 SGLang（CPU 也可只看到加载阶段日志），搜索 `_resolved_model_arch` 或 `resolved model` 字样。 
 - 预期：Qwen3/Llama 等常见模型应显示 native 类名（如 `Qwen3ForCausalLM`），而非 `TransformersForCausalLM`。 
 - 对应：Q1、Q8、[[13-Models-通用-03-数据流与交互|03-数据流与交互 §1]]

3. **`SGLANG_DISABLED_MODEL_ARCHS` 跳过注册** 
 - 操作：启动 Python（无需 GPU），设 `SGLANG_DISABLED_MODEL_ARCHS=llama` 后 `import sglang.srt.models.registry`；再 `from sglang.srt.models.registry import ModelRegistry` 查 `LlamaForCausalLM` 是否在映射表。 
 - 预期：被禁模块不注册，对应 architecture 解析失败或走 fallback；日志有 `Skip loading ... due to SGLANG_DISABLED_MODEL_ARCHS`。 
 - 对应：Q9、[[13-Models-通用-00-MOC|README §核心源码锚点]]

4. **Llama vs Qwen3 的 TP 切分差异（静态对照）** 
 - 操作：并排 grep `llama.py` 的 `get_parallel().tp_size` 与 `qwen3.py` 的 `attn_tp_rank` / `attn_tp_size`，口述 Attention head 数如何除以并行度。 
 - 预期：Llama 用全局 TP；Qwen3 Attention 用独立 attn_tp 组，支持 DP Attention（MLP 与 Attention 不同并行布局）。 
 - 对应：Q3、[[13-Models-通用-01-核心概念|01-核心概念 §Qwen3 差异]]
