---
title: "专用模型 · 排障指南"
type: troubleshooting
framework: sglang
topic: "专用模型"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# 专用模型 · 排障指南

本页面向 DeepSeek V2/V3/V3.2 等专用模型排障。读完后，你应该能区分类名注册、attention backend 分派、expert parallel、shared expert fusion、DSA/PP proxy、权重加载和 CP shape 几类问题的第一源码入口。

## 症状速查

| 症状 | 第一入口 | 先判断什么 |
|------|----------|------------|
| DeepSeek V3/V3.2 命中了同一个文件，行为却不同 | `EntryClass`、config 字段 | 类名命中相同实现，行为由 config/server args 分叉 |
| prefill 没走 MLA 或走了 MHA chunked KV | `dispatch_attn_forward_method`、backend handler | 当前 forward mode、backend、prefix 长度和 CP 条件 |
| `tp_size > n_routed_experts` 报错 | `DeepseekV2MoE.__init__` | 把 expert parallel 当成 tensor parallel 配错了 |
| shared experts fusion 没开启 | `determine_num_fused_shared_experts` | 是否被 SBO/TBO、DeepEP、硬件、量化、architecture 白名单关闭 |
| V3.2 DSA 在 PP 下断言缺 `topk_indices` | `DeepseekV2Model.forward` | 上一 PP stage 是否应该把 top-k index 放进 `PPProxyTensors` |
| 权重名找不到或 shared expert 权重错槽 | `DeepseekV2WeightLoaderMixin.do_load_weights` | stage skip、shared remap、expert mapping、fused qkv_a 是否命中 |
| CP shape 异常 | `ForCausalLM.forward`、model forward CP split/gather | metadata 是否设置，split/gather token 数是否一致 |

## Q1：DeepSeek V2/V3/V3.2 为什么在同一个文件里？

源码上，V3 和 V3.2 只是继承 `DeepseekV2ForCausalLM`，`EntryClass` 暴露三个类名给 Registry：

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

判断方法：类账只解决“实例化哪个 Python 类”。具体是否 DSA、是否有 `q_lora_rank`、是否 shared experts fusion，要继续看 config 字段和 server args。不要因为它们在同一个文件里，就认为 V2/V3/V3.2 运行行为完全相同。

验证点：启动后看 `_resolved_model_arch`；再打印 `config.architectures[0]`、`is_deepseek_dsa(config)`、`config.n_routed_experts`、`config.n_shared_experts`。

## Q2：为什么 prefill 走 MHA，而 decode 走 MLA？

DeepSeek attention 的第一层选路会根据 forward mode 在 prefill backend 和 decode backend 之间选字符串，再交给 backend handler：

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L1788-L1816
    def dispatch_attn_forward_method(
        self, forward_batch: ForwardBatch
    ) -> AttnForwardMethod:
        # Determine attention backend name for current forward batch: prefer the
        # name stamped per-runner on the backend object, else resolve from server args.
        backend = get_attn_backend()
        server_args = get_global_server_args()
        default_prefill_str, default_decode_str = server_args.get_attention_backends()
        prefill_backend_str = (
            backend.prefill_attention_backend_str or default_prefill_str
        )
        decode_backend_str = backend.decode_attention_backend_str or default_decode_str
        if forward_batch.forward_mode.is_decode_or_idle():
            attention_backend = decode_backend_str
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            # Use the specified backend for speculative operations (both verify and draft extend)
            if server_args.speculative_attention_mode == "decode":
                attention_backend = decode_backend_str
            else:  # default to prefill
                attention_backend = prefill_backend_str
        else:
            attention_backend = prefill_backend_str
        self.current_attention_backend = attention_backend

        handler = AttentionBackendRegistry.get_handler(attention_backend)
        return handler(self, forward_batch)
```

generic handler 还会在 extend 且 prefix 较长或 prefix 为 0 时选择 MHA one-shot/chunked KV：

```python
# 来源：python/sglang/srt/models/deepseek_common/attention_backend_handler.py L93-L108
    if (
        not disable_ragged
        and forward_batch.forward_mode.is_extend_without_speculative()
        and (
            (
                sum_extend_prefix_lens >= attn.chunked_prefix_cache_threshold
                and not attn.disable_chunked_prefix_cache
            )
            or sum_extend_prefix_lens == 0
        )
    ):
        if _support_mha_one_shot(attn, forward_batch, backend_name):
            return AttnForwardMethod.MHA_ONE_SHOT
        return AttnForwardMethod.MHA_CHUNKED_KV
    else:
        return _dispatch_mla_subtype(attn, forward_batch)
```

判断方法：prefill 走 MHA 不一定是回退失败。对长 prefix，MHA chunked KV 是显式策略，用来避免 prefix KV 一次性展开过大。

验证点：记录 `forward_batch.forward_mode`、`self.current_attention_backend`、`sum_extend_prefix_lens`、`chunked_prefix_cache_threshold` 和最终 `AttnForwardMethod`。

## Q3：为什么 `tp_size > n_routed_experts` 直接报错？

MoE 的 expert 维度不是 hidden dim。DeepSeek 在 MoE 初始化时直接拦截 TP 大于 routed expert 数的配置：

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L580-L587
        n_hash_layers = getattr(config, "num_hash_layers", 0)
        self.is_hash = layer_id < n_hash_layers and not (is_deepseek_v4 and is_nextn)

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )
```

判断方法：如果你想切 expert，应该优先看 EP/MoE parallel 配置，而不是盲目增大 TP。TP 大到超过 expert 数，会让 expert shard 语义失效。

验证点：打印 `get_parallel().tp_size`、`get_parallel().moe_ep_size`、`config.n_routed_experts`。

## Q4：shared experts fusion 为什么被自动关闭？

`determine_num_fused_shared_experts` 会集中判断禁用原因。SBO/TBO、DeepEP 默认、architecture 或 routed expert 数不在白名单、硬件能力不足、W4AFP8 等都会关闭 fusion：

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2730-L2795
    def determine_num_fused_shared_experts(
        self, architecture: str = "DeepseekV3ForCausalLM"
    ):
        self.num_fused_shared_experts = 0
        server_args = get_global_server_args()

        if server_args.disable_shared_experts_fusion:
            return

        disable_reason = None
        if server_args.enforce_shared_experts_fusion:
            pass
        elif is_sbo_enabled() or is_tbo_enabled():
            disable_reason = "SBO/TBO enabled: incompatible with fusing shared expert into MoE kernel."
        elif is_deepep_class_backend():
            disable_reason = "DeepEP: fusion off by default (use --enforce-shared-experts-fusion to enable)."
        elif (
            self.config.architectures[0] != architecture
            # Allow-list of n_routed_experts values that have been validated
            # for shared-experts fusion under this code path. Currently:
            #   256 -> DeepSeek-V3 / R1
            #   384 -> Kimi-K2.5, only when the checkpoint is Quark MXFP4
            #          (amd/Kimi-K2.5-MXFP4); the standard
            #          moonshotai/Kimi-K2.5 (compressed-tensors) checkpoint
            #          stores the shared expert loose and is NOT pre-fused,
            #          so the fused path silently mis-loads it.
            or self.config.n_routed_experts not in (256, 384)
            or self.config.n_shared_experts != 1
            or (
                self.config.n_routed_experts == 384
                and (
                    self.quant_config is None or self.quant_config.get_name() != "quark"
                )
            )
        ):
            disable_reason = "Config does not support fused shared expert(s)."
        elif (
            (not _is_cuda or torch.cuda.get_device_capability("cuda") < (8, 0))
            and (not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4))
            and (not _is_musa or torch.musa.get_device_capability("musa") < (3, 1))
        ):
            disable_reason = (
                "Only Deepseek V3/R1 on NV-platform with capability >= 80 "
                "or AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization."
                "or MT-platform with capability >= 31 can use shared experts fusion optimization."
            )
        elif get_parallel().moe_ep_size > 1 and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = (
                "Only Deepseek V3/R1 on AMD-platform with capability >= gfx942(MI30x) "
                "can use shared experts fusion optimization under expert parallelism."
            )
        elif self.quant_config and self.quant_config.get_name() == "w4afp8":
            disable_reason = "Deepseek V3/R1 W4AFP8 model uses different quant method for routed experts and shared experts."

        if disable_reason is not None:
            server_args.disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts
```

判断方法：fusion 关闭通常是安全策略，不是 silent failure。先找日志里的 disable reason，再决定是否应该 `--enforce-shared-experts-fusion`。

验证点：记录 `server_args.disable_shared_experts_fusion`、`server_args.enforce_shared_experts_fusion`、`num_fused_shared_experts`、`quant_config.get_name()`。

## Q5：DeepEP fusion 如何改变 expert 数和 top-k？

DeepEP shared expert fusion 把 shared expert 当成本 EP rank 的额外 expert slot。普通 fusion 是 `n_routed + num_fused_shared_experts`；DeepEP fusion 是 `n_routed + moe_ep_size`，top-k 多选一个 shared slot：

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L553-L573
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
```

判断方法：看到 `num_experts` 比 `n_routed_experts` 大，先确认是不是 shared expert slot。不要用 checkpoint 原始 expert 数直接否定运行时布局。

验证点：打印 `num_experts_for_moe`、`top_k_for_moe`、`moe_ep_size`、`num_fused_shared_experts`。

## Q6：V3.2 DSA 为什么会缺 `topk_indices`？

DSA 支持 skip-topk。跨 PP stage 时，如果下一 stage 的起始层需要复用上一层 top-k index，上一 stage 必须把 `topk_indices` 放进 proxy：

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2600-L2625
        if not self.pp_group.is_last_rank:
            proxy_tensors = {
                "hidden_states": hidden_states,
                "residual": residual,
            }
            if (
                self.use_dsa
                and dsa_forward_uses_topk
                and self.end_layer < self.config.num_hidden_layers
                and dsa_layer_skips_topk(self.config, self.end_layer)
            ):
                if (
                    not forward_batch.forward_mode.is_idle()
                    and hidden_states.shape[0] != 0
                ):
                    assert topk_indices is not None, (
                        f"PP stage ending at layer {self.end_layer} must forward "
                        "DSA topk_indices because the next stage starts on a "
                        "skip-topk layer."
                    )
                if topk_indices is None:
                    topk_indices = hidden_states.new_empty(
                        (0, get_dsa_index_topk(self.config)), dtype=torch.int32
                    )
                proxy_tensors["topk_indices"] = topk_indices
            return PPProxyTensors(proxy_tensors)
```

判断方法：如果断言说起始 stage 需要 topk，检查前一 stage 的 `end_layer` 是否正好落在 skip-topk 边界。

验证点：打印每个 PP rank 的 `start_layer/end_layer`、`dsa_layer_skips_topk(config, start_layer/end_layer)`、`PPProxyTensors.tensors.keys()`。

## Q7：权重名找不到时先查什么？

先查 PP stage skip 和 shared expert remap。DeepSeek loader 会跳过不属于本 PP stage 的层，并在 fusion 开启时把 `mlp.shared_experts` 改成 `mlp.experts.<n_routed_experts>`：

```python
# 来源：python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py L202-L224
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            params_dict = dict(self.named_parameters())
            weight_names = []
            for name, loaded_weight in weights:
                use_async_loading = should_async_load(loaded_weight)
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
                if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
                    name = name.replace(
                        "mlp.shared_experts",
                        f"mlp.experts.{self.config.n_routed_experts}",
                    )

                weight_names.append(name)
```

如果还没命中，再查 dense stacked mapping、expert mapping、embedding/norm PP skip、fused qkv_a：

```python
# 来源：python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py L327-L359
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        # Skip loading embed_tokens if not first rank in pipeline parallelism
                        if ".embed_tokens." in name and not self.pp_group.is_first_rank:
                            continue
                        # Skip loading norm if not last rank in pipeline parallelism
                        if ".norm." in name and not self.pp_group.is_last_rank:
                            continue
                        if fuse_qkv_a_proj and (
                            "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                        ):
                            cached_a_proj[name] = _clone_if_runai_streamed_tensor(
                                loaded_weight
                            )
                            q_a_proj_name = (
                                name
                                if "q_a_proj" in name
                                else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                            )
                            kv_a_proj_name = (
                                name
                                if "kv_a_proj_with_mqa" in name
                                else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                            )

                            # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                            if (
                                q_a_proj_name in cached_a_proj
                                and kv_a_proj_name in cached_a_proj
                            ):
                                q_a_proj_weight = cached_a_proj[q_a_proj_name]
```

判断方法：不要把 warning 中的 “not found in params_dict” 立即等价为 checkpoint 错。DeepSeek loader 有大量合法跳过和延迟 fusion。

验证点：记录原始 name、stage skip 结果、remap 后 name、`name in params_dict`、是否进入 `cached_a_proj`。

## Q8：CP shape 异常应该从哪里查？

ForCausalLM 入口设置 metadata，model forward 根据 metadata split/gather。两个地方要一起看：

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2531-L2536
        if dsa_use_prefill_cp(
            forward_batch, self.dsa_enable_prefill_cp
        ) or mla_use_prefill_cp(forward_batch, self.mla_enable_prefill_cp):
            if self.pp_group.is_first_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)
```

```python
# 来源：python/sglang/srt/models/deepseek_v2.py L2633-L2643
        if self.pp_group.is_last_rank and (
            dsa_use_prefill_cp(forward_batch, self.dsa_enable_prefill_cp)
            or mla_use_prefill_cp(forward_batch, self.mla_enable_prefill_cp)
        ):
            # allgather + rerrange
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
```

判断方法：如果 split 后 shape 对、gather 后错，查 `attn_cp_metadata` 和 rerange；如果 split 前就错，查 `seq_lens_cpu`、`extend_seq_lens_cpu` 和 CP split 条件。

验证点：打印 split 前后 token 数、positions shape、last rank gather 后 shape。

## 小结

DeepSeek 专用模型的排障顺序应该是：先判断落在哪本账，再查对应接缝。Attention 问题从 method 选路开始，MoE 问题从 expert 数和 fusion 形态开始，PP/DSA 问题从 `topk_indices` 传递开始，权重问题从 name remap 开始，CP 问题从 metadata 与 split/gather 开始。
