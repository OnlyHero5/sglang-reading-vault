---
title: "MoE · 排障指南"
type: troubleshooting
framework: sglang
topic: "MoE"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# MoE · 排障指南

## 读者任务

这篇按症状排障。MoE 问题通常看起来像“吞吐下降”或“某个 rank 慢”，但原因可能分别在 top-k、A2A、expert GEMM、EPLB、padding 或 graph capture。

## Q1：MoE 层到底慢在 router 还是 dispatch？

先看 `FusedMoE.forward_impl` 的三段：dispatch、core、combine。router/top-k 在进入 `experts` 前已经发生；如果 profiler 显示慢在 `dispatcher.dispatch` 或 `dispatcher.combine`，那是搬运问题，不是 gate 打分问题。

```python
# 来源：sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py L1138-L1150
        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )

        combine_input = self.run_moe_core(
            dispatch_output=dispatch_output,
        )

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            final_hidden_states = self.dispatcher.combine(combine_input=combine_input)
```

验证：分别在 dispatch、`run_moe_core`、combine 前后打时间戳。EP 下慢在 dispatch/combine 很常见。

## Q2：为什么 top-k 看起来正常，实际 dispatch 到了别的专家？

因为 EPLB/EP 可能把 logical expert id 映射成 physical expert id。`topk_ids_logical_to_physical` 是关键入口。

```python
# 来源：sglang/python/sglang/srt/eplb/expert_location_dispatch.py L79-L98
def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor,
    info: Optional[ExpertLocationDispatchInfo],
    log2phy_prob: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if info is None:
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        return _topk_ids_logical_to_physical_static(topk_ids, info)
    if info.ep_dispatch_algorithm in ["dynamic", "fake"]:
        return _topk_ids_logical_to_physical_dynamic(topk_ids, info)
    if info.ep_dispatch_algorithm == "lp":
        if log2phy_prob is None:
            raise RuntimeError(
                "ep_dispatch_algorithm='lp' but log2phy_prob is None at dispatch "
                f"time (topk_ids.shape={tuple(topk_ids.shape)})."
            )
        return _topk_ids_logical_to_physical_probability(topk_ids, info, log2phy_prob)
    raise NotImplementedError(f"Unknown algorithm {info.ep_dispatch_algorithm}")
```

排查：同时打印 remap 前后的 ids，并确认 `ExpertLocationDispatchInfo.init_new(layer_id)` 是否返回非空信息。

## Q3：DeepEP 为什么会出现阶段断言？

DeepEP 是有阶段状态的。它要求 dispatch A/B、combine A/B 按顺序执行，并最终回到 `INITIAL`。

```python
# 来源：sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py L913-L924
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)
```

验证：在异常前打印 `_stage`。如果停在 `AFTER_DISPATCH_A`，通常是 dispatch B 没被执行或中途异常。

## Q4：EPLB 什么时候会导致短暂停顿？

EPLB 每隔固定 forward pass 数才 rebalance。rebalance 时会 dump 统计、计算新 metadata，并调用 `update_expert_location`。

```python
# 来源：sglang/python/sglang/srt/eplb/eplb_manager.py L47-L87
    # can be more complex if needed
    def _entrypoint(self):
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield

            yield from self.rebalance()

    def rebalance(self):
        logger.info("[EPLBManager] rebalance start")

        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.get_device_module().synchronize()
            time_start = time.time()

        dump_record_output = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )
        logical_count = dump_record_output["logical_count"]
        average_utilization_rate_over_window = dump_record_output[
            "average_utilization_rate_over_window"
        ]

        # Check whether rebalancing is needed
        if not self._check_rebalance_needed(average_utilization_rate_over_window):
            return

        expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )
```

验证：看日志里的 `[EPLBManager] rebalance start/end` 是否与 latency spike 对齐。

## Q5：量化 runner 选错应该看哪里？

`FusedMoE.run_moe_core` 只调用 `self.quant_method.apply`。如果专家 GEMM 跑到错误 backend，要追 `quant_method` 是如何在初始化和 weight load 时确定的。

```python
# 来源：sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py L1178-L1183
    def run_moe_core(self, dispatch_output: DispatchOutput) -> CombineInput:
        # TODO: consider using symmetric memory
        return self.quant_method.apply(
            layer=self,
            dispatch_output=dispatch_output,
        )
```

验证：打印 `type(self.quant_method)` 和 runner backend 配置。dispatch/combine 顺序不应因为量化改变。

## Q6：为什么 piecewise CUDA Graph 会 fallback？

`FusedMoE.forward` 只对 standard 和 bypassed top-k 格式走 custom op；其他格式会回到 `forward_impl`。

```python
# 来源：sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py L1107-L1132
        if is_in_tc_piecewise_cuda_graph():
            if TopKOutputChecker.format_is_standard(topk_output):
                return moe_forward_piecewise_cuda_graph_impl(
                    hidden_states,
                    topk_output.topk_weights,
                    topk_output.topk_ids,
                    topk_output.router_logits,
                    self.layer_id,
                )
            elif TopKOutputChecker.format_is_bypassed(topk_output):
                return fused_moe_bypassed_piecewise_cuda_graph_impl(
                    hidden_states,
                    topk_output.router_logits,
                    topk_output.topk_config.top_k,
                    topk_output.topk_config.topk_group,
                    topk_output.topk_config.num_expert_group,
                    topk_output.topk_config.correction_bias,
                    topk_output.topk_config.renormalize,
                    self.layer_id,
                    topk_output.topk_config.allow_routed_experts_capture,
                )
            else:
                # Make sure there is torch lib op registration for the whole moe layer
                return self.forward_impl(hidden_states, topk_output)
        else:
            return self.forward_impl(hidden_states, topk_output)
```

验证：打印 `topk_output.format` 或用 `TopKOutputChecker` 判断格式。graph break 时不要先怀疑 GEMM kernel，先看 top-k 格式。

## Q7：为什么某些 padded token 会污染 MoE 输出？

top-k post process 会 mask padded region。HIP 路径尤其需要把 padded token 的 expert id 填成安全值，并把权重归零。

```python
# 来源：sglang/python/sglang/srt/layers/moe/topk.py L1766-L1785
    elif _is_hip:
        # On AMD HIP the aiter MoE kernels do not handle topk_ids=-1 safely
        # (negative indices cause illegal memory access). Always fill the padded
        # region with 0 so every kernel sees a valid in-range expert id.
        # Routing weights for padded tokens are zeroed below so their
        # contribution to the hidden state is still zero regardless of the id.
        # Regression: skipping this mask when EPLB is disabled caused garbage
        # MoE routing for models like DeepSeek-R1-MXFP4 (accuracy ~0.09 vs 0.94+).
        _mask_topk_ids_padded_region(topk_ids, num_token_non_padded, fill_value=0)
        # The logical->physical remap is only meaningful when a real
        # expert-location mapping exists. With a trivial placement and EPLB off
        # the map is identity so the remap can be skipped safely.
        if _eplb_remap_enabled():
            topk_ids = topk_ids_logical_to_physical(
                topk_ids, expert_location_dispatch_info
            )
        # NOTE (HIP): padded-token routing-weight zeroing is deferred to the
        # single pass at the end of this function (gated by SGLANG_MORI_NO_PAD_MASK).
        # That final pass re-zeros after any shared-expert append/remap, so a
        # second zeroing here would be redundant (zeroing is idempotent).
```

验证：检查 `num_token_non_padded` 是否正确传入 top-k，padded 区域权重是否为 0。

## Q8：EP rank 上 local expert 数不对怎么办？

`FusedMoE.__init__` 用 `moe_ep_size` 和 shared expert slot 计算本 rank 的 local expert 数。DeepEP 下 shared expert slot 会乘以 EP size。

```python
# 来源：sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py L197-L214
        self.moe_ep_size = get_parallel().moe_ep_size
        self.moe_ep_rank = get_parallel().moe_ep_rank
        self.moe_tp_size = get_parallel().moe_tp_size
        self.moe_tp_rank = get_parallel().moe_tp_rank

        # DeepEP: each rank has its own shared expert slot, so total shared
        # weight slots = num_fused_shared_experts * ep_size.
        # AMD/Standard: shared experts are global, slots = num_fused_shared_experts.
        if num_fused_shared_experts > 0 and is_deepep_class_backend():
            num_shared_slots = num_fused_shared_experts * self.moe_ep_size
        else:
            num_shared_slots = num_fused_shared_experts

        assert (num_experts - num_shared_slots) % self.moe_ep_size == 0
        self._num_global_routed = num_experts - num_shared_slots
        self._num_local_routed = self._num_global_routed // self.moe_ep_size
        self.num_local_experts = self._num_local_routed + num_fused_shared_experts
        self._has_fused_shared = num_fused_shared_experts > 0
```

排查：核对 `num_experts`、`num_fused_shared_experts`、`moe_ep_size` 和实际 dispatcher backend。

## 运行验证

MoE 的问题不要只从一个 runner 看。下面两组检索分别覆盖 runner fallback、top-k 格式、DeepEP dispatch、EPLB、padded token mask 和 EP local expert 计算。

```powershell
rg -n 'TopKOutputChecker|format_is_bypassed|fused_moe_bypassed_piecewise_cuda_graph_impl|def run_moe_core|moe_ep_size|num_local_experts' sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py sglang/python/sglang/srt/layers/moe/topk.py
rg -n 'def topk_ids_logical_to_physical|def dispatch_a|def dispatch_b|class EPLBManager|def _mask_topk_ids_padded_region|def _zero_topk_weights_padded_region' sglang/python/sglang/srt/eplb/expert_location_dispatch.py sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py sglang/python/sglang/srt/eplb/eplb_manager.py sglang/python/sglang/srt/layers/moe/topk.py
```

读输出时先看 `FusedMoE` 是否进入 bypassed piecewise CUDA Graph，再看 `topk.py` 的 padded token 处理，最后看 DeepEP 的 `dispatch_a/dispatch_b`。如果线上症状是 expert id 错位，优先从 `topk_ids_logical_to_physical` 和 `num_local_experts` 两端夹查。
