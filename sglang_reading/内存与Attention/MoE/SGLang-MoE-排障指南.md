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
updated: 2026-07-12
---
# MoE · 排障指南

## 读者任务

这篇按症状排障。每次都同时记录 backend、TopK format、logical/physical ids、dispatcher mode/dtype、quant method 与关键环境变量；缺其中一项，两个完全不同的故障很容易表现成同一个“某 rank 慢”。

## Q1：MoE 层到底慢在 router 还是 dispatch？

先看 `FusedMoE.forward_impl` 的三段：dispatch、core、combine。router/top-k 在进入 `experts` 前已发生；但 dispatcher 内还包含 layout、量化通信、event/hook 等工作，所以慢在 dispatch/combine 只能排除 gate kernel，不能简单归为“纯网络搬运”。

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

排查：同时打印 logical ids、physical routed ids、recorder ids 和最终 dispatch ids，并确认 `ExpertLocationDispatchInfo.init_new(layer_id)` 是否返回非空信息。`dynamic/fake/lp` 本就可能让同一 logical expert 逐 token 落到不同 replica。

特别检查契约漂移：dataclass 字段注解仍写 `Literal["static", "random"]`，ServerArgs 与运行分支实际使用 `static/dynamic/fake/lp`。若配置校验、序列化或 IDE 提示只认旧注解，会出现“运行支持、静态工具拒绝”的错觉。

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

验证：在异常前打印 `_stage`、resolved mode、event/hook 和本轮 handle。若停在 `AFTER_DISPATCH_A`，检查 dispatch B 前的 hook/异步接收是否抛错；若跨请求复用错误 handle，normal combine 也可能在更晚位置暴露问题。

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

验证：先看 `eplb_rebalance_layers_per_chunk`。未分 chunk 时可把同步计时与 spike 对齐；分 chunk 时每个 chunk 前会 `yield`，`start/end` 可能跨多个 forward，不能把整个日志区间当成一次停顿。

## Q5：量化 runner 选错应该看哪里？

`FusedMoE.run_moe_core` 调用 `self.quant_method.apply`，但量化影响面更早开始：runner backend 可决定 TopK format，method 可给 dispatcher 写 quant config，DeepEP output dtype 还会被硬件校正。专家 GEMM backend 错只是其中一种不匹配。

```python
# 来源：sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py L1178-L1183
    def run_moe_core(self, dispatch_output: DispatchOutput) -> CombineInput:
        # TODO: consider using symmetric memory
        return self.quant_method.apply(
            layer=self,
            dispatch_output=dispatch_output,
        )
```

验证：打印 `type(self.quant_method)`、runner backend、`topk_output.format`、dispatcher output dtype/scale 和 routed scaling 的应用位置。外层 dispatch/core/combine 顺序可以不变，中间 ABI 仍可能完全不同。

## Q6：为什么 piecewise CUDA Graph 走了另一条函数？

`FusedMoE.forward` 在 piecewise context 中为 standard 与 bypassed 选择专门函数，其他格式调用 `forward_impl`。源码注释要求为整个 MoE layer 注册 torch op，因此仅凭“调用了 `forward_impl`”不能证明发生 eager fallback 或 graph break。

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

验证：打印 `topk_output.format`、piecewise context，并查看整层 op 是否已注册/捕获。只有结合 graph 日志或 capture 证据，才能下“graph break”结论。

## Q7：为什么某些 padded token 会污染 MoE 输出？

top-k post-process 会处理 padded region。HIP 路径总把 padded expert id 填成安全的 0；最终是否清零 weights 还受 `SGLANG_MORI_NO_PAD_MASK` 控制，不能无条件假定为 0。

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

验证：检查 `num_token_non_padded` 与 `SGLANG_MORI_NO_PAD_MASK`。若开关为 false，期望最终 weights 为 0；若开关为 true，跳过清零是显式配置行为，应继续核下游是否安全消费。

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

## Q9：为什么 expert 分布突然“完美均匀”？

先检查 `SGLANG_SIMULATE_UNIFORM_EXPERTS` 与 `SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS`。它们是 benchmark-only override，会在真实 gating 之后重写 ids/weights，且两者互斥。若忘记记录环境，recorder 与 dispatch 都可能看起来合理，却不再代表模型路由。

操作：输出两个环境变量，并在 `select_experts` 的 override 前后各抓一次 ids。预期：生产环境均为 false；benchmark 环境应在报告里明确标注。

## Q10：为什么 shared expert 输出整体偏小或偏大？

检查 routed scaling ownership。默认 post-MoE scaling 下，DeepEP shared weight 用 `1/routed_scaling_factor`；Aiter 把 scaling 预折叠进 routed weights，shared weight应为 `1.0`。源码已注明其他预折叠 backend 仍有未收口边界，因此 ModelOpt NVFP4、CUTLASS/TRTLLM-routed FP8 等路径要做数值对照，不能只看 kernel 是否成功运行。

操作：固定输入，对照非 DeepEP/高精度路径，记录 routed weights、shared weight 与最终输出范数。预期：shared expert 的净贡献保持 1.0×，不随 routed scaling 重复缩放。

## Q11：为什么零 token rank 在 LP dispatch 下卡死？

LP solver 含 EP all-reduce，空 rank 也必须参与。通用入口要求调用 `empty_topk_output(layer_id=...)` 才会让该层 solver 处理空 tensor；若调用方省略 `layer_id`，要结合该模型是否实际允许 LP 判断可达性。

操作：构造 DP-attention 下某 rank 零 token 的 batch，记录所有 rank 是否进入 solver collective。预期：空 rank 不做 expert compute，但必须完成同一 collective。

## Q12：DeepEP normal 与 low-latency 为什么结果形状或 scale 不同？

两者输出 ABI 本就不同：normal 有 `num_recv_tokens_per_expert`，low-latency 有 packed hidden、`masked_m/expected_m`；FP8/NVFP4/INT8 通信还可附带 scale。`auto` 对 prefill/extend 与 decode 解析到不同实现，不能拿一边的 shape 断言另一边异常。

操作：记录 `get_is_extend_in_batch()`、resolved mode、dispatch output type、hidden scale 和同步方式。预期：同一 mode 内字段稳定，mode 切换时按相应 contract 变化并最终回到 `INITIAL`。

## 运行验证

MoE 的问题不要只从一个 runner 看。下面两组检索覆盖 graph 分支、top-k 格式、DeepEP ABI、EPLB、padding 与 local expert 计算。

```powershell
rg -n 'TopKOutputChecker|format_is_bypassed|fused_moe_bypassed_piecewise_cuda_graph_impl|def run_moe_core|moe_ep_size|num_local_experts' sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py sglang/python/sglang/srt/layers/moe/topk.py
rg -n 'def topk_ids_logical_to_physical|def dispatch_a|def dispatch_b|class EPLBManager|def _mask_topk_ids_padded_region|def _zero_topk_weights_padded_region' sglang/python/sglang/srt/eplb/expert_location_dispatch.py sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py sglang/python/sglang/srt/eplb/eplb_manager.py sglang/python/sglang/srt/layers/moe/topk.py
```

读输出时先冻结“配置快照”，再沿 carrier → ids → dispatcher output → core → combine 查。若是 id 错位，从 logical capture 与 final dispatch 两端夹查；若是数值漂移，从 routed/shared scaling 两端夹查；若是卡死，从 stage state 与 collective participation 两端夹查。
