---
title: "MoE · 数据流"
type: dataflow
framework: sglang
topic: "MoE"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/dataflow
  - source-reading
updated: 2026-07-10
---
# MoE · 数据流

## 读者任务

这篇只追数据对象：`hidden_states`、`router_logits`、`topk_ids`、`topk_weights`、logical/physical expert id、dispatch state、combine input。目标是能判断一个异常发生在“路由决策”还是“专家搬运”。

## 生命周期表

| 阶段 | 输入 | 输出 | 下游消费者 |
|------|------|------|------------|
| gate | `[tokens, hidden]` | `router_logits` | `TopK` |
| top-k | `router_logits` | `topk_ids/topk_weights` | dispatcher、recorder |
| remap | logical expert id | physical expert id | EP/DeepEP dispatcher |
| dispatch | hidden + physical ids | expert 分组 hidden | `quant_method.apply` |
| core | expert 分组 hidden | per-expert output | combine |
| combine | expert output + weights | 原 token 顺序 output | transformer 后续层 |
| EPLB | expert 计数 | expert location metadata | 下一轮 top-k/remap |

## 1. `TopKOutput` 是跨模块数据契约

标准格式让 dispatcher 和 MoE runner 不必关心 top-k 是如何算出来的。

```python
# 来源：sglang/python/sglang/srt/layers/moe/topk.py L270-L279
class StandardTopKOutput(NamedTuple):
    """Standard top-k output format."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.STANDARD
```

当 runner 自己能处理 routing 时，可以使用 bypassed 格式延迟 materialize。

```python
# 来源：sglang/python/sglang/srt/layers/moe/topk.py L313-L336
class BypassedTopKOutput(NamedTuple):
    """Bypassed top-k output format."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    topk_config: TopKConfig
    num_token_non_padded: Optional[torch.Tensor] = None
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.BYPASSED

    def to_standard(self, layer_id: Optional[int] = None) -> StandardTopKOutput:
        """Materialize routing tensors. Used by MoE kernels that need explicit
        topk_ids / topk_weights rather than doing routing internally."""
        return select_experts(
            hidden_states=self.hidden_states,
            router_logits=self.router_logits,
            topk_config=self.topk_config,
            layer_id=layer_id,
            num_token_non_padded=self.num_token_non_padded,
            expert_location_dispatch_info=self.expert_location_dispatch_info,
        )
```

这个格式差异直接影响 piecewise CUDA Graph：有些 graph op 接受 standard，有些接受 bypassed。

## 2. logical expert id 可以被映射为 physical expert id

EPLB/EP 下，top-k 先选 logical expert，再在 dispatch 前映射到 physical expert。静态映射直接查表，动态映射在候选 physical expert 里随机选一个。

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

```python
# 来源：sglang/python/sglang/srt/eplb/expert_location_dispatch.py L101-L127
def _topk_ids_logical_to_physical_static(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    physical_topk_ids = info.partial_logical_to_rank_dispatch_physical_map[topk_ids]
    if _is_hip:
        physical_topk_ids = physical_topk_ids.to(topk_ids.dtype)
    return physical_topk_ids


def _topk_ids_logical_to_physical_dynamic(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    topk_ids_original_shape = topk_ids.shape
    original_dtype = topk_ids.dtype
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    chosen_dispatch_index = (
        torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
        % info.partial_logical_to_all_physical_map_num_valid[topk_ids]
    )
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]
    if _is_hip:
        topk_ids = topk_ids.to(original_dtype)

    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids
```

排查“同一个 logical expert 为什么跑到不同 rank”时，先看这里的 dispatch algorithm。

## 3. post process 同时处理 remap、padding 和统计

top-k 结束后，`_post_process_topk_ids` 负责 logical-to-physical remap、padding region mask、shared expert 追加和 recorder id 选择。

```python
# 来源：sglang/python/sglang/srt/layers/moe/topk.py L1720-L1747
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_fused_shared_experts = topk_config.num_fused_shared_experts
    fused_shared_experts_scaling_factor = (
        topk_config.fused_shared_experts_scaling_factor
    )
    capture_routed_experts_if_allowed(topk_config, layer_id, topk_ids)
    recorder_topk_ids = None
    if _is_cuda:
        # LP path: solve LP outside torch.compile (the solver contains an
        # EP all-reduce that can't run inside compiled regions).
        log2phy_prob = None
        if (
            expert_location_dispatch_info is not None
            and getattr(expert_location_dispatch_info, "ep_dispatch_algorithm", None)
            == "lp"
        ):
            from sglang.srt.eplb.lplb_solver import get_global_lplb_solver

            lplb_solver = get_global_lplb_solver(layer_id)
            if lplb_solver is not None:
                log2phy_prob = lplb_solver.solve(topk_ids)

        if log2phy_prob is not None:
            topk_ids = topk_ids_logical_to_physical(
                topk_ids, expert_location_dispatch_info, log2phy_prob
            )
            _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
```

这段也解释了为什么 LP 路径不能简单放进 torch.compile：它包含 EP all-reduce。

## 4. `FusedMoE` 根据 EP/TP 切出本地 expert 视图

初始化时会读取并行状态，计算每个 rank 持有多少 routed expert，以及 shared expert slot 如何计数。

```python
# 来源：sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py L188-L218
        self.layer_id = layer_id
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_fused_shared_experts = num_fused_shared_experts

        self.enable_flashinfer_cutlass_moe = (
            get_moe_runner_backend().is_flashinfer_cutlass()
        )
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

        assert intermediate_size % self.moe_tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.moe_tp_size
        self.reduce_results = reduce_results
```

如果 local expert 数量不符合预期，先看 `moe_ep_size`、shared expert slot 和 `num_experts` 是否整除。

## 5. DeepEP dispatcher 保存阶段中间状态

DeepEP 的 `dispatch_a` 把内部状态暂存到 `_dispatch_intermediate_state`，`dispatch_b` 取出后删除；combine 也是同样模式。

```python
# 来源：sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py L908-L951
    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
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

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        self.combine_a(combine_input)
        ret = self.combine_b()
        return ret

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)
```

如果你在 overlap 或异常路径看到 stage assert，重点查是否漏掉了 B 阶段或 combine 没回到 `INITIAL`。

## 交互矩阵

| 读写方 | 写入 | 读取 |
|--------|------|------|
| 模型 MoE block | `router_logits`、`topk_output` | `hidden_states`、`forward_batch` |
| `select_experts` | `topk_ids/topk_weights`、recorder ids | `TopKConfig`、`router_logits`、EPLB info |
| EPLB dispatch | physical top-k ids | logical top-k ids、metadata |
| `BaseDispatcher` | `DispatchOutput`、combined hidden | hidden、top-k 输出、hook |
| `quant_method.apply` | `CombineInput` | expert 分组 hidden、expert weights |
| `EPLBManager` | expert location metadata | distribution recorder output |

## 状态自检

排查时按这个顺序问：

1. `router_logits` 形状是否是 `[tokens, experts]`。
2. `topk_ids/topk_weights` 是否是 `[tokens, top_k]`。
3. padded token 区域是否被 mask。
4. logical id 是否被映射为 physical id。
5. dispatch 后 token 是否按 expert 分组。
6. `run_moe_core` 是否用了预期的量化 runner。
7. combine 后是否回到原 token 顺序。
8. TP/EP all-reduce 是否只在需要时发生。

## 运行验证

MoE 数据流的轻量验证要覆盖 `TopKOutput` 格式、logical/physical expert 映射、fused MoE 核心执行、DeepEP dispatch/combine 四段。

```powershell
rg -n 'class StandardTopKOutput|class BypassedTopKOutput|def topk_ids_logical_to_physical|def _topk_ids_logical_to_physical|def select_experts|def forward_impl|def dispatch_a|def dispatch_b|def combine_a|def combine_b|run_moe_core|TopKOutput' sglang/python/sglang/srt/layers/moe/topk.py sglang/python/sglang/srt/eplb/expert_location_dispatch.py sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py
```

如果上游改动后 `TopKOutput` 或 DeepEP stage 命中位置变化，优先重新检查“logical id 到 physical id”和“dispatch A/B、combine A/B 是否成对”这两个不变量。
