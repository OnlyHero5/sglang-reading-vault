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
updated: 2026-07-12
---
# MoE · 数据流

## 读者任务

这篇只追数据对象及其所有权：`hidden_states`、`router_logits`、不同 `TopKOutput`、logical/physical/recorder ids、dispatch output、scales、combine input。目标是能判断异常发生在 scoring、materialization、remap、通信、expert core 还是 scaling，而不是笼统归因于“路由”或“搬运”。

## 生命周期表

| 阶段 | 输入 | 输出 | 下游消费者 |
|------|------|------|------------|
| gate | `[tokens, hidden]` | `router_logits` | `TopK` |
| top-k contract | hidden + logits + config | standard / bypassed / Triton routing data | runner、dispatcher |
| post-process | logical ids/weights | physical ids、recorder ids、shared slots | dispatcher、recorder |
| dispatch | hidden + format-specific routing | expert 分组 hidden、可选 scales、layout metadata | `quant_method.apply` |
| core | expert 分组 hidden | per-expert output | combine |
| combine | expert output + ids/weights/handle | 原 token 顺序 output | transformer 后续层 |
| EPLB placement | expert 计数 | expert location metadata | 后续 physical dispatch |

## 1. `TopKOutput` 是跨模块数据契约

标准格式让需要显式 ids/weights 的 dispatcher 和 runner 不必关心 top-k 如何算出；它不是全系统唯一 ABI。

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

完整格式矩阵是：

| carrier | 主要内容 | 典型选择条件 | materialize 位置 |
|---|---|---|---|
| `StandardTopKOutput` | weights、ids、logits | 默认路径 | `TopK/select_experts` |
| `StandardTopKOutputPacked` | standard + packed ids | 实验性 LoRA fused top-k pack | gating kernel |
| `TritonKernelTopKOutput` | routing/gather/scatter data | Triton kernels runner | `routing(...)` |
| `BypassedTopKOutput` | hidden、logits、config、metadata | FlashInfer/TRTLLM 类 runner | runner 内部或 `to_standard` |

这组差异会影响 piecewise CUDA Graph 的函数选择，但“进入通用 `forward_impl`”本身不等于 eager graph break。

## 2. logical expert id 可以被映射为 physical expert id

EPLB/EP 下，top-k 先选 logical expert，再在 dispatch 前映射到 physical expert。静态映射直接查表，`dynamic/fake` 在候选 replica 中随机选一个，`lp` 用求解器给出的概率选择；因此 placement 更新频率与逐 token dispatch 频率必须分开讨论。

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

排查“同一个 logical expert 为什么跑到不同 rank”时，先看 dispatch algorithm；同时注意 dataclass 注解仍写 `static/random`，实际运行值已经是 `static/dynamic/fake/lp`。

## 3. post process 同时处理 remap、padding 和统计

top-k 结束后，`_post_process_topk_ids` 不是一个简单 remap helper，而是数据语义的汇合点。顺序大致为：

1. 先捕获 logical routed experts。
2. 若为 LP，先在 compiled region 外求 `log2phy_prob`，再做 logical→physical。
3. 处理 CUDA/HIP padding id；HIP 的最终 weight-zero 还受环境变量控制。
4. 固化 `recorder_topk_ids`；它可能保留 routed physical ids，而最终 dispatch ids 随后还会变化。
5. 追加 shared experts，必要时改成 DeepEP interleaved layout。
6. 按 scaling ownership 设置 shared weights。

因此，同一轮 forward 至少可能同时存在 logical ids、physical routed ids、recorder ids、带 shared slot 的最终 dispatch ids 四份视图。

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

这段也解释了为什么 LP 求解不能简单放进 torch.compile：它包含 EP all-reduce。空 token rank 也不能跳过 collective；通用 `empty_topk_output(layer_id=...)` 会主动让 LP solver 处理空 tensor，否则 DP-attention 下可能死锁。

DeepEP shared slot 还有一个未完全收口的 correctness 边界：默认 post-MoE scaling 要把 shared weight 设为 `1/routed_scaling_factor`，Aiter 因 routed scaling 已预折叠而设为 `1.0`。源码注释明确指出 ModelOpt NVFP4、CUTLASS/TRTLLM-routed FP8 等其他预折叠 family 理论上也应为 `1.0`，但当前修复刻意只覆盖 Aiter。这里不能用“量化只改变 GEMM”来解释。

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

DeepEP 的 `dispatch_a` 把内部状态暂存到 `_dispatch_intermediate_state`，`dispatch_b` 取出后删除；combine 也是同样模式。外层 stage 相同，内部 normal 与 low-latency 的数据形状却不同。

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

| DeepEP 模式 | dispatch output 关键字段 | 同步/状态 | combine 关键依赖 |
|---|---|---|---|
| normal | recv hidden、optional scale、recv ids/weights、每 expert 接收数 | layout + handle + event | member `handle`，normal combine config |
| low-latency | packed hidden、optional scale、原 ids/weights、`masked_m/expected_m` | packed count + handle + event/hook | top-k ids/weights、overlap stream/signals |

`deepep_mode=auto` 会按 batch 类型选择实现；从 normal 切到 low-latency 时还会清理 low-latency buffer。通信输出 dtype 可是 BF16、FP8、INT8 或 NVFP4，并由量化配置与硬件共同校正。

## 交互矩阵

| 读写方 | 写入 | 读取 |
|--------|------|------|
| 模型 MoE block | `router_logits`、`topk_output` | `hidden_states`、`forward_batch` |
| `TopK.forward_*` | format-specific `TopKOutput` | runner backend、LoRA/FP4、config |
| `select_experts/post-process` | logical/physical/recorder ids、weights | logits、EPLB info、padding、shared config |
| EPLB dispatch | physical replica ids | logical ids、metadata、可选 LP probability |
| `BaseDispatcher` | `DispatchOutput`、combined hidden | hidden、top-k、hook、quant config |
| `quant_method.apply` | `CombineInput` | expert 分组 hidden、expert weights |
| `EPLBManager` | expert location metadata | distribution recorder output |

## 状态自检

排查时按这个顺序问：

1. `router_logits` 形状是否是 `[tokens, experts]`。
2. 当前 carrier 是 standard、packed、bypassed 还是 Triton-kernel。
3. logical、physical、recorder、final dispatch ids 是否被混为一谈。
4. padded id 与 weight 是否按平台和环境变量正确处理。
5. shared expert append/remap 后 weights 是否符合 scaling ownership。
6. dispatch output format、dtype 与 optional scale 是否匹配 runner。
7. `run_moe_core` 是否用了预期量化 method。
8. combine 的 handle/event/hook 是否与本轮 dispatch 成对。
9. TP/EP/LP collective 是否所有 rank 都参与。

## 运行验证

MoE 数据流的轻量验证要覆盖 format 选择、post-process 顺序、scaling ownership、logical/physical/recorder ids、DeepEP normal/LL ABI 和 collective participation。

```powershell
rg -n 'class StandardTopKOutput|class BypassedTopKOutput|def topk_ids_logical_to_physical|def _topk_ids_logical_to_physical|def select_experts|def forward_impl|def dispatch_a|def dispatch_b|def combine_a|def combine_b|run_moe_core|TopKOutput' sglang/python/sglang/srt/layers/moe/topk.py sglang/python/sglang/srt/eplb/expert_location_dispatch.py sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py
```

如果上游改动后命中位置变化，优先重新检查三组契约：format 的生产者/消费者、logical→physical→final ids 的演化、dispatch A/B 与 combine A/B 的状态和 dtype 是否成对。
