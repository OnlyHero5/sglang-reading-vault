---
title: "MoE · 核心概念"
type: concept
framework: sglang
topic: "MoE"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-12
---
# MoE · 核心概念

## 读者任务

这篇先建立对象边界。MoE 不是一条“gate 后固定生成两个 tensor”的流水线，而是一组会随 backend 改变的契约：模型层产生 logits，`TopK` 选择立即 materialize、延迟 materialize 或直接生成 kernel routing data；post-process 再处理 logical/physical ids、shared expert 与 padding；dispatcher、runner、combine 共同约定数据格式和 scaling 归属。

## 先建立模型

| 类比 | 源码对象 | 失效边界 |
|------|----------|----------|
| 分诊台 | `gate` / `router_logits` | 不解释 expert GEMM kernel |
| 转诊单 | `TopKOutput` | 可能是显式 ids/weights，也可能是 bypassed 或 Triton routing data |
| 转运车 | `BaseDispatcher` / `DeepEPDispatcher` | 默认职责是搬运，但 hook 可改写 hidden、top-k、dispatch/combine 对象 |
| 专科诊室 | `quant_method.apply` / MoeRunner | 不负责跨 rank all-to-all |
| 处方合并 | `dispatcher.combine` | 不重新选择 expert |

读源码时先问：这段代码是在做“选择专家”，还是“把 token 送到专家”，还是“专家计算”，还是“把结果合回来”。混在一起读，MoE 很快会变成一团 backend 名词。

## 模型侧调用：gate、top-k、experts 三个动作

典型 MoE 层先用 gate 算 router logits，再调用 `topk`，最后把 hidden states 和 top-k 输出交给 expert 层。

```python
# 来源：sglang/python/sglang/srt/models/bailing_moe.py L337-L341
    def _forward_router_experts(self, hidden_states: torch.Tensor):
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        return self.experts(hidden_states, topk_output)
```

DeepEP 路径多传了 `forward_batch` 和 `ExpertLocationDispatchInfo`，但核心仍是 gate → top-k → experts。

```python
# 来源：sglang/python/sglang/srt/models/bailing_moe.py L389-L413
    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        shared_output = None
        forward_mode = forward_batch.forward_mode
        if is_non_idle_and_non_empty(forward_mode, hidden_states):
            router_logits = self.gate(hidden_states)
            if self.num_shared_experts > 0:
                shared_output = self.shared_experts(hidden_states)

            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
```

这里可以看出 DeepEP 不是替代 gate，也不是替代专家计算，而是影响 top-k 后的 expert location 和 dispatcher。

## `TopKOutput` 是 MoE 层的路由契约

standard top-k 输出有权重、expert id、原始 logits；它只是协议的一种实现，并非所有 dispatcher、runner、combine 的共同输入。

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

还有 bypassed 格式：它延迟 materialize top-k，直到某个 runner 需要显式 `topk_ids/topk_weights` 时再调用 `to_standard`。

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

协议实际有三种公开 format，另外还有保持 `STANDARD` format 的实验性 packed carrier。CUDA 路径会依据显式配置、runner backend、LoRA 与 FP4 条件决定 format，而不是等到“格式不匹配”才被动降级。

```python
# 来源：sglang/python/sglang/srt/layers/moe/topk.py L254-L257
class TopKOutputFormat(IntEnum):
    STANDARD = auto()
    TRITON_KERNEL = auto()
    BYPASSED = auto()
```

```python
# 来源：sglang/python/sglang/srt/layers/moe/topk.py L287-L295
class StandardTopKOutputPacked(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor
    packed_topk_ids: torch.Tensor

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.STANDARD
```

```python
# 来源：sglang/python/sglang/srt/layers/moe/topk.py L301-L310
class TritonKernelTopKOutput(NamedTuple):
    """Triton kernel top-k output format."""

    routing_data: RoutingData
    gather_indx: GatherIndx
    scatter_indx: ScatterIndx

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.TRITON_KERNEL
```

```python
# 来源：sglang/python/sglang/srt/layers/moe/topk.py L472-L475
        if self.topk_config.output_format is not None:
            output_format = self.topk_config.output_format
        elif get_moe_runner_backend().is_triton_kernels():
            output_format = TopKOutputFormat.TRITON_KERNEL
```

实验性 SGL-TRTLLM runner 在启用 LoRA 时选择 standard，否则选择 bypassed；FlashInfer TRTLLM 及特定 MXFP4 条件也会选择 bypassed：

```python
# 来源：sglang/python/sglang/srt/layers/moe/topk.py L490-L495
        elif get_moe_runner_backend().is_flashinfer_trtllm() or (
            get_moe_runner_backend().is_flashinfer_mxfp4() and not self.is_fp4_experts
        ):
            output_format = TopKOutputFormat.BYPASSED
        else:
            output_format = TopKOutputFormat.STANDARD
```

因此，piecewise CUDA Graph 中其他 format 调用 `forward_impl`，只说明选择了通用执行函数；是否真的 graph break，还要结合 piecewise context 与整层 torch op 注册机制判断。

## `FusedMoE` 固定了执行骨架

无论底层 runner 如何变化，`forward_impl` 的外层结构稳定：dispatch、core、combine、必要时 all-reduce。稳定的是控制骨架，不是各阶段的 tensor ABI。

```python
# 来源：sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py L1134-L1150
    def forward_impl(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        origin_hidden_states_dim = hidden_states.shape[-1]
        assert self.quant_method is not None

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

```python
# 来源：sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py L1156-L1159
        if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states
```

注意 `run_moe_core` 本身把控制权交给量化 method。某些 runner 可能内部消费 router logits 或要求特殊 top-k format，所以“选择在哪一步 materialize”与“GEMM 在哪里执行”要分开看。

```python
# 来源：sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py L1178-L1183
    def run_moe_core(self, dispatch_output: DispatchOutput) -> CombineInput:
        # TODO: consider using symmetric memory
        return self.quant_method.apply(
            layer=self,
            dispatch_output=dispatch_output,
        )
```

## dispatcher 是 MoE 的搬运协议

`BaseDispatcher` 规定 dispatch/combine 两个抽象动作并提供 hook。hook 不只是旁路观测：pre-dispatch 可替换 `hidden_states/topk_output`，post-dispatch、pre-combine、post-combine 也能替换阶段输出；量化 method 还会向 dispatcher 写入 quant config。因此“dispatcher 默认承担搬运”是职责描述，不是不可越过的数据不变量。

```python
# 来源：sglang/python/sglang/srt/layers/moe/token_dispatcher/base.py L279-L332
    @abstractmethod
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        pass

    def _dispatch_with_hook(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        if self._pre_dispatch_hooks is not None:
            hidden_states, topk_output = self._pre_dispatch_hooks(
                self, hidden_states, topk_output
            )
        dispatch_output = self._original_dispatch_func(
            hidden_states=hidden_states, topk_output=topk_output
        )
        if self._post_dispatch_hooks is not None:
            dispatch_output = self._post_dispatch_hooks(self, dispatch_output)
        return dispatch_output

    def _override_dispatch_func(self) -> None:
        if self._original_dispatch_func is None:
            self._original_dispatch_func = self.dispatch
            self.dispatch = self._dispatch_with_hook

    @abstractmethod
    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        pass

    def _combine_with_hook(self, combine_input: CombineInput) -> torch.Tensor:
        if self._pre_combine_hooks is not None:
            combine_input = self._pre_combine_hooks(self, combine_input)
        hidden_states = self._original_combine_func(combine_input=combine_input)
        if self._post_combine_hooks is not None:
            hidden_states = self._post_combine_hooks(self, hidden_states)
        return hidden_states

    def _override_combine_func(self) -> None:
        if self._original_combine_func is None:
            self._original_combine_func = self.combine
            self.combine = self._combine_with_hook

    def register_pre_dispatch_hook(
        self,
        hook: Callable[
            [BaseDispatcher, torch.Tensor, TopKOutput],
            Optional[Tuple[torch.Tensor, TopKOutput]],
        ],
    ) -> _RemovableDispatcherHandle:
        if self._pre_dispatch_hooks is None:
            self._pre_dispatch_hooks = _PreDispatchHooks()
            self._override_dispatch_func()
        handle = self._pre_dispatch_hooks.register_hook(hook)
        return handle
```

这段说明 hook 覆盖整个 dispatch/combine 边界。通常它不重新运行 gate，但它完全可以改写承载路由决策的 `TopKOutput`，排障时不能把 dispatcher 当成透明管道。

## DeepEP 是 dispatcher 的一种状态机

DeepEP 把 dispatch 拆成 A/B 两段，combine 也拆成 A/B 两段，并用 `_stage` 强制顺序。这是排查 DeepEP 阶段错乱的关键。

```python
# 来源：sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py L897-L951
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        if self._deepep_dispatch_hooks is not None:
            self._deepep_dispatch_hooks(self)
        ret = self.dispatch_b()
        return ret

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

状态机之下还有两套不同 ABI。normal dispatch 先计算 layout，再返回按 expert 聚合的 hidden、可选 scale、接收后的 ids/weights 和 `num_recv_tokens_per_expert`；low-latency dispatch 则返回 packed hidden、`masked_m`、`expected_m`，并以 event 或 recv hook 完成同步。`auto` 会按当前 batch 是 extend 还是 decode 解析到不同实现，切换到 low-latency 前还可能清理 buffer。不要把两者理解为只改一个性能参数。

DeepEP 通信 dtype 也受 quant config 和硬件约束：GPU 上请求 INT8 会转 FP8，NPU 上请求 FP8 会转 INT8；low-latency 可直接让通信层输出 FP8/NVFP4 及 scale。由此可见，量化契约已经进入 dispatcher，而不是只存在于 expert GEMM。

## EPLB placement 与 replica dispatch 是两层机制

EPLB placement 根据统计窗口重排 logical expert 到 physical expert 的 metadata；dispatch algorithm 再决定一个 logical id 的哪份 physical replica 接收当前 token。`static` 查固定映射，`dynamic/fake` 随机选择有效 replica，`lp` 使用求解出的概率。前者不是逐 token 重算，后者却可以逐 token 选择。

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

这里还有一个源码契约漂移：`ExpertLocationDispatchInfo` 的字段注解仍是 `Literal["static", "random"]`，但 ServerArgs 与运行分支实际接受 `static/dynamic/fake/lp`。调用方应以运行分支为准，同时把这个漂移视为升级时的高风险审计点。

所以排查 expert 倾斜时要分三层看：gate 选择了哪个 logical expert，placement 为它布置了哪些 physical replicas，以及 dispatch algorithm 本次选中了哪一份。

## 运行验证

这篇的验证目标不是只找五个类名，而是核对五份契约：TopK format、logical/physical id、dispatcher output format、quant/scaling ownership、placement/dispatch algorithm。

```powershell
rg -n 'BailingMoE|class TopK|def select_experts|class FusedMoE|def forward_impl|class BaseDispatcher|class DeepEPDispatcher|def dispatch_a|def dispatch_b|def combine_a|def combine_b|class ExpertLocationDispatchInfo|def topk_ids_logical_to_physical' sglang/python/sglang/srt/models/bailing_moe.py sglang/python/sglang/srt/layers/moe/topk.py sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py sglang/python/sglang/srt/layers/moe/token_dispatcher/base.py sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py sglang/python/sglang/srt/eplb/expert_location_dispatch.py
```

如果上游改版后这个命令的命中位置大幅变化，优先重新核对两件事：`TopKOutput` 的格式是否仍能覆盖 standard / bypassed / Triton kernel 路径，以及 EPLB 是否仍在 dispatch 前把 `topk_ids` 从 logical expert 转成 physical expert。
