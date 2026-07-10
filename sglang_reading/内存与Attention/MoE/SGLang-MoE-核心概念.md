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
updated: 2026-07-10
---
# MoE · 核心概念

## 读者任务

这篇先建立对象边界。你要把 MoE 层看成一个 token 生命周期，而不是一堆后端名字：模型层负责产生 `router_logits`，`TopK` 负责把 logits 变成 `topk_ids/topk_weights`，dispatcher 负责把 token 送到专家，`MoeRunner` 负责本地专家 GEMM，combine 负责按权重还原到原 token 顺序。

## 先建立模型

| 类比 | 源码对象 | 失效边界 |
|------|----------|----------|
| 分诊台 | `gate` / `router_logits` | 不解释 expert GEMM kernel |
| 转诊单 | `topk_ids` / `topk_weights` | 不等于最终 hidden state |
| 转运车 | `BaseDispatcher` / `DeepEPDispatcher` | 不决定 expert 选择，只执行搬运 |
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

标准 top-k 输出只有三个字段：权重、专家 id、原始 logits。这个小结构是后续 dispatcher、expert runner、combine 的共同输入。

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

这解释了 piecewise CUDA Graph 的常见分叉：有些路径接受 standard，有些路径接受 bypassed；格式不匹配时只能回到普通 `forward_impl`。

## `FusedMoE` 固定了执行骨架

无论底层是 Triton、DeepGEMM、FlashInfer 还是未量化 runner，`forward_impl` 的结构都很稳定：dispatch、core、combine、必要时 all-reduce。

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

注意 `run_moe_core` 并不自己选择 expert。它只消费 dispatch 后的 expert 分组输入。

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

`BaseDispatcher` 只规定 dispatch/combine 两个抽象动作，并提供 hook。EPLB、统计、overlap 等机制都可以挂在这些 hook 上。

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

这段说明 hook 覆盖的是 dispatch/combine 入口。dispatcher 是搬运协议，不是专家选择算法。

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

## EPLB 改的是映射，不是 gate 数学

EPLB 的核心动作是根据统计重排 logical expert 到 physical expert 的映射；dispatch 前再把 `topk_ids` 转成 physical id。

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

所以排查 expert 倾斜时，要同时看 logical 选择和 physical 放置：前者来自 gate/top-k，后者来自 EPLB metadata。

## 运行验证

这篇的验证目标不是跑完整 MoE 模型，而是确认五个边界仍在源码里各自独立：模型层发起 gate，`TopK` 选择 expert，`FusedMoE` 执行专家计算，dispatcher 负责 token 搬运，EPLB 只改 logical 到 physical 的映射。

```powershell
rg -n 'BailingMoE|class TopK|def select_experts|class FusedMoE|def forward_impl|class BaseDispatcher|class DeepEPDispatcher|def dispatch_a|def dispatch_b|def combine_a|def combine_b|class ExpertLocationDispatchInfo|def topk_ids_logical_to_physical' sglang/python/sglang/srt/models/bailing_moe.py sglang/python/sglang/srt/layers/moe/topk.py sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py sglang/python/sglang/srt/layers/moe/token_dispatcher/base.py sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py sglang/python/sglang/srt/eplb/expert_location_dispatch.py
```

如果上游改版后这个命令的命中位置大幅变化，优先重新核对两件事：`TopKOutput` 的格式是否仍能覆盖 standard / bypassed / Triton kernel 路径，以及 EPLB 是否仍在 dispatch 前把 `topk_ids` 从 logical expert 转成 physical expert。
