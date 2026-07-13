---
title: "模型初始化 · 核心概念"
type: concept
framework: slime
topic: "模型初始化"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/concept
  - source-reading
updated: 2026-07-13
---
# 模型初始化 · 核心概念

## 你为什么要读

本篇先建立模型：初始化不是单纯 `load_checkpoint`。它先决定模型结构，再决定哪些参数训练，再装 optimizer/scheduler，最后把 checkpoint 状态加载进去。

## provider 是模型图纸

`get_model_provider_func(args, role)` 返回一个 callable，Megatron 的 `get_model` 会多次调用它，按 PP/VPP 构建不同 model chunks。

provider 有三条路径：

| 路径 | 触发条件 | 适合场景 |
|------|----------|----------|
| custom provider | `custom_model_provider_path` | 多模态、自定义 Megatron 模型 |
| Megatron Bridge | `megatron_to_hf_mode == "bridge"` | 从 HF checkpoint 自动构建 Megatron provider |
| legacy/MCore GPTModel | 默认 | 常规 Megatron GPTModel |

源码证据：

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/model_provider.py L61-L123
if getattr(args, "custom_model_provider_path", None):
    ...
    return wrapped_model_provider

if args.megatron_to_hf_mode == "bridge":
    bridge = patch_auto_bridge_hf_config(AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True))
    provider = bridge.to_megatron_provider(load_weights=False)
    ...
    return provider.provide
```

默认路径才会显式构造 `GPTModel`：

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/model_provider.py L125-L240
config = core_transformer_config_from_args(args)
...
kwargs = {
    "config": config,
    "transformer_layer_spec": transformer_layer_spec,
    "vocab_size": args.padded_vocab_size,
    "max_sequence_length": args.max_position_embeddings,
    "pre_process": pre_process,
    "post_process": post_process,
    ...
}
model = GPTModel(**kwargs)
```

## actor 和 critic 的差别在 output head

actor 保留 LM head，输出 token logits。critic 在 `post_process` stage 把 `output_layer` 换成 `LinearForLastLayer(hidden_size, 1)`，输出每个 response token 的 value。

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/model_provider.py L25-L58
class LinearForLastLayer(torch.nn.Linear):
    ...
    def forward(self, input_, weight=None, runtime_gather_output=None):
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None
```

critic head 只在 `post_process and role == "critic"` 时替换。非 last PP stage 不持有最终输出层。

## freeze 是 provider 外层包装

冻结或只训练部分参数发生在 model 构建之后、返回给 Megatron 之前。`get_model_provider_func` 会把真实 provider 包一层，再调用 `freeze_model_params`。

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/model_provider.py L245-L286
def wrap_model_provider_with_freeze(original_provider, args):
    def wrapped_provider(...):
        model = original_provider(**provider_kwargs)
        freeze_model_params(model, args)
        return model
    return wrapped_provider

def freeze_model_params(model, args):
    if getattr(args, "only_train_params_name_list", None):
        for name, param in model.named_parameters():
            param.requires_grad = False
            for pattern in args.only_train_params_name_list:
                if re.search(pattern, name):
                    param.requires_grad = True
                    break
```

参数层禁止同时设置 allowlist 和 blocklist：

源码入口：来源：slime/utils/arguments.py L1977-L1978

正则本身没有预编译或命中数校验：非法 pattern 会在各 rank 构模时抛错；allowlist 一个都没命中会把全部参数冻结，blocklist 一个都没命中则静默不冻结。出版级排障不能只确认参数“传进来了”，还要打印 trainable 参数数量和名称样例。

## setup 是装配模型、optimizer、scheduler

`setup_model_and_optimizer` 的职责是把 provider 交给 Megatron `get_model`，再从 args 构造 `OptimizerConfig`、Megatron optimizer 和 LR scheduler。

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/model.py L270-L318
assert not args.moe_use_upcycling
assert args.load is not None or args.pretrained_checkpoint is not None

model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)
...
optimizer = get_megatron_optimizer(
    config=config,
    model_chunks=model,
    use_gloo_process_groups=args.enable_gloo_process_groups,
)
opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
return model, optimizer, opt_param_scheduler
```

`use_stateless_adam` 是特殊 optimizer 路径，只支持 Adam，且要求不保存 optimizer state。

源码入口：来源：slime/backends/megatron_utils/model.py L304-L316

这里的 setup 断言写成 `load is not None or pretrained_checkpoint is not None`，但收口函数随后无条件调用 Slime `load_checkpoint`，该 loader 读取并校验 `args.load`。因此当前完整初始化实际上仍要求最终 `args.load` 指向存在且非空的目录；只设置 `pretrained_checkpoint` 不能由这条链路单独完成加载。

## scheduler 用估算步数，但真实进度由 step 更新

`train_iters` 来自 rollout 总数、batch size、采样数的估算。动态采样或过滤会让真实 step 有漂移，但 scheduler 后续通过 `opt_param_scheduler.step(increment=step_global_batch_size)` 追踪实际消耗。

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/model.py L182-L235
args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
if args.lr_decay_iters is None:
    args.lr_decay_iters = args.train_iters
lr_decay_steps = args.lr_decay_iters * args.global_batch_size
...
opt_param_scheduler = OptimizerParamScheduler(...)
```

## initialize 是 setup + checkpoint load

`initialize_model_and_optimizer` 是本专题的收口函数。它构建模型与 optimizer，设置 role，判断 critic head 是否要重置，加载 checkpoint，然后返回 iteration。

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/model.py L968-L1007
model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
model[0].role = role
reinit_critic_output_layer = _critic_output_layer_needs_reinit(args, model, role)
clear_memory()
iteration, _ = load_checkpoint(
    model,
    optimizer,
    opt_param_scheduler,
    checkpointing_context={},
    skip_load_to_model_and_opt=False,
)
if reinit_critic_output_layer:
    _reinitialize_critic_output_layer(args, model)
    if (args.fp16 or args.bf16) and optimizer is not None:
        optimizer.reload_model_params()
```

checkpoint 分流还要再拆一层：Megatron checkpoint 交给 Megatron loader，HF 目录仅在 Bridge 模式下由 `AutoBridge.load_hf_weights` 加载；HF 路径返回 iteration 0，不恢复 optimizer/scheduler/RNG 状态。仓库 loader 在 import 时还 monkey-patch 了 ShardedTensor 元数据验证以换取大模型加载速度，这意味着跨 rank shard 正确性更多依赖上游 checkpoint 生产与部署纪律。

critic reinit 不是通用“任意 actor checkpoint 转 critic”保证：它只在解析出的 checkpoint 目录存在 `.metadata` 时检查，且重置发生在 `load_checkpoint` 成功之后。旧格式、HF Bridge 或严格 loader 先因 shape mismatch 失败的场景，不会被这段 post-load 重置自动挽救。

## forward_only 是无梯度特征提取通道

`forward_only` 把模型切到 eval，跑 Megatron pipeline forward-only，用 post-forward callback 生成 `log_probs`、`entropy` 或 `values`，只在 pipeline last stage 聚合输出。

源码入口：来源：slime/backends/megatron_utils/model.py L344-L506

它是 [[Slime-Advantage计算]] 和 [[Slime-Policy-Loss]] 的前置通道：advantage 阶段用它收集 old/ref/teacher logprob 和 critic values，policy backward 则走 `train_one_step`。

当前实现没有 `try/finally`：custom before-logprob hook、pipeline forward 或结果聚合抛异常时，模型可能停在 eval mode，进度条也未关闭。动态 batch 的恢复又用 `zip(strict=False)` 写入固定长度数组，不检查 index 是否唯一、越界或数量完全相等；“恢复原序”是输入 permutation 完整时的条件性结论。
