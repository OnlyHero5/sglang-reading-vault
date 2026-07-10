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
updated: 2026-07-10
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
# 来源：slime/backends/megatron_utils/model_provider.py L61-L123
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
# 来源：slime/backends/megatron_utils/model_provider.py L125-L240
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
# 来源：slime/backends/megatron_utils/model_provider.py L25-L58
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
# 来源：slime/backends/megatron_utils/model_provider.py L245-L286
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

## setup 是装配模型、optimizer、scheduler

`setup_model_and_optimizer` 的职责是把 provider 交给 Megatron `get_model`，再从 args 构造 `OptimizerConfig`、Megatron optimizer 和 LR scheduler。

```python
# 来源：slime/backends/megatron_utils/model.py L270-L318
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

## scheduler 用估算步数，但真实进度由 step 更新

`train_iters` 来自 rollout 总数、batch size、采样数的估算。动态采样或过滤会让真实 step 有漂移，但 scheduler 后续通过 `opt_param_scheduler.step(increment=step_global_batch_size)` 追踪实际消耗。

```python
# 来源：slime/backends/megatron_utils/model.py L182-L235
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
# 来源：slime/backends/megatron_utils/model.py L968-L1007
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

## forward_only 是无梯度特征提取通道

`forward_only` 把模型切到 eval，跑 Megatron pipeline forward-only，用 post-forward callback 生成 `log_probs`、`entropy` 或 `values`，只在 pipeline last stage 聚合输出。

源码入口：来源：slime/backends/megatron_utils/model.py L344-L506

它是 [[Slime-Advantage计算]] 和 [[Slime-Policy-Loss]] 的前置通道：advantage 阶段用它收集 old/ref/teacher logprob 和 critic values，policy backward 则走 `train_one_step`。
