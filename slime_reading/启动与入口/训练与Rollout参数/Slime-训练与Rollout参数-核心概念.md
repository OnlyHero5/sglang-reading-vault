---
title: "训练与Rollout参数 · 核心概念"
type: concept
framework: slime
topic: "训练与Rollout参数"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/concept
  - source-reading
updated: 2026-07-12
---
# 训练与Rollout参数 · 核心概念

## 你为什么要读

这篇先建立几个概念，不急着读完整参数列表。你要抓住的是：`arguments.py` 中的 train/rollout/data/algo/reward 参数不是静态配置，而是在运行时打开或替换一段明确的调用链。

## 四类参数

| 类别 | 典型字段 | 运行时含义 |
|------|----------|------------|
| 样本账参数 | `rollout_batch_size`、`n_samples_per_prompt`、`num_steps_per_rollout` | 默认路径决定 prompt group 与 rollout execution 数；训练调度再按唯一 `rollout_id` 分步 |
| 函数路径参数 | `data_source_path`、`rollout_function_path`、`custom_rm_path` | 通过 `load_function` 变成 callable 或 class |
| 算法开关 | `loss_type`、`advantage_estimator`、`use_rollout_logprobs` | 决定 Megatron loss 和 advantage 分支 |
| 后端透传 | `--sglang-*`、HF config 校验、Megatron 默认值 | 把 Slime 语义翻译给 SGLang 与 Megatron |

这四类不要混读。比如 `custom_generate_function_path` 和 `rollout_function_path` 都是 path，但一个替换 per-sample generate，另一个替换整个 rollout 函数。

## 函数路径是最小插件系统

Slime 的插件机制很朴素：CLI 保存一个字符串，运行时用 `load_function` import。

```python
# 来源：slime/utils/misc.py L37-L45
def load_function(path):
    """
    Load a function from a module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
```

这个设计给读者两个结论：

- path 必须能按 Python import 规则找到模块和属性。
- CLI 层只存字符串，真正签名约束来自消费点和 contract tests。

## RolloutManager 是 path 参数的第一道装配线

`data_source_path`、`rollout_function_path`、`eval_function_path`、reward postprocess、samples to train data 都在 RolloutManager 初始化时装配。

```python
# 来源：slime/ray/rollout.py L424-L451
def __init__(self, args, pg):
    configure_logger()

    self.pg = pg
    self.args = args

    rollout_init_handles: list[Any] = []
    if self.args.debug_train_only:
        self.servers: dict[str, Any] = {}
    else:
        init_http_client(args)
        self.servers, rollout_init_handles = start_rollout_servers(args, pg)

    data_source_cls = load_function(self.args.data_source_path)
    self.data_source = data_source_cls(args)

    self.generate_rollout = load_function(self.args.rollout_function_path)
    self.eval_generate_rollout = load_function(self.args.eval_function_path)
    self.custom_reward_post_process_func = None
    if self.args.custom_reward_post_process_path is not None:
        self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
    self.custom_convert_samples_to_train_data_func = None
    if self.args.custom_convert_samples_to_train_data_path is not None:
        self.custom_convert_samples_to_train_data_func = load_function(
            self.args.custom_convert_samples_to_train_data_path
        )
    logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
    logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")
```

这段说明 `rollout_function_path` 和 `eval_function_path` 是“整条 rollout 函数”的替换点，而不是只改 SGLang HTTP 参数。

## 样本账：五种计数不能混用

默认 SGLang 路径中，`rollout_batch_size` 是要保留的有效 prompt group 数，每组启动 `n_samples_per_prompt` 个生成 execution。通常一个 execution 返回一条 Sample，于是训练数据行数等于两者乘积；compact/subagent 路径却可以把一个 execution 展成多条 sibling Sample，它们必须共享同一个 `Sample.rollout_id`。

| 单位 | 含义 | 是否总等于 `rollout_batch_size * n_samples_per_prompt` |
|------|------|-------------------------------------------------------|
| prompt group | 默认动态过滤保留的题目组 | group 数等于 `rollout_batch_size` |
| rollout execution | 一次生成/子轨迹的训练计数身份 | 默认路径通常等于乘积 |
| Sample 行 | converter 展平后的训练记录 | compact 时可多于 execution 数 |
| global batch | 一个训练 step 消费的唯一 rollout id 数 | 不是裸 Sample 行数 |
| micro-batch | token/样本打包后的设备执行单元 | 由 DP schedule、长度和动态 batch 决定 |

```python
# 来源：slime/utils/arguments.py L676-L702
# batch sizes
parser.add_argument(
    "--rollout-batch-size",
    type=int,
    required=True,
    help=(
        "The number of prompts in each rollout step. "
        "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
    ),
)
parser.add_argument(
    "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation"
)

# gbs of the training, note that the gbs is of sample, not of prompts,
# so if you hope to train 1 step for each rollout, the global_bach_size should be set as
# `rollout_batch_size * n_samples_per_prompt`.
reset_arg(parser, "--global-batch-size", type=int, default=None)
parser.add_argument(
    "--num-steps-per-rollout",
    type=int,
    default=None,
    help=(
        "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
        "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
    ),
)
```

validator 仍用默认 rollout 规模公式推导 `global_batch_size`：

```python
# 来源：slime/utils/arguments.py L1911-L1919
if args.num_steps_per_rollout is not None:
    global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
    if args.global_batch_size is not None:
        assert args.global_batch_size == global_batch_size, (
            f"global_batch_size {args.global_batch_size} is not equal to "
            f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
            f"// num_steps_per_rollout {args.num_steps_per_rollout}"
        )
    args.global_batch_size = global_batch_size
```

真正调度时，scheduler 明确把它解释为每步 rollout 数，而不是训练 Sample 行数：

```python
# 来源：slime/utils/dp_schedule.py L100-L110
        global_batch_size: number of rollouts (NOT training samples) per
            training step. Number of training steps =
            ``num_rollouts // global_batch_size``; trailing rollouts whose
            samples don't fit are dropped.
        rollout_indices: rollout id for each sample (``samples[i].index``).
            Samples sharing the same id are kept together in one step.

    Returns:
        ``(partitions, micro_batch_indices, num_microbatches, global_batch_sizes)``.
        ``global_batch_sizes[s]`` = rollout count for step s (constant
        ``global_batch_size`` for every step).
```

例子：默认路径下 `64 × 4 ÷ 2 = 128` 个 rollout execution/step。若每个 execution compact 成 3 条 sibling，训练表面上可能有 768 行，但仍是 256 个唯一 rollout id、每步 128 个 rollout；不能把 `global_batch_size` 改成 384。

## 生成插件有两层

第一层是整条 rollout 函数：

```python
# 来源：slime/utils/arguments.py L327-L340
parser.add_argument(
    "--rollout-function-path",
    type=str,
    default="slime.rollout.sglang_rollout.generate_rollout",
    help=(
        "Path to the rollout generation function."
        "You should use this model to create your own custom rollout function, "
        "and then set this to the path of your custom rollout function. "
        "The signature of the function should be "
        "`def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput`"
        "and within the output sample, you should at least set `tokens`, `response_length`, `reward` "
        "and `status`."
    ),
)
```

第二层是默认 SGLang rollout 内部的 per-sample generate：

```python
# 来源：slime/rollout/sglang_rollout.py L249-L261
with state.dp_rank_context() as _:
    # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
    custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

    if custom_func_path is not None:
        custom_generate_func = load_function(custom_func_path)
        # if signature has evaluation, pass evaluation
        if "evaluation" in inspect.signature(custom_generate_func).parameters:
            sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
        else:
            sample = await custom_generate_func(args, sample, sampling_params)
    else:
        sample = await generate(args, sample, sampling_params)
```

这解释了一个常见排障点：如果你只想改“单个样本怎么调用工具/多轮生成”，用 `custom_generate_function_path`；如果你要改采样循环、filter、abort、返回对象，改 `rollout_function_path`。

compact generate 返回 `list[Sample]` 时还有额外边界：siblings 必须共享 `rollout_id`；当前 fanout E2E 明确避免同时启用 `group_rm`，因为 group 路线假设返回已经是扁平 `list[Sample]`，再嵌套一层会破坏 RM 输入契约。不要从“两个功能分别可用”推导它们的组合一定可用。

GRPO reward normalization 也要重新审视：默认实现只在总行数等于 `rollout_batch_size * n_samples_per_prompt` 时按固定组宽 reshape；可变 fanout 会落入非均匀 fallback，未必仍按原 prompt 分组。当前 fanout E2E 因此显式提供按 `group_index` 归一化的 custom reward postprocess。

## Reward 和 filter 是不同层级

Reward 负责给 sample 赋分；dynamic filter 负责判断一个 prompt group 是否进入本轮有效数据；sample filter 只影响 loss 参与。

```python
# 来源：slime/rollout/sglang_rollout.py L394-L431
# instantiate data filters
dynamic_filter = (
    load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
)

metric_gatherer = MetricGatherer()

# target_data_size is the total number of valid samples to get
target_data_size = args.rollout_batch_size

data = []
all_data = []
do_print = True
pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
while len(data) < target_data_size:
    while state.remaining_batch_size < target_data_size:
        # get samples from the buffer and submit the generation requests.
        samples = data_source(args.over_sampling_batch_size)
        state.submit_generate_tasks(samples)

    # wait for the generation to finish
    done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        group: list[Sample] = task.result()

        if do_print:
            sample = group[0][0] if isinstance(group[0], list) else group[0]
            logger.info(
                f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
            )
            do_print = False

        assert len(group) == args.n_samples_per_prompt
        all_data.append(group)

        dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
        if not dynamic_filter_output.keep:
            metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
```

`rollout_sample_filter_path` 在生成完成后运行，它修改 `Sample.remove_sample`，不是重新决定 prompt group 是否补样。

## 算法参数最终由 loss 消费

`advantage_estimator` 和 `loss_type` 的最终消费点在 Megatron loss，而不是参数定义处。

```python
# 来源：slime/backends/megatron_utils/loss.py L715-L764
if args.custom_advantage_function_path is not None:
    custom_adv_fn = load_function(args.custom_advantage_function_path)
    custom_adv_fn(args, rollout_data)
    advantages, returns = rollout_data["advantages"], rollout_data["returns"]

elif args.advantage_estimator in ["grpo", "gspo", "cispo"]:
    rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
    returns = get_grpo_returns(rewards, kl)
    # TODO: is the copy necessary?
    advantages = [r for r in returns]

elif args.advantage_estimator == "ppo":
    old_rewards = rewards
    rewards = []
    kl_coef = -args.kl_coef
    cp_rank = mpu.get_context_parallel_rank()
    for reward, k in zip(old_rewards, kl, strict=False):
        k *= kl_coef
        if cp_rank == 0:
            k[-1] += reward
        rewards.append(k)
    advantages, returns = get_advantages_and_returns_batch(
        total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
    )

elif args.advantage_estimator == "reinforce_plus_plus":
    rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
    returns = get_reinforce_plus_plus_returns(
        rewards=rewards,
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
        kl_coef=args.kl_coef,
        gamma=args.gamma,
    )
    advantages = [r for r in returns]

elif args.advantage_estimator == "reinforce_plus_plus_baseline":
    rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
    advantages = get_reinforce_plus_plus_baseline_advantages(
        rewards=rewards,
        kl=kl,
        loss_masks=loss_masks,
        kl_coef=args.kl_coef,
    )
    returns = advantages

else:
    raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")
```

`loss_type=custom_loss` 也在这里接管：

```python
# 来源：slime/backends/megatron_utils/loss.py L1264-L1274
match args.loss_type:
    case "policy_loss":
        func = policy_loss_function
    case "value_loss":
        func = value_loss_function
    case "sft_loss":
        func = sft_loss_function
    case "custom_loss":
        func = load_function(args.custom_loss_function_path)
    case _:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
```

## 后端透传是带边界的

SGLang 原生参数被 Slime 加上 `--sglang-` 前缀，但不是无限透传。`skipped_args` 表示 Slime 自己管理的 ownership 边界，例如模型路径、TP、端口、分布式初始化。

```python
# 来源：slime/backends/sglang_utils/arguments.py L35-L63
def add_sglang_arguments(parser):
    """
    Add arguments to the parser for the SGLang server.
    """
    parser = add_sglang_router_arguments(parser)
    parser.set_defaults(router_balance_abs_threshold=10, router_balance_rel_threshold=1.2)
    parser.add_argument("--sglang-server-concurrency", type=int, default=512)

    old_add_argument = parser.add_argument

    skipped_args = [
        "model_path",
        "config",
        "trust_remote_code",
        "random_seed",
        # memory
        "enable_memory_saver",
        # distributed
        "tp_size",
        "port",
        "nnodes",
        "node_rank",
        "dist_init_addr",
        "gpu_id_step",
        "base_gpu_id",
        "nccl_port",
        "skip_server_warmup",
        "enable_return_routed_experts",
    ]
```

读后端参数时要先问：这个字段归 Slime 管，还是透传给 SGLang/Megatron。

## 复盘

阅读训练与 Rollout 参数时，建议按以下顺序：

1. 先区分参数是样本账、函数路径、算法开关还是后端透传。
2. 对 path 参数，直接找 `load_function` 消费点。
3. 对 batch 参数，依次算 prompt group、rollout execution、Sample 行、global batch 和 micro-batch，再看 validator 与 scheduler。
4. 对算法参数，看 Megatron loss 的最终分支，不要停在 CLI help。
5. 对后端参数，看 ownership 边界：Slime 管哪些，透传哪些。

下一篇 [[Slime-训练与Rollout参数-源码走读]] 沿一次 rollout 到 train 的真实路径展开。
