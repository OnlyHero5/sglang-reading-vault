---
title: "训练与Rollout参数 · 排障指南"
type: troubleshooting
framework: slime
topic: "训练与Rollout参数"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-13
---
# 训练与Rollout参数 · 排障指南

## 你为什么要读

Slime 参数会经过 SGLang、Megatron、Slime 自身解析与后续校验，同一个最终值可能来自默认、派生或显式覆盖。本文从参数来源、转换时机和第一个消费者入手，解释自定义 path、batch 规模和 debug 模式为何会在运行前就改变主线。

这篇按症状排障。每个问题都落到一个源码入口或 contract test。

## 我的自定义 path 为什么 import 失败

Slime 的 path 加载只做一件事：按最后一个点拆成 module 和 attribute。

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

排查顺序：

- path 必须是 `pkg.mod.attr`，不是文件路径，也不是 `pkg/mod.py:fn`。
- 先在同一环境跑 `python -c "from pkg.mod import attr"`。
- 再跑对应的 `tests/plugin_contracts`，确认签名和返回值。

## `rollout_function_path` 和 `custom_generate_function_path` 怎么选

如果你要替换整个 rollout 生命周期，用 `rollout_function_path`。如果只想改单个样本如何生成，用 `custom_generate_function_path`。

整条 rollout 函数的签名来自 CLI help：

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

per-sample generate 的消费点在默认 rollout 内部：

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

判断规则：

- 要改数据源、filter、abort、返回对象：选 `rollout_function_path`。
- 要改工具调用、多轮对话、单样本 HTTP 行为：选 `custom_generate_function_path`。

## 为什么 eval 用了 rollout 的生成函数

因为 validate 中 `eval_function_path=None` 会继承 `rollout_function_path`。

```python
# 来源：slime/utils/arguments.py L1908-L1909
if args.eval_function_path is None:
    args.eval_function_path = args.rollout_function_path
```

如果 eval 需要单独逻辑，显式传 `--eval-function-path`。如果只是 eval dataset 中个别样本需要特殊 generate，可以通过 per-sample `generate_function_path` 覆盖。

## `global_batch_size` 为什么和我写的不一致

先看是否传了 `--num-steps-per-rollout`。只要传了，validator 会用默认路径的 rollout execution 规模反推 `global_batch_size`，并要求显式值一致。

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

例子：默认路径下 `64 × 4 ÷ 2 = 128` 个 rollout execution/step。如果你显式传 256，会在 validate 阶段失败。若 compact rollout 把一个 execution 展成多条 sibling Sample，仍按共享的 `rollout_id` 计一次；不要拿展开行数覆盖 128。

## dynamic filter 和 sample filter 为什么表现不同

dynamic filter 在采样过程中决定 prompt group 是否进入有效 batch；sample filter 在生成结束后修改 loss 参与。

```python
# 来源：slime/rollout/sglang_rollout.py L429-L439
dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
if not dynamic_filter_output.keep:
    metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
    state.remaining_batch_size -= 1
    continue

# add the samples to the data
# NOTE: here we have not stored all the unused samples back to the data buffer.
if len(data) < target_data_size:
    data.append(group)
    pbar.update(args.n_samples_per_prompt)
```

```python
# 来源：slime/rollout/sglang_rollout.py L456-L460
# reset the global state to prevent effects on the next rollout or eval.
state.reset()
if args.rollout_sample_filter_path is not None:
    filter_func = load_function(args.rollout_sample_filter_path)
    filter_func(args, data)
```

结论：

- dynamic filter drop 后会继续补样，直到保留 `rollout_batch_size` 个 group。
- sample filter 不补样，它只改变后续 loss 参与。

## compact generate 为什么和 `group_rm` 一起崩

当前 fanout E2E 的注释明确记录：per-sample custom generate 可以返回 sibling `list[Sample]`，非 group RM 路线会对扁平 sibling 列表调用 batched RM；`group_rm` 路线本身又按 group 组织，组合后可能形成 `list[list[Sample]]` 并把 list 当 Sample 访问。处理方式是先保持 `group_rm=False`，或为该嵌套形态实现完整的 custom rollout/RM 协议，不要只叠两个 flag。

同一场景若使用 GRPO 类 reward normalization，还要检查分组：可变 sibling 数不再满足固定 `n_samples_per_prompt` reshape，默认 fallback 可能把更大范围当成一组。应像当前 fanout E2E 一样，按保留下来的 `group_index`/业务身份实现 `custom_reward_post_process_path`，并验证每组均值与标准差。

## custom RM 在 group mode 下签名为什么变了

普通 RM 对单个 sample 打分；`group_rm=True` 时 batched RM 一次接收 samples 列表。

```python
# 来源：slime/rollout/rm_hub/__init__.py L99-L107
async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
```

contract test 也固定了这个边界：

```python
# 来源：slime/tests/plugin_contracts/test_plugin_path_loading_contracts.py L319-L345
def test_custom_rm_path_aligns_with_expected_format():
    path = get_contract_path("CUSTOM_RM_PATH")
    if get_contract_path("GROUP_RM") == "1":
        fn = load_function(path or "plugin_contracts.test_plugin_path_loading_contracts.reference_batched_rm")
        assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "samples")
        rewards = asyncio.run(
            batched_async_rm(
                make_args(
                    group_rm=True,
                    custom_rm_path=path or "plugin_contracts.test_plugin_path_loading_contracts.reference_batched_rm",
                ),
                [make_sample(0), make_sample(1)],
            )
        )
        assert isinstance(rewards, list) and len(rewards) == 2
    else:
        fn = load_function(path or "plugin_contracts.test_plugin_path_loading_contracts.reference_single_rm")
        assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "sample")
        reward = asyncio.run(
            async_rm(
                make_args(
                    custom_rm_path=path or "plugin_contracts.test_plugin_path_loading_contracts.reference_single_rm"
                ),
                make_sample(3),
            )
        )
        assert isinstance(reward, (int, float))
```

这张测试只覆盖普通单 Sample/扁平 samples 签名，不覆盖 compact generate 与 group RM 的嵌套组合。

## custom converter 为什么 contract test 过了，训练仍报 `rollout_ids`

当前 runtime hook contract 只检查 converter 返回 tokens、reward、mask 等旧字段；实际 `_split_train_data_by_dp()` 已无条件用 `rollout_ids` 做按 execution 分步。完全替换 converter 时应至少保留默认转换产生的调度身份与训练字段，并对照 `_split_train_data_by_dp()` 的读取清单。现有 contract 通过只证明 hook 可加载和被调用，不证明返回字典足以完成训练。

## `loss_type=custom_loss` 为什么还在跑默认逻辑

检查两件事：

- 是否真的传了 `--loss-type custom_loss`。
- `custom_loss_function_path` 是否能被训练进程 import。

实际分支在 loss 函数里：

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

如果只是传了 custom path 但没改 `loss_type`，不会进入 custom loss 分支。

## custom advantage 和 `advantage_estimator` 谁优先

custom advantage 优先。源码先检查 `custom_advantage_function_path`，只有没设置时才进入内置 estimator。

```python
# 来源：slime/backends/megatron_utils/loss.py L715-L724
if args.custom_advantage_function_path is not None:
    custom_adv_fn = load_function(args.custom_advantage_function_path)
    custom_adv_fn(args, rollout_data)
    advantages, returns = rollout_data["advantages"], rollout_data["returns"]

elif args.advantage_estimator in ["grpo", "gspo", "cispo"]:
    rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
    returns = get_grpo_returns(rewards, kl)
    # TODO: is the copy necessary?
    advantages = [r for r in returns]
```

custom function 必须原地写入 `rollout_data["advantages"]` 和 `rollout_data["returns"]`。

## 为什么 `reinforce_plus_plus` 要打开 advantage normalization

validate 明确要求这两个 estimator 必须配 `--normalize-advantages`。

```python
# 来源：slime/utils/arguments.py L1798-L1802
if args.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
    assert args.normalize_advantages, (
        "The 'reinforce_plus_plus' and 'reinforce_plus_plus_baseline' advantage estimators "
        "require advantage normalization. Please add `--normalize-advantages` to your command."
    )
```

这是参数层提前挡掉算法配置不变量。

## 为什么 disk/delta 权重同步启动失败

看三个条件：disk 需要 shared dir；delta 必须 disk；delta 不能 colocate；delta 还需要 rollout-host-local checkpoint dir。

```python
# 来源：slime/utils/arguments.py L1980-L2002
# disk-backed sync (full or delta) writes on the trainer and reads on the engines: needs a shared dir
if args.update_weight_transport == "disk" and not args.update_weight_disk_dir:
    raise ValueError(
        "--update-weight-transport=disk requires --update-weight-disk-dir to point at "
        "a filesystem shared between the trainer and the rollout engines."
    )
if args.update_weight_mode == "delta":
    if args.update_weight_transport != "disk":
        raise ValueError(
            "--update-weight-mode=delta requires --update-weight-transport=disk, "
            f"got {args.update_weight_transport!r}."
        )
    if args.colocate:
        raise ValueError(
            "--update-weight-mode=delta is not supported with --colocate. Colocate transfers "
            "weights via CUDA IPC (only a handle crosses processes), so the delta bookkeeping "
            "(snapshot + diff + encode) is pure overhead."
        )
    if not args.update_weight_local_checkpoint_dir:
        raise ValueError(
            "--update-weight-mode=delta requires --update-weight-local-checkpoint-dir "
            "(a rollout-host-local NVMe directory)."
        )
```

排障时不要只看 `update_weight_mode`，要把 transport、colocate、shared dir、本地 checkpoint dir 一起检查。

## 为什么 `--sglang-*` 参数有些生效、有些被忽略

SGLang 参数不是无限透传。Slime 会把大多数 SGLang server args 加 `--sglang-` 前缀，但跳过它自己管理的字段。

```python
# 来源：slime/backends/sglang_utils/arguments.py L45-L63
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

例如 `tp_size` 不应该通过 `--sglang-tp-size` 手工接管，而是由 `rollout_num_gpus_per_engine` 和 PP size 推导。

## HF config 校验为什么报结构不一致

Slime 会拿 `--hf-checkpoint` 的 config 对齐 Megatron 结构字段，例如 hidden size、head 数、layer 数、FFN/MoE、embedding tying、norm eps、rope theta。

```python
# 定位骨架（据 `slime/backends/megatron_utils/arguments.py` L128-L144 删节）：
if hasattr(hf_config, hf_config_name) and hasattr(args, megatron_config_name):
    if not compare_fn(getattr(hf_config, hf_config_name), getattr(args, megatron_config_name)):
        errors.append(
            f"{hf_config_name} in hf config {getattr(hf_config, hf_config_name)} is not equal to "
            f"{megatron_config_name} {getattr(args, megatron_config_name)}, please check the config."
        )

# Validate rope_theta separately using the resolved value
if _hf_rope_theta is not None:
    if not equal(_hf_rope_theta, getattr(args, "rotary_base", None)):
        errors.append(
            f"rope_theta in hf config {_hf_rope_theta} is not equal to "
            f"rotary_base {getattr(args, 'rotary_base', None)}, please check the config."
        )

if len(errors) > 0:
    raise AssertionError("hf_validate_args failed: " + "; ".join(errors))
```

`debug_rollout_only` 会跳过部分 HF validate，但正常训练不要依赖这个绕过。

## 该跑哪些验证

| 修改内容 | 优先测试 |
|----------|----------|
| path 插件 | `python -m pytest slime/tests/plugin_contracts -q` |
| rollout/global/micro-batch 调度 | `python -m pytest slime/tests/test_dp_schedule.py -q` |
| 参数校验 | `python -m pytest slime/tests/test_megatron_argument_validation.py -q` |
| SGLang 参数透传 | `python -m pytest slime/tests/test_external_sglang_engines.py -q`，若环境有 `httpx` |
| 权重同步模式 | `slime/tests/test_megatron_argument_validation.py` 加对应断言，或读 WeightSync 专题测试 |

若 batch 数量异常，再补查 `slime/utils/dp_schedule.py`：`global_batch_size` 是每步 rollout 数，`rollout_indices` 相同的 sibling 必须留在同一步。只看参数公式无法解释 compact/subagent 行数。

下一篇 [[Slime-训练与Rollout参数-学习检查]] 用推导题检查这些边界。
