---
title: "其他Rollout路径 · 核心概念"
type: concept
framework: slime
topic: "其他Rollout路径"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/concept
  - source-reading
updated: 2026-07-12
---
# 其他Rollout路径 · 核心概念

## 你为什么要读

这篇先建立一个判断框架：Alt Rollout 的差异不在“是不是 rollout”，而在“替换了默认链路的哪一层”。同样是替代方案，fully-async 改外层调度，streaming 改内层 HTTP，SFT 改样本来源和字段填充，OPD 改 reward 语义，forge 改数据来源但保留 serving 生命周期。

## 模型一：替换层级比文件名更重要

默认链路可以切成五层：

| 层级 | 默认对象 | 替代方案 | 判断问题 |
|------|----------|----------|----------|
| step 驱动 | `train.py` | `train_async.py` | generate 和 train 是否跨 step 重叠 |
| rollout function | `sglang_rollout.generate_rollout` | fully-async、SFT、forge、sleep | 是否重写有效 batch 主循环 |
| single generate | `sglang_rollout.generate` | streaming、自定义 agent | 是否只改一条 sample 的生成 |
| reward hook | `async_rm` / scorer | OPD teacher server | reward 是标量任务分，还是结构化 logprob |
| Sample 契约 | `append_response_tokens` | SFT 手写、forge 反序列化 | tokens、mask、logprob、status 是否对齐 |

替换越外层，需要自己承担的默认能力越多。替换内层则更安全，但必须遵守 Sample 字段契约。

## 模型二：fully-async 是跨 step 热队列

默认 rollout 的异步只发生在一个 rollout step 内。fully-async 把一部分状态移到进程级后台 worker：worker 持续从 DataSource 取 group，持续调用 `generate_and_rm_group`，完成的 group 放进队列；每次 rollout function 调用只负责从队列取满 `rollout_batch_size`。

```python
# 定位骨架（非逐行摘录）：slime/rollout/fully_async_rollout.py L1-L24
"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async``.
Plug in per-sample logic via ``--custom-generate-function-path`` and
per-sample reward via ``--custom-rm-path`` — the worker calls slime's stock
:func:`generate_and_rm_group` which dispatches to those.
```

把它想成“热队列”：训练 step 要货时，从已经在后台生产的 group 里取；后台 worker 不等这个 step 完全结束才开始下一批。

失效边界：

- 它不是 eval 路径，`evaluation=True` 会报错。
- 它保留 `generate_and_rm_group`，所以 custom generate 和 custom RM 仍生效。
- 它不复制默认 dynamic filter 和 oversampling 主循环。
- 它也没有 exactly-once 保证：task 异常、返回类型错误或 ABORTED 回灌失败会直接丢 group；一次 drain 超过 batch target 时，多出的完成 group 会被切掉而不回队列。
- 模块级 worker 固定第一次调用的 args 和 DataSource；同进程后续换配置不会重建，除非线程已死或显式 stop。

## 模型三：`train_async.py` 解决 step 间重叠

fully-async 解决 rollout 内长尾，`train_async.py` 解决 generate/train 串行。两者经常组合，但不是同一个层级。

```python
# 来源：train_async.py L30-L49
    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            actor_trains_this_step = rollout_id >= args.num_critic_only_steps
            value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
```

关键不变量：下一步 generate 提前启动，但权重更新前必须同步生成，避免生成中途推新权重。

```python
# 来源：train_async.py L65-L69
        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()
```

## 模型四：streaming 是单 sample HTTP 替换

streaming 不替换外层 rollout。它仍由默认 `sglang_rollout` 管理 semaphore、DP rank、abort 和 partial buffer，只把单次 `/generate` HTTP 调用从完整 JSON 改成 SSE chunk。

```python
# 定位骨架（非逐行摘录）：slime/rollout/sglang_streaming_rollout.py L1-L24
"""Streaming sglang rollout (example).

Drop-in alternative to :func:`slime.rollout.sglang_rollout.generate` that
consumes sglang's SSE stream incrementally instead of awaiting one final JSON
response. The win is on **abort**: every chunk we receive lands directly on
``sample`` (tokens, response text, log-probs), so when a partial-rollout
recycling or weight-update abort fires mid-generation, the partial state is
already on the sample — we don't depend on ``/abort_request`` returning the
collected text.
```

心理模型：每个 chunk 都重建“调用前 Sample + 本调用累计生成结果”，这样 abort 时 Sample 停在最后一个已接收 chunk 的边界。

这个模型只在 server 保持 cumulative streaming 时成立；若开启 incremental output，当前代码会把 delta 当全量。若 chunk 有 text 却没有 `output_token_logprobs`，代码还会用空 token 列表更新文本，形成 token/text 不一致，不能只检查 SSE 是否不断流。

## 模型五：SFT rollout 是数据转换器

SFT 不需要在线生成，也不需要 RM。它从 DataSource 拿 messages，用 `MultiTurnLossMaskGenerator` 生成 tokens 和 loss mask，然后把这些字段写回 Sample，让后续训练路径复用同一套 batch、checkpoint 和 logging。

```python
# 来源：slime/rollout/sft_rollout.py L42-L68
    samples = data_buffer.get_samples(args.rollout_batch_size)

    for i, sample in enumerate(samples):
        (sample,) = sample
        messages = sample.prompt
        tools = sample.metadata.get("tools", None)

        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)
        if len(token_ids) != len(loss_mask):
            raise ValueError(
                f"SFT rollout produced mismatched token_ids/loss_mask lengths: {len(token_ids)=}, {len(loss_mask)=}"
            )

        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]

        if i == 0 and not SAMPLE_PRINTED:
            logger.info(
                f"sft_rollout::generate_rollout example data: {sample=} (raw){messages=} (raw){token_ids=} (raw){loss_mask=} {response_length=}"
            )
            SAMPLE_PRINTED = True

    return samples
```

这里最重要的不是 `reward=0`，而是 `loss_mask[-response_length:]`：训练只关心 response 侧哪些 token 参与 SFT loss。

但要记住 Python 的反直觉边界：`response_length == 0` 时，`loss_mask[-0:]` 等于完整 `loss_mask`，不是空列表。当前实现没有 fail fast；只有 user/system、没有 assistant 的样本必须在数据入口拒绝，或由定制实现显式生成空 response mask。

## 模型六：OPD reward hook 写的是教师 logprob

OPD 使用教师服务打分，但它不是把 task reward 变大。`reward_func` 请求 teacher 的 token logprob，`post_process_rewards` 把 response 段 teacher logprob 写进 `sample.teacher_log_probs`，训练侧再把 KL penalty 加进 advantage 或 loss。

```python
# 定位骨架（非逐行摘录）：slime/rollout/on_policy_distillation.py L32-L67
def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]
```

因此排查 OPD 时要盯 `teacher_log_probs` 的长度和 response 对齐，而不是只看 reward 均值。

OPD 同样使用 `t_log_prob[-response_length:]`，零响应时会错误保留整段 teacher logprob；非零响应时若 teacher 返回更短，切片只会静默得到短 tensor，没有长度断言。此外每条样本新建一个无显式 timeout 的 `aiohttp.ClientSession`，吞吐与故障等待都要单独观测。

## 模型七：forge load 是保留 serving 生命周期的 replay

`load_debug_rollout_data` 会偏向跳过生成链路；forge load 的目标相反：样本来自磁盘，但 SGLang server、router、权重更新、colocate offload/onload 仍然跑，用来测真实资源占用。

```python
# 来源：slime/rollout/forge_load.py L19-L25
Unlike --load-debug-rollout-data, this path does NOT set
skip_sglang=True / debug_train_only=True (see
slime/utils/arguments.py: skip_sglang computation in _pre_parse_mode and
the debug_train_only flip when load_debug_rollout_data is set), so
sglang servers, router, weight_update and the full colocate
offload/onload dance still run. That is exactly what we want when
measuring real GPU memory.
```

forge 的核心风险是误改 `sample.rollout_id`。源码明确不覆盖它，让下游按原有 index 或 dump 中的身份分组。

下一步读 [[Slime-其他Rollout路径-源码走读]]，把这些模型和源码执行顺序对齐。
