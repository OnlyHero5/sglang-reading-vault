---
title: "其他Rollout路径 · 排障指南"
type: troubleshooting
framework: slime
topic: "其他Rollout路径"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# 其他Rollout路径 · 排障指南

## 你为什么要读

这篇按症状排障。先问一个问题：你替换的是整段 rollout、单 sample generate、reward hook，还是样本来源？层级判断错了，后面的参数通常都会配错。

## 症状一：不知道 sync、train_async、fully-async 怎么选

判断：如果只是 rollout 内 sample 长尾，考虑 fully-async；如果 generate 和 train 本身串行等待明显，先看 `train_async.py`；如果 colocate，不能用 `train_async.py`。

```python
# 来源：train_async.py L9-L11
# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
```

处理：

- colocate 场景：用默认 `train.py`，再考虑 streaming 或默认 rollout 优化。
- 非 colocate 且 generate/train 时间接近：用 `train_async.py`。
- 非 colocate 且 rollout 内长尾明显：`train_async.py` 加 fully-async rollout。

验证：看日志中是否出现下一步 `generate.remote` 提前启动，以及 fully-async 的 `queue_warm`、`queue_left`。

## 症状二：fully-async 在 eval 报错

判断：这是设计，不是 bug。fully-async 的全局 worker 持续跨 rollout 调用运行，和 eval 的固定 dataset、固定输出字典语义冲突。

```python
# 来源：slime/rollout/fully_async_rollout.py L251-L256
def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
```

处理：

- 训练用 `--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async`。
- eval 保持默认 `--eval-function-path`，不要指向 fully-async。
- 如果需要自定义 eval，单独写 eval function，不要复用 fully-async worker。

## 症状三：fully-async 队列一直不增长

判断：worker 从 `data_buffer.get_samples(1)` 补 in-flight；如果 DataSource 空、task 崩溃、SGLang 不返回，output queue 都不会增长。

```python
# 来源：slime/rollout/fully_async_rollout.py L135-L154
                # Top up.
                while len(active_tasks) < max_concurrent and self.running:
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        gid = gid_counter
                        gid_counter += 1
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        task.add_done_callback(self._make_done_cb(gid))
                        active_tasks.add(task)

                await asyncio.sleep(1)
```

处理：

- 检查 DataSource 是否还有样本，尤其是 buffer 和 dataset 游标。
- 检查 worker 日志 `fully-async task crashed`。
- 检查 SGLang router 是否健康，以及 custom generate 是否卡住。
- 降低 `sglang_server_concurrency` 验证是否是并发压垮 server。
- 同时确认 `get_rollout_num_engines(args)>0`；并发乘积为 0 时不会创建任何 task，而前台没有超时。

## 症状四：ABORTED 样本进了训练或丢了

判断：fully-async 的规则是 ABORTED group 回灌 DataSource，不进入 output queue。如果 DataSource 不支持 `add_samples`，回灌会失败。

```python
# 来源：slime/rollout/fully_async_rollout.py L182-L189
            # Aborted group → requeue, don't ship to training.
            if any(getattr(s, "status", None) == Sample.Status.ABORTED for s in result):
                try:
                    self.data_buffer.add_samples([result])
                except Exception:  # noqa: BLE001
                    logger.exception("fully-async: failed to requeue aborted group")
                return
            self.output_queue.put((gid, result))
```

处理：

- 确认使用的是 global DataSource。
- 确认 `add_samples` 可用。
- 区分 fully-async callback 回灌和默认 `sglang_rollout.abort` 的 partial 回灌。
- task crash、返回类型错误、回灌异常都不是自动重试：应按 gid/sample index 建立外部审计，否则只看 batch 输出无法知道丢了哪些组。

## 症状四补充：下一 step 少了已完成 group

若 queue_warm 明明大于 batch target，下一 step 却没有继承余量，检查 `_generate_rollout_async`：它会一次 drain 整个 output queue，随后仅返回排序后的前 `target` 项。超出的完成 group 没有放回队列，这是当前实现的数据丢失边界。验证时预装 target+N 个 group，预期当前实现返回 target 且 queue 变为 0，而不是保留 N。

## 症状五：streaming 开了但 partial 还是没有保存

判断：streaming 只保证“收到的 chunk 已写入 Sample”。如果没有 chunk 返回、HTTP client 未初始化、server 输出不是 cumulative 语义，Sample 仍可能没有可回收状态。

```python
# 来源：slime/rollout/sglang_streaming_rollout.py L108-L115
    client = http_utils._http_client
    assert client is not None, "http client not initialized; call init_http_client first"

    with trace_span(
        sample, "sglang_generate_stream", attrs={"max_new_tokens": sampling_params["max_new_tokens"]}
    ) as span:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
```

处理：

- 确认是 `--custom-generate-function-path slime.rollout.sglang_streaming_rollout.generate_streaming`，不是 `--rollout-function-path`。
- 确认 `--partial-rollout` 与 oversampling 配置确实会触发 abort。
- 看 `sglang_generate_stream` trace 和 Sample 的 `response_length` 是否随 chunk 增长。

## 症状六：streaming 后 top-p 或 loss mask 对不上

判断：streaming 每个 chunk 会先恢复调用前状态，再用 `append_response_tokens` 重建本次调用累计结果。如果 custom 修改绕过这个过程，就会破坏对齐。

```python
# 定位骨架（非逐行摘录）：slime/rollout/sglang_streaming_rollout.py L136-L154
                # Surface partial state on the sample immediately. If the
                # outer abort path cuts us, whatever we've written so far is
                # what survives — no /abort_request round-trip needed.
                sample.tokens = list(base_tokens)
                sample.response = base_response
                sample.response_length = base_response_length
                sample.rollout_log_probs = None if base_log_probs is None else list(base_log_probs)
                sample.rollout_top_p_token_ids = base_top_p_token_ids
                sample.rollout_top_p_token_offsets = base_top_p_token_offsets
                sample.loss_mask = None if base_loss_mask is None else list(base_loss_mask)
                sample.append_response_tokens(
                    args,
                    tokens=call_tokens,
                    log_probs=call_log_probs,
                    trainable=True,
                    meta_info=meta,
```

处理：

- 不要手写 `rollout_top_p_token_offsets`。
- 不要把 cumulative chunk 当 delta 重复追加。
- 如果 server 开启 incremental streaming，需要重新实现 delta 语义。
- 同时检查每个 chunk 是否都有 `output_token_logprobs`；只有 text 没有 token logprob 时会产生文本/token 不一致。

## 症状七：SFT 训练 loss 全零或 user token 进入 loss

判断：SFT 的核心是 `response_length` 和 response tail 的 `loss_mask`。messages 模板、tools、`loss_mask_type` 任一不对都会影响 mask。

```python
# 定位骨架（非逐行摘录）：tests/gemma4/test_gemma4_sft_rollout.py L66-L91
def test_multi_turn_response_length_spans_from_first_assistant():
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    samples, tok = _run_rollout([messages])
    sample = samples[0]

    tail_tokens = sample.tokens[-sample.response_length :]
    masked = tok.decode([tail_tokens[i] for i in range(len(tail_tokens)) if sample.loss_mask[i] == 1])
    assert "A1" in masked
    assert "A2" in masked
    assert "Q2" not in masked

    assert sample.effective_response_length == sum(sample.loss_mask)
    assert sample.effective_response_length < sample.response_length
```

处理：

- 确认 `sample.prompt` 是 messages 列表，不是已拼接字符串。
- 确认 `loss_mask_type` 与模型 chat template 匹配。
- 用最小 messages 复现，解码 masked token 看 assistant/user 边界。
- 增加纯 user/system 反例；若 `response_length=0`，当前 `[-0:]` 会取完整 mask，应在数据层拒绝而不是继续训练。

## 症状八：OPD reward 均值为 0，以为没学习

判断：纯 OPD 的标量 reward 可以全是 0。学习信号来自 `sample.teacher_log_probs` 和训练侧 KL penalty。

```python
# 来源：slime/rollout/on_policy_distillation.py L58-L67
    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
```

处理：

- 检查 `teacher_log_probs` 是否存在且长度等于 response token 数。
- 检查 teacher server 的 `input_token_logprobs` 是否返回。
- 如果还需要任务 reward，在 post-process 里叠加，不要覆盖 teacher logprob。
- 对每条样本断言 `len(teacher_log_probs)==response_length`；特别测试 response_length=0 和 teacher 返回过短。

## 症状九：forge load 触发 DP schedule 或 rollout_id 断言

判断：forge load 明确不覆盖 `sample.rollout_id`。如果你在外部脚本里把所有样本改成当前 rollout_id，可能把多个样本压成一个 rollout 分组。

```python
# 来源：slime/rollout/forge_load.py L99-L114
    logger.info("forge_load: loading samples from %s", path)
    blob = torch.load(path, weights_only=False)
    samples = [Sample.from_dict(s) for s in blob["samples"]]
    # IMPORTANT: do NOT overwrite sample.rollout_id with the current rollout_id.
    # Default-shape rollouts leave rollout_id=None and slime falls back to
    # sample.index in slime/ray/rollout.py (the dp-schedule grouping key).
    # Forcing all samples to share one rollout_id collapses them into a single
    # "rollout", which trips the num_rollouts >= global_batch_size assert in
    # slime/utils/dp_schedule.py.
    logger.info(
        "forge_load: loaded %d samples for rollout_id=%d from %s",
        len(samples),
        rollout_id,
        Path(path).name,
    )
    return RolloutFnTrainOutput(samples=samples)
```

处理：

- 不要在 forge 加载后重写 `sample.rollout_id`。
- 检查 dump 中 `sample.index` 是否覆盖全局 batch 所需分组。
- eval 没有 train fallback；缺 eval dump 返回空 eval 是预期。

## 症状十：sleep rollout 任务永不结束

判断：这是它的唯一行为，用于 profiling 等待状态，不是生产 rollout。

```python
# 来源：slime/rollout/sleep_rollout.py L7-L12
def sleep(args, rollout_id, data_source, evaluation=False):
    count = 0
    while True:
        time.sleep(3600)
        count += 1
        logger.info(f"rollout sleep for {count} hours")
```

处理：

- 只在明确要让 rollout 阻塞时使用。
- 不要把它配置进正常训练或 eval。
- 如果 Ray generate 永久 pending，先检查是否误用了这个函数路径。

## 运行验证

替代 rollout 的排障可以先用源码检索确认各路径仍然是独立入口：train_async、fully-async、streaming、SFT mask、OPD teacher logprob、forge load、sleep rollout。

```powershell
rg -n 'train_async|fully_async|generate_rollout_fully_async|generate_streaming|partial|loss_mask_type|teacher_log_probs|forge_load|rollout_id|def sleep|on_policy_distillation' slime/train_async.py slime/slime/rollout/fully_async_rollout.py slime/slime/rollout/sglang_streaming_rollout.py slime/tests/gemma4/test_gemma4_sft_rollout.py slime/slime/rollout/on_policy_distillation.py slime/slime/rollout/forge_load.py slime/slime/rollout/sleep_rollout.py
```

读输出时先按“替换了哪一层”定位：外层 step 重叠看 `train_async`，rollout 内部调度看 fully-async，生成调用看 streaming，样本字段看 SFT/OPD/forge。不要把所有 pending 或 mask 问题都归到同一条 rollout 主线。
