---
title: "Reward与过滤 · 学习检查"
type: exercise
framework: slime
topic: "Reward与过滤"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# Reward与过滤 · 学习检查

## 读者能做什么

- [ ] 能画出 `generate_and_rm_group -> generate_and_rm -> async_rm/batched_async_rm -> call_dynamic_filter -> RolloutFnTrainOutput` 主线。
- [ ] 能解释 RM Hub 与 Filter Hub 的分工：一个写 `sample.reward`，一个决定整组 keep/drop。
- [ ] 能说出单条 `async_rm` 中 `sample.custom_rm_path`、`args.custom_rm_path`、`metadata["rm_type"]`、`args.rm_type` 的优先级，并解释为什么全局 batch custom RM 会绕过 per-sample path。
- [ ] 能解释 `group_rm=True` 为什么要求 custom RM 可能使用 `(args, samples)` 签名。
- [ ] 能对比 `math`、`dapo`、`deepscaler` 的返回形状和答案提取边界。
- [ ] 能说明 `compute_score_dapo` 的 strict-box 是函数级能力，默认 `rm_type=dapo` 分发没有暴露该开关。
- [ ] 能说明 `dapo` 或 remote RM 返回 dict 时为什么需要 `--reward-key`。
- [ ] 能解释 `check_reward_nonzero_std` drop 的是一整组 prompt 样本，以及 drop 后为什么会继续补样。
- [ ] 能根据 `rollout/dynamic_filter/drop_*` metrics 判断 dynamic filter 是否过严。
- [ ] 能解释 `boxed_math`、`boxed_deepscaler`、`boxed_remote_rm` 为什么不能按“通用预处理前缀”理解。
- [ ] 能发现 batch RM 输出与输入不等长时 `zip(strict=False)` 的静默截断。
- [ ] 能说明单样本 group 的 `std=nan` 与永久 drop 时无终止门禁这两个运行风险。

## 可执行检查

**操作：** 从知识库根目录进入 `slime/` upstream，再执行 CPU scorer 与插件契约测试。若依赖不完整，保留 collection error，并完成下面的静态入口定位。

**预期：** pytest 通过时证明 scorer 或 hook 契约成立；缺 `httpx`、Torch ABI 或其他依赖时，结论只能是“环境未满足”，不能写成 reward 实现失败。

CPU scorer 单测：

```powershell
Push-Location slime
python -m pytest tests/test_rm_math_dapo.py -q
Pop-Location
```

插件契约：

```powershell
Push-Location slime
python -m pytest tests/plugin_contracts/test_plugin_path_loading_contracts.py -k "custom_rm or dynamic_filter" -q
Pop-Location
```

静态入口：

```powershell
rg -n 'generate_and_rm_group|async_rm|batched_async_rm|call_dynamic_filter|get_reward_value' slime/slime/rollout
rg -n 'reward_key|eval_reward_key|group_rm|dynamic_filter' slime/slime/utils/arguments.py
```

预期：能把 RM 赋值、整组过滤、补样和 metrics 四个边界串起来，而不是只找到 scorer 文件名。

## 排障演练

- [ ] 给 `--rm-type dapo` 故意不设 `--reward-key`，能预测 dynamic filter 为什么会失败或训练 reward 标量不明确。
- [ ] 把 custom RM 从单条签名改成 batch 签名，能说明它适用于 `--group-rm`。
- [ ] 看到 `drop_zero_std_1.0` metrics 很高，能解释这是全对组被 drop。
- [ ] 看到 `drop_zero_std_-1.0` metrics 很高，能解释这是 DAPO 全错组被 drop。
- [ ] remote RM 返回 dict 后，能指出 `reward_key` 和 `eval_reward_key` 分别影响训练与 eval 输出。
- [ ] 给 custom batch RM 故意少返回一个 reward，能预测哪个 sample 仍为 `None`，并在插件内加入等长断言。
- [ ] 给 `boxed_math` 输入 `\boxed{42}`，能沿“先抽成 42→math 再找 box”解释为何当前实现判 0。
- [ ] 设置 `n_samples_per_prompt=1` 和 nonzero-std filter，能预测它会 drop 而不是 keep。

## 复盘问题

- [ ] 为什么 `group_rm` 不是 dynamic filter 开关？
- [ ] 为什么 `batched_async_rm` 默认不等于“批量 RPC”？
- [ ] 为什么 dynamic filter 要用 `get_reward_value(args)` 而不是直接读 `sample.reward`？
- [ ] 为什么 `math_utils` 和 `math_dapo_utils` 不能简单合并？
- [ ] 如果 rollout 变慢，如何区分 SGLang 生成慢、remote RM 慢和 dynamic filter 有效样本率低？
- [ ] 为什么内置 `remote_rm` 即使从 `batched_async_rm` 调用，也不等于一次 batch HTTP RPC？
- [ ] 为什么默认 eval rollout 不能直接复用训练侧的 `group_rm`？
