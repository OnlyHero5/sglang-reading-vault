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
updated: 2026-07-10
---
# Reward与过滤 · 学习检查

## 读者能做什么

- [ ] 能画出 `generate_and_rm_group -> generate_and_rm -> async_rm/batched_async_rm -> call_dynamic_filter -> RolloutFnTrainOutput` 主线。
- [ ] 能解释 RM Hub 与 Filter Hub 的分工：一个写 `sample.reward`，一个决定整组 keep/drop。
- [ ] 能说出 `sample.custom_rm_path`、`args.custom_rm_path`、`metadata["rm_type"]`、`args.rm_type` 的优先级。
- [ ] 能解释 `group_rm=True` 为什么要求 custom RM 可能使用 `(args, samples)` 签名。
- [ ] 能对比 `math`、`dapo`、`deepscaler` 的返回形状和答案提取边界。
- [ ] 能说明 `dapo` 或 remote RM 返回 dict 时为什么需要 `--reward-key`。
- [ ] 能解释 `check_reward_nonzero_std` drop 的是一整组 prompt 样本，以及 drop 后为什么会继续补样。
- [ ] 能根据 `rollout/dynamic_filter/drop_*` metrics 判断 dynamic filter 是否过严。

## 可执行检查

**操作：** 在具备对应依赖的环境中执行下面的单测和审计命令；若本机缺少训练依赖，至少完成源码证据与双链检查。

**预期：** pytest 命令通过，源码审计没有缺失文件或越界行号，双链审计没有断链。任何一项失败都应先保留原始输出，再按本页“排障演练”定位契约边界。

CPU scorer 单测：

```powershell
$env:PYTHONPATH='F:\源码阅读\slime'
python -m pytest slime/tests/test_rm_math_dapo.py -q
```

插件契约：

```powershell
$env:PYTHONPATH='F:\源码阅读\slime'
python -m pytest slime/tests/plugin_contracts/test_plugin_path_loading_contracts.py -k "custom_rm or dynamic_filter" -q
```

源码证据：

```powershell
node maintenance/audit_source_evidence.mjs --note slime_reading/Rollout生成/Reward与过滤/Slime-Reward与过滤-源码走读.md
```

双链检查：

```powershell
node maintenance/audit_wikilinks.mjs
```

旧结构残留检查：

```powershell
$terms = @(
  ('Ex' + 'plain'),
  ('Co' + 'de'),
  ('Com' + 'ment'),
  ('^## ' + 'Hop'),
  ('七 ' + 'Hop'),
  ('论文' + '库'),
  ('内部' + '编号'),
  ('派' + '工'),
  ('来源' + '：.*' + '\|')
)
rg -n ($terms -join '|') slime_reading/Rollout生成/Reward与过滤
```

## 排障演练

- [ ] 给 `--rm-type dapo` 故意不设 `--reward-key`，能预测 dynamic filter 为什么会失败或训练 reward 标量不明确。
- [ ] 把 custom RM 从单条签名改成 batch 签名，能说明它适用于 `--group-rm`。
- [ ] 看到 `drop_zero_std_1.0` metrics 很高，能解释这是全对组被 drop。
- [ ] 看到 `drop_zero_std_-1.0` metrics 很高，能解释这是 DAPO 全错组被 drop。
- [ ] remote RM 返回 dict 后，能指出 `reward_key` 和 `eval_reward_key` 分别影响训练与 eval 输出。

## 复盘问题

- [ ] 为什么 `group_rm` 不是 dynamic filter 开关？
- [ ] 为什么 `batched_async_rm` 默认不等于“批量 RPC”？
- [ ] 为什么 dynamic filter 要用 `get_reward_value(args)` 而不是直接读 `sample.reward`？
- [ ] 为什么 `math_utils` 和 `math_dapo_utils` 不能简单合并？
- [ ] 如果 rollout 变慢，如何区分 SGLang 生成慢、remote RM 慢和 dynamic filter 有效样本率低？
