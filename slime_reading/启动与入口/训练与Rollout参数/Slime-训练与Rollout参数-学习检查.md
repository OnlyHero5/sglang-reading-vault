---
title: "训练与Rollout参数 · 学习检查"
type: exercise
framework: slime
topic: "训练与Rollout参数"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# 训练与Rollout参数 · 学习检查

这个 checkpoint 检查你能不能从配置推出运行行为，而不是背参数名。

## 读者能做什么

- [ ] 能区分 `rollout_function_path` 与 `custom_generate_function_path` 的边界。
- [ ] 能区分 prompt group、rollout execution、Sample 行、global batch 和 micro-batch，并在默认路径中推导 `global_batch_size`。
- [ ] 能说出 `dynamic_sampling_filter_path`、`rollout_sample_filter_path`、`rollout_all_samples_process_path` 的不同阶段。
- [ ] 能解释 `custom_rm_path` 在 `group_rm=True` 时签名为什么变成 `(args, samples)`。
- [ ] 能指出 custom advantage 和 custom loss 的实际消费点。
- [ ] 能说明 disk/delta weight sync 的四个必要边界。
- [ ] 能选择应跑的 contract test。
- [ ] 能说明 contract test 通过为何仍不能证明 custom converter 满足当前 `rollout_ids` 调度契约。

## 推导题

1. `rollout_batch_size=64`、`n_samples_per_prompt=4`、`num_steps_per_rollout=2`，最终 `global_batch_size` 是多少？
2. 你只想给每个 sample 加工具调用循环，但保留默认 dynamic filter 和 abort 行为。应该改哪个 path？
3. 你想完全替换 rollout 数据来源、生成循环和返回对象。应该改哪个 path？
4. `group_rm=True` 且传了 `custom_rm_path`，函数前两个参数应该是什么？
5. `loss_type=custom_loss` 但没传 `custom_loss_function_path`，会在哪个阶段暴露？
6. `update_weight_mode=delta`、`update_weight_transport=nccl` 是否合法？
7. `eval_function_path` 没传时，最终会是什么？

参考答案：

1. 默认路径中是每步 `128` 个 rollout execution；不是无条件的 128 条 Sample。compact sibling 共享 id 时，行数可以更多。
2. `custom_generate_function_path`。
3. `rollout_function_path`。
4. `(args, samples)`。
5. Megatron loss 分支加载 custom loss 时。
6. 不合法；delta 必须 disk。
7. validate 后等于 `rollout_function_path`。

## 运行验证与静态替代

在 `slime/` repo 根目录下优先跑：

```powershell
python -m pytest tests/test_dp_schedule.py -q
python -m pytest tests/plugin_contracts -q
python -m pytest tests/test_megatron_argument_validation.py -q
```

预期结果：

- DP schedule tests 固定 rollout-id/compact/micro-batch 调度。
- contract tests 固定 path 加载、部分签名和默认行为，但 custom converter 返回字段覆盖仍落后于当前 scheduler。
- argument validation tests 固定 HF/AllGather-CP、zero rollout、delta/disk 等边界；当前没有固定 batch 推导或 eval 默认值。

当前轻量环境中 DP schedule 9 项和 argument validation 14 项通过；plugin contracts 缺 `httpx`，在 collection 阶段失败。不要把依赖缺失当成源码行为失败，也不要把静态阅读记作运行通过。

## 下一步

继续读 [[Slime-数据准备工具]]，看数据准备脚本如何给这套参数契约提供 prompt data。
