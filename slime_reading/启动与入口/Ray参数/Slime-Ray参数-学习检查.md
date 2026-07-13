---
title: "Ray参数 · 学习检查"
type: exercise
framework: slime
topic: "Ray参数"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# Ray参数 · 学习检查

这个 checkpoint 不检查“看过多少源码”，只检查你能否从 CLI 推导最终资源事实。

## 读者能做什么

- [ ] 能画出 `CLI → pre-parse → SGLang parser → Megatron parser → slime_validate_args → placement group` 这条链路。
- [ ] 能解释 actor GPU、rollout GPU、engine GPU 三个粒度的区别。
- [ ] 能判断 `rollout_num_gpus=None`、`0`、正整数在 colocate、decoupled、external 下的不同含义。
- [ ] 能说出 `--offload` 为什么在 validate 后消失。
- [ ] 能指出 external engines 为什么会写回 `rollout_num_gpus` 与 `rollout_num_engines`。
- [ ] 能解释 `train_async.py + --colocate` 为什么不是 parser 层错误，而是训练入口错误。

## 推导题

假设 actor 默认是 `actor_num_nodes=2`、`actor_num_gpus_per_node=8`，先不要看答案，推导 `_get_placement_group_layout(args)` 的返回值。

| 场景 | 输入 | 你的答案 |
|------|------|----------|
| 普通 decoupled | `rollout_num_gpus=32` | `(_, _)` |
| colocate 小 rollout | `colocate=True`、`rollout_num_gpus=8` | `(_, _)` |
| zero rollout | `rollout_num_gpus=0` | `(_, _)` |
| external | `rollout_external=True` | `(_, _)` |
| external rollout-only debug | `rollout_external=True`、`debug_rollout_only=True` | `(_, _)` |

答案：

- 普通 decoupled：`(48, 16)`。
- colocate 小 rollout：`(16, 0)`。
- zero rollout：`(16, 16)`。
- external：`(16, 16)`。
- external rollout-only debug：`(0, 0)`。

## 配置题

1. `--colocate`，没有传 `--rollout-num-gpus`，actor 是 2 节点 × 8 卡。最终 `rollout_num_gpus` 是多少，`offload_train/offload_rollout` 默认是什么？
2. `--debug-rollout-only --rollout-num-gpus 0`。最终 actor 节点数和每节点 GPU 是多少？
3. 两个 external engines 的 `/server_info` 分别返回 `tp_size=2, pp_size=1` 和 `tp_size=4, pp_size=1`，都没有 `num_gpus` 字段。最终 `rollout_num_gpus` 和 `rollout_num_engines` 是多少？
4. `--update-weight-mode delta --update-weight-transport disk --colocate`。会在哪个 validate 分支失败，为什么？
5. 普通 decoupled 没传 `--rollout-num-gpus`，也不是 external。这个字段会在哪里被自动补成 actor GPU 吗？

参考答案：

1. `rollout_num_gpus=16`，两个 offload 默认 True。
2. `actor_num_nodes=0`、`actor_num_gpus_per_node=0`。
3. `rollout_num_gpus=6`、`rollout_num_engines=2`。
4. `slime_validate_args` 的 delta 校验分支失败；colocate 走 CUDA IPC handle，delta 的磁盘 diff 没有意义。
5. 不会。源码没有普通 decoupled 的通用 fallback，实际运行应显式传数字或使用 external discovery。

## 运行验证与静态替代

```powershell
python -m pytest slime/tests/test_placement_group.py -q
```

建议再跑：

```powershell
python -m pytest slime/tests/test_megatron_argument_validation.py -q
python -m pytest slime/tests/test_external_sglang_engines.py -q
```

依赖齐全时的预期结果：

- placement group 矩阵全部通过。
- argument validation 覆盖 zero rollout、larger rollout、delta + colocate。
- external tests 覆盖 `/server_info` discovery 和 `rollout_num_engines` 写回。

当前 Windows 轻量环境中，argument validation 14 项已通过；placement group 与 external tests 分别缺 `ray`、`httpx`，会在 collection 阶段失败。无法安装依赖时，完整阅读对应测试参数表并手算预期值，作为静态替代；不要勾选“运行通过”。

## 下一步

继续读 [[Slime-训练与Rollout参数]]，把资源事实和训练 batch、rollout batch、数据集、算法参数接起来。
