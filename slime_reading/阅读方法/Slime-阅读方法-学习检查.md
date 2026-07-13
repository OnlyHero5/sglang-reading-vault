---
title: "阅读方法 · 学习检查"
type: exercise
framework: slime
topic: "阅读方法"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# 阅读方法 · 学习检查

## 你为什么要做

本练习验收的是“能否从源码恢复对象与顺序”，不是记住笔记里的结论。全部命令均为只读；不具备 Ray/GPU/Megatron 环境时，静态替代仍可完成。

## 1. 五本账自测

- [ ] 资源账：能解释 colocate 与同进程共享参数的区别。
- [ ] 样本账：能区分外层 `rollout_id`、`Sample.rollout_id` 和训练 sample 行。
- [ ] 训练账：能从 Sample 追到 per-DP object refs。
- [ ] 版本账：能区分 optimizer step、权重发布和 version 递增。
- [ ] 等待账：能说明同步入口为何仍调用名为 `async_train` 的方法。

## 2. 实验一：恢复同步时序

执行：

```powershell
rg -n 'generate.remote|ray.get|async_train|update_weights' slime/train.py
```

把命中项按执行顺序抄成四行，并在每行写出主体、输入、输出/未来值和等待关系。

**预期**：得到“发起并等待 generation → 发起并等待 training → 发布 actor 权重”。如果写成“generate 与 train 并发”，说明把 `.remote()` 与后续立即发生的 `ray.get` 分开看了。

## 3. 实验二：找出 pipeline async 的边界

执行：

```powershell
rg -n 'assert not args.colocate|rollout_data_next_future|Start the next rollout early|sync generate before update weights|update_weights' slime/train_async.py
```

**预期**：能画出 `generate(i+1)` 与 `train(i)` 的重叠，并指出 weight update 前会收口在途 generation。还应找到 colocate 的显式禁止。

## 4. 实验三：从 Sample 追到 rank-local 数据

执行：

```powershell
rg -n 'def generate|_get_rollout_data|_convert_samples_to_train_data|_split_train_data_by_dp|ray.put' slime/slime/ray/rollout.py
```

填写下表：

| 阶段 | 输入 | 输出 | 所有者 |
|------|------|------|--------|
| rollout function |  |  |  |
| converter |  |  |  |
| DP split |  |  |  |
| object store |  |  |  |

**预期**：最后一行不是 HTTP response，而是每个 DP rank 的训练数据引用。

## 5. 实验四：证明 SGLang 参数是受控透传

执行：

```powershell
rg -n 'skipped_args|new_add_argument_wrapper|ServerArgs.add_cli_args|sglang_' slime/slime/backends/sglang_utils/arguments.py
```

任选一个被透传参数和一个被跳过字段，说明它们为何走不同路径。

**预期**：结论应是“复用当前安装版 SGLang 参数表，加前缀并排除 Slime 接管字段”，而不是“无条件复制所有参数”。

## 6. 实验五：恢复 parser 的条件分支

执行：

```powershell
rg -n '_pre_parse_mode|skip_sglang|sglang_parse_args|slime_validate_args|megatron_validate_args|sglang_validate_args|load_debug_rollout_data' slime/slime/utils/arguments.py
```

**预期**：能解释：

- train-only/加载调试 rollout 数据为何不解析 SGLang；
- namespace 在何时合并；
- Slime validator 为何不能被当成纯校验器；
- Megatron 与 SGLang validator 分别在什么条件下跳过。

## 7. 口述验收

不看正文，用两分钟回答：

1. Slime 的 Training / Rollout / Data Buffer 是逻辑角色还是固定进程拓扑？
2. 一批 Sample 在哪里变成训练 actor 消费的数据？
3. 哪个等待点保证同步训练完成后才发布权重？
4. pipeline async 允许哪两件事重叠，又在哪里重新同步？
5. native pass-through 为什么仍然存在 skip、转换和 validator？

五题中任何一题只能回答模块名、不能说清对象或等待点，就回到 [[Slime-阅读方法-数据流]] 重画路径。

## 8. 可选运行验证与环境限制

完整运行 Slime 需要匹配的 SGLang、Megatron、Ray、CUDA/GPU 与模型数据环境，本专题不提供无条件可执行的训练命令。若环境齐全，可在 rollout-only/train-only debug 中记录：PID、Ray actor、GPU bundle、rollout id、weight version 和关键 `ray.get` 前后时间戳；否则以上五个静态实验就是最低验收。

完成后继续读 [[Slime-训练主循环]]，把方法论应用到真正的入口脚本。
