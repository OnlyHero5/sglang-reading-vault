---
title: "Slime 综合学习检查"
type: exercise
framework: slime
topic: "总结复盘"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-13
---
# Slime 综合学习检查

> **通过标准：** 不以“记住类名”为准，而以能画状态账、解释失败分支、运行合适层级的验证并诚实声明环境限制为准。

## 你为什么要做这组检查

模块级笔记能证明你读过局部实现，这组检查要证明你能把资源、样本、训练、权重版本和证据层级连成同一套因果模型。完成后的产物应能直接用于设计实验、解释故障和审查改动。

## 1. 闭环口试

不看笔记，逐题回答；每题都要说出对象、所有者、版本或屏障：

- [ ] PlacementGroup 如何把 actor、critic、rollout engine 放到资源上；colocate 时谁在何时 offload/onload？
- [ ] DataSource 交出 prompt group 后，`Sample` 如何经过 generate、RM、filter、convert 和 DP split？
- [ ] `rollout_id` 是迭代标识、group 身份还是 weight version？为什么三者不能混用？
- [ ] critic-only 阶段仍执行了哪些 rollout/训练/保存动作，actor 又刻意跳过什么？
- [ ] advantage 层的 old/reference/rollout logprob，与 policy loss 层的 current/old baseline 分别是谁？
- [ ] NCCL、full disk、delta disk、tensor 四条权重路径，各自的介质、前提和半失败状态是什么？
- [ ] `train_async.py` 为什么更新权重前要等待 next-generation future？这能消除哪种风险，不能消除哪种陈旧？
- [ ] Slime 经 HTTP 调用 SGLang 时，Slime 管什么，SGLang scheduler 又管什么？

判定：任一答案只有模块名、没有数据形状或时序，就回对应专题补读。

## 2. 画出两张必须有的图

### 图 A：同步闭环

至少包含：`train.py`、PlacementGroup、RolloutManager、DataSource、SGLangEngine、actor、可选 critic、`rollout_data_ref`、weight updater、eval。标出首次权重推送和每轮更新位置。

### 图 B：异步版本账

画第 `n`、`n+1` 两轮 generate/train 的重叠，并给每批 Sample 标注生成时 engine 的 weight version。再标出 `update_weights_interval` 边界前为什么要 drain future。

通过标准：图中箭头必须写对象或事件，不能只连接框名。

## 3. 仓库级静态验证

在仓库根目录执行：

```powershell
rg -n "create_placement_groups|create_rollout_manager|create_training_models|generate\.remote|async_train|update_weights" `
  slime/train.py slime/train_async.py

rg -n "debug_rollout_only|debug_train_only|load_debug_rollout_data|skip_sglang" `
  slime/slime/utils/arguments.py slime/slime/ray/rollout.py `
  slime/slime/backends/megatron_utils/actor.py

rg -n "recover_updatable_engines|get_updatable_engines_and_lock|weight_version" `
  slime/slime/ray/rollout.py `
  slime/slime/backends/megatron_utils/actor.py `
  slime/slime/backends/megatron_utils/update_weight
```

预期：

- 同步主循环表现为 generate → actor/critic train → update；异步主循环提前提交下一轮 generate，并在更新边界 drain。
- debug mode 同时改变 parser、rollout 和 actor，不是一个局部 if。
- fault recovery、engine 选择与实际 updater 分属不同层，weight version 由 updater/engine 共同对照。

## 4. CPU 可执行验证

从 `slime/` 目录运行：

```powershell
python -m pytest tests/utils/test_trace_utils.py `
  tests/utils/test_megatron_server_arguments.py `
  tests/plugin_contracts/test_plugin_generate_contracts.py -q
```

预期：Sample trace、teacher-only 参数和 custom generate contract 通过。若依赖不满足，记录缺失包、collection 阶段和静态替代；不要把“测试文件存在”写成“测试通过”。

## 5. 五个故障推演

每个推演写出“症状 → 假设 → 源码入口 → 操作 → 预期”：

1. rollout dump 有数据，但 actor 没收到 batch。
2. multi-agent fan-out 后 reward normalization 变成整批一组。
3. engine crash 恢复后，下一批仍表现为旧策略。
4. delta disk update 中途失败，磁盘文件、updater state 和 engine version 分别可能处于什么状态。
5. `--debug-train-only` 下仍试图访问 SGLang 参数或 eval engine。

通过标准：至少指出一个“静默错误”路径和一个“显式异常/卡住”路径。

## 6. GPU e2e 的正确读法

完整阅读 `slime/tests/test_qwen3_4B_ppo.py`，先写出它的资源前提：8 GPU、Qwen3-4B、两份数据、checkpoint 转换、Megatron + SGLang、colocate。再列它实际覆盖的闭环：actor/critic、critic-only step、PPO advantage、CP/TP、动态 batch、初始/后续权重同步和可选 eval。

只有满足这些前提时才运行。未运行时，结论应写“完成静态审阅，未完成 GPU e2e”，不能写“PPO 闭环验证通过”。

## 7. 最终交付物

- [ ] 一张同步闭环图和一张异步版本图。
- [ ] 一页六大不变量说明。
- [ ] 五个故障推演及预期证据。
- [ ] 一份验证分层记录：静态、CPU、GPU、外部服务分别到哪里。
- [ ] 能从 [[Slime-复杂度热点]] 路由到对应专题，而不是全文搜索猜入口。

完成后回看 [[Slime-RL训练全链路]]，再用 [[Slime与SGLang-阅读对照]] 和 [[三框架知识地图]] 检查跨框架职责边界。
