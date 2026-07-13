---
title: "阅读方法 · 排障指南"
type: troubleshooting
framework: slime
topic: "阅读方法"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# 阅读方法 · 排障指南

## 你为什么要读

源码阅读也会“排障”：症状是结论互相矛盾、图画不通、某个对象找不到；根因往往是把逻辑角色当进程、把远程发起当完成，或把设计文章当当前实现。本页按“症状—可能原因—源码入口—操作—预期”给出纠偏路径。

## 1. 症状：一直在找 Data Buffer 服务

**可能原因**：把 README 的架构角色当成一一对应的 daemon。

**源码入口**：`slime/rollout/data_source.py`、`slime/ray/rollout.py`、`slime/rollout/types.py`。

**操作**：定位 DataSource 的 `get_samples/add_samples`、RolloutManager 的 converter 与 `_split_train_data_by_dp()`，再找 `ray.put`。

**预期**：你能把 Data Buffer 拆成 prompt/sample 管理、rollout 编排、训练转换和对象存储交付，而不是找到一个名为 DataBuffer 的独立服务才算成功。

## 2. 症状：看到 `async_train()` 就判断整轮训练是异步的

**可能原因**：把远程 API 名称当成 happens-before 关系。

**源码入口**：`train.py`、`train_async.py`。

**操作**：给每个 `.remote()` 标出对应 future，再圈出每个 `ray.get`；分别画同步入口和 async 入口的时间线。

**预期**：同步入口在训练后立即等待；pipeline async 只重叠下一批 generation 与当前 train，并在发布前等待在途 generation；fully async 需要另读 example，不能由 `train_async.py` 外推。

## 3. 症状：认为 colocate 等于同进程共享参数

**可能原因**：混淆 Ray PlacementGroup、GPU bundle、Ray actor 进程和 tensor transport。

**源码入口**：`slime/ray/placement_group.py`、`slime/ray/rollout.py`、weight updater 实现。

**操作**：分别记录 training/rollout actor 的创建位置、placement view 和 updater 类型。

**预期**：colocate 的第一含义是资源重叠；训练与 serving 通常仍是不同 Python 进程。tensor/CUDA IPC 路线传递 handle/metadata，不等于共享同一个 Python `Parameter` 对象。

## 4. 症状：参数明明写在 CLI，运行事实却不一致

**可能原因**：忽略 pre-parse、namespace merge、validator 派生值和 custom config 后期覆盖。

**源码入口**：`slime/utils/arguments.py::parse_args`、`slime_validate_args`，以及 SGLang/Megatron validator。

**操作**：从 flag 定义追到 parse 后字段，再追所有赋值点；不要只读 `add_argument(default=...)`。

**预期**：你能解释 train-only 为何跳过 SGLang parse、`load_debug_rollout_data` 为何改写 `debug_train_only`、`use_critic` 如何由 advantage estimator 派生，并能指出后期 custom config 不一定重跑全部前置校验。

## 5. 症状：把所有权重更新都叫 CheckpointEngine

**可能原因**：把“发布新权重”的统一语义误当成统一类和 transport。

**源码入口**：actor/updater 选择逻辑、SGLang update endpoint/worker、disk delta 实现。

**操作**：按 colocate/full/delta 与 NCCL/disk 两个维度列实际 updater；分别追 layout 转换、transport、SGLang 应用和版本递增。

**预期**：至少分清 tensor/CUDA IPC、distributed NCCL、full disk、delta disk 四路；知道 weight version 只是发布序号，参数相等性需要另行检查。

## 6. 症状：custom generate、custom RM、rollout function 总是画成固定串行链

**可能原因**：没有先判断 hook 替换层级。

**源码入口**：参数中的 `*_path`，`load_function` 调用点，默认 `sglang_rollout.py`。

**操作**：对每个 hook 写出“由谁加载、收到什么、返回什么、默认实现是否仍执行”。

**预期**：完整 rollout function 能替换外层生成组织；custom generate/RM 是默认流水线中的槽位；DataSource 管数据供给/回填。并非所有 hook 都同时执行。

## 7. 症状：源码与 docstring、博文或笔记冲突

**可能原因**：版本漂移，或说明文字概括了旧行为。

**源码入口**：当前 baseline 的函数体、调用点和 tests。

**操作**：先确认 baseline；用调用者验证分支是否可达；把冲突记录为“说明文字声称 X，当前函数体执行 Y”，必要时补静态或运行实验。

**预期**：结论明确注明证据等级，不静默调和冲突。例如 advantage 计算的 docstring 与 pipeline-last-stage early return 条件不一致时，以当前函数体和调用契约作为实现判断，同时保留漂移提示。

## 8. 症状：得出“更快”“稳定”“阈值是 X”之类结论

**可能原因**：从架构或依赖表直接推导性能，缺少硬件、版本和 workload。

**源码入口**：相关 benchmark、profiling/trace 文档和实际运行日志。

**操作**：补齐 GPU、节点、模型、并行策略、序列长度、batch、精度、版本和统计方法；无法运行时，把结论降级为设计意图或待验证假设。

**预期**：静态源码只证明机制存在；性能数字仅对记录的环境和 workload 成立。

## 9. 最小只读诊断

```powershell
rg -n 'def generate|_convert_samples_to_train_data|_split_train_data_by_dp' slime/slime/ray/rollout.py
rg -n 'ray.get|async_train|update_weights' slime/train.py slime/train_async.py
rg -n 'skip_sglang|load_debug_rollout_data|use_critic' slime/slime/utils/arguments.py
rg -n 'ServerArgs.add_cli_args|skipped_args' slime/slime/backends/sglang_utils/arguments.py
```

预期是同时看见对象加工、等待关系、参数派生和受控透传。任何一类缺失，都说明当前结论只覆盖了 Slime 闭环的一部分。
