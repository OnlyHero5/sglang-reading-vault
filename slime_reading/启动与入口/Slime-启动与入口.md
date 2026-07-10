---
title: "启动与入口"
type: map
framework: slime
topic: "启动与入口"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/map
  - source-reading
updated: 2026-07-10
---
# 启动与入口

> **你只需阅读本目录，不必打开 `slime/` 源码。**
> 内嵌代码对应 slime Git commit `22cdc6e1`。

---

## 本目录解决什么问题

阅读方法部分讲清了如何读 Slime。本目录回答：**一条 `python train.py` 命令如何从 CLI 参数出发，完成 GPU 资源预订、Rollout 子系统与 Megatron Actor 初始化，并进入 `generate → train → update_weights` 主循环？**

四个专题覆盖启动链全路径：

| 模块 | 角色 | 一句话 |
|------|------|--------|
| [[Slime-训练主循环]] | 入口与主循环 | `train()` / `train_async()` bootstrap 与 sync/async 迭代 |
| [[Slime-Ray参数]] | 集群参数 | `--colocate` / `--offload-*` / PG 相关 CLI 与 validate |
| [[Slime-训练与Rollout参数]] | 训练与 Rollout 参数 | Megatron/SGLang 透传、`--*-path` 扩展挂载 |
| [[Slime-数据准备工具]] | 训练前工具 | HF ↔ Megatron `torch_dist` 双向转换与 MODEL_ARGS |

---

## 端到端时序

这张图用于检查是否能复述 `parse_args()` → `create_placement_groups()` → `create_rollout_manager()` → `create_training_models()` → 主循环。

```mermaid
sequenceDiagram
 participant CLI as 命令行 / 脚本
 participant PA as parse_args
 participant PG as PlacementGroup<br/>Ray 编排
 participant RM as RolloutManager<br/>Rollout 生成
 participant TM as RayTrainGroup<br/>Ray 编排
 participant LOOP as 训练主循环

 CLI->>PA: sys.argv
 Note over PA: cluster/colocate<br/>train/rollout/*-path
 PA->>PG: create_placement_groups()
 Note over PG: bundle 分配<br/>colocate 共用 PG
 PG->>RM: create_rollout_manager()
 Note over RM: 引擎拓扑 · DataSource
 PG->>TM: create_training_models()
 Note over TM: MegatronTrainRayActor<br/>async_init + update_weights
 LOOP->>RM: generate(rollout_id)
 LOOP->>TM: async_train(rollout_data_ref)
 LOOP->>TM: update_weights()
 Note over LOOP: sync vs async<br/>offload / save / eval
```

这张图的读法是：启动链上 **PG 是第一个 GPU 决策点**；RolloutManager 与 RayTrainGroup 都依赖 PG 分配结果。主循环留在 driver 进程里负责节拍控制，实际 generate/train 通过 Ray remote 下发到 Rollout 与 Megatron Actor。

---

## 零基础一句话

**像「开店前的筹备」：** 参数文档是装修图纸，数据准备工具是进货，训练主循环是开业后的日常运营（generate → train → update_weights），PlacementGroup 与 RayTrainGroup 负责租场地和排班。

---

## 推荐阅读顺序

建议按训练主循环 → Ray 参数 → 训练与 Rollout 参数 → 数据准备工具阅读。若时间紧，先理解 colocate/offload，再读主循环。

| 顺序 | 文档 | 必读理由 |
|------|------|----------|
| 1 | [[Slime-训练主循环-核心概念]] | sync vs async、bootstrap 术语 |
| 2 | [[Slime-训练主循环-源码走读]] | `train.py` / `train_async.py` 全文精读 |
| 3 | [[Slime-Ray参数-源码走读]] | colocate、offload、validate 分支 |
| 4 | [[Slime-训练与Rollout参数-数据流]] | `*-path` → `load_function` 挂载链 |
| 5 | [[Slime-数据准备工具-源码走读]] | HF ↔ torch_dist 转换主流程 |

---

## 上下游衔接

| 方向 | 模块 | 衔接点 |
|------|------|--------|
| ← 阅读方法 | [[Slime-阅读方法]] | 阅读策略与 Git 基线 |
| → Ray 编排 | PlacementGroup 与 RayTrainGroup | `create_placement_groups()` 产出资源视图 |
| → Rollout | [[Slime-RolloutManager]] | `create_rollout_manager()` 创建生成侧 |
| → 训练 | [[Slime-Megatron-Actor初始化]] | `create_training_models()` 创建训练侧 |
| → 权重 | [[Slime-分布式权重同步]] | 首次 `update_weights` 建立权重版本 |

---

## 验证建议（零基础可试）

1. **参数 dry-run：** 用 `--help` 分别查看 cluster / train / rollout 参数组，确认 `--colocate` 与 `--offload-rollout` 的默认值关系。
2. **启动链 grep：** 在笔记 [[Slime-训练主循环-源码走读]] 中对照 `train()` 前 50 行，口述 PG → RM → TM 顺序。
3. **权重转换：** 按 [[Slime-数据准备工具-排障指南]] 走一遍 `convert_hf_to_torch_dist.py`，确认 `--ref-load` 指向转换产物。

---

## 模块导航

| 目录 | 状态 |
| ------ | ------ |
| [[Slime-训练主循环|训练主循环]] | ✅ |
| [[Slime-Ray参数|Arguments-Ray]] | ✅ |
| [[Slime-训练与Rollout参数|Arguments-TrainRollout]] | ✅ |
| [[Slime-数据准备工具|Tools-DataPrep]] | ✅ |

← [[Slime-阅读方法]] · → [[Slime-Ray编排]]
