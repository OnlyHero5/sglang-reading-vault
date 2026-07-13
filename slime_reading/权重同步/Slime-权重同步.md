---
title: "权重同步"
type: map
framework: slime
topic: "权重同步"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/map
  - source-reading
updated: 2026-07-13
---
# 权重同步

> 本目录先提供连续的权重同步主线；准备修改实现、核对版本漂移或遇到证据争议时，仍应按 Git `22cdc6e1` 打开 `slime/` 源码验证。
> SGLang 交叉对照：[[SGLang-ModelLoader]]、[[SGLang-CheckpointEngine]]、[[Slime-SGLang-Engine]]。

---

## 本目录解决什么问题

训练后端部分讲清了 Megatron 如何完成一次 train step。本目录回答：**训练后的 actor 权重如何推送到 SGLang Rollout 引擎，nccl / disk / delta / tensor（colocate）四条路径各适用什么场景？**

三个专题覆盖权重同步与 checkpoint 全链路：

| 专题 | 角色 | 一句话 |
|------|------|--------|
| [[Slime-分布式权重同步]] | NCCL 广播 | `UpdateWeightFromDistributed`、PP source → engine GPU |
| [[Slime-磁盘权重同步]] | 磁盘 / tensor | full disk、delta disk、colocate IPC tensor |
| [[Slime-Megatron到HF转换]] | 存取与转换 | `load_checkpoint`、`save_hf`、Megatron→HF 管线 |

---

## 端到端时序

这张图用于检查是否能对比 nccl / disk / delta / tensor 四种权重同步路径的数据流。

```mermaid
sequenceDiagram
 participant TR as 训练主循环
 participant ACT as MegatronTrainRayActor
 participant UW as update_weights()
 participant NCCL as Distributed NCCL
 participant DISK as Disk/Delta
 participant TSR as Tensor colocate
 participant ENG as SGLangEngine
 participant CK as Checkpoint save

 TR->>ACT: update_weights()
 ACT->>UW: weight_updater 选型
 alt transport=nccl · 分离部署
 UW->>NCCL: UpdateWeightFromDistributed
 NCCL->>ENG: NCCL broadcast HF 张量
 else transport=disk
 UW->>DISK: save HF → engine reload
 Note over DISK: full 或 delta 链
 else colocate · tensor
 UW->>TSR: IPC / 可选 NCCL
 TSR->>ENG: update_weights_from_tensor
 end
 opt 周期性 checkpoint
 ACT->>CK: 先 Megatron save，再可选 save_hf
 end
```

这张图的读法是：训练主循环按 `update_weights_interval` 的门禁触发 `update_weights`；transport 由 `--update-weight-transport`、colocate 拓扑和 update mode 共同决定。checkpoint save 是另一条周期性路径，不是每次 rollout 权重同步都会发生的一步。

## 先选介质，再谈快慢

| 路径 | 核心介质 | 必要前提 | 关键失败状态 |
|------|----------|----------|----------------|
| distributed NCCL | HF named tensor 经 trainer↔engine process group 广播 | 分离拓扑、group 与 rank 映射正确 | engine 可能被 pause/lock，版本号已前移，无统一 rollback |
| full disk | 完整 HF 目录写共享文件系统后 reload | 所有节点可见同一路径 | 最终版本目录中可留半成品，无整目录原子发布 |
| delta disk | trainer snapshot → XOR delta → host-local mmap base | Linux/POSIX `flock`、初始 HF base 一致 | snapshot/base 可前移或半写，失败后应重建 host-local base |
| tensor colocate | 同机 CUDA IPC tensor bucket | colocated actor/engine 与可用 IPC handle | pause/flush/continue 只覆盖 colocated 子集，混合 remote 需单独审计 |

这四条路径没有一个共同的“原子提交 + 统一回滚”层。正确的工程问法不是只问“哪个快”，而是同时问：谁持有新权重、谁已 pause、版本号是否前移、磁盘/base/snapshot 哪个已改变，以及失败后从哪个已知好状态重建。

---

## 零基础一句话

**像「总部给分店换菜单」：** 分布式同步是实时对讲机（NCCL 直传），磁盘 full/delta 是快递包，tensor 路径是同机闪送（colocate IPC），Megatron 到 HF 转换负责存档和格式翻译。

---

## 推荐阅读顺序

建议先读分布式权重同步，再比较磁盘 full/delta 与 tensor colocate，最后理解 Megatron 到 HF 转换。时间紧时至少分清 NCCL 与磁盘路径的同步边界。

| 顺序 | 文档 | 必读理由 |
|------|------|----------|
| 1 | [[Slime-分布式权重同步-核心概念]] | NCCL 组命名、PP source rank |
| 2 | [[Slime-分布式权重同步-源码走读]] | `UpdateWeightFromDistributed` 主路径 |
| 3 | [[Slime-磁盘权重同步-核心概念]] | full / delta / tensor 模式矩阵 |
| 4 | [[Slime-磁盘权重同步-排障指南]] | delta 链与 colocate 分工 |
| 5 | [[Slime-Megatron到HF转换-源码走读]] | `save_hf` 与 megatron_to_hf 管线 |

---

## 阶段衔接

| 方向 | 专题 | 衔接点 |
|------|------|--------|
| ← 训练后端 | [[Slime-训练步骤]] | train 完成 → `update_weights` |
| → 高级特性 | [[Slime-Agent轨迹]] · [[Slime-自定义扩展]] | Agent/customization 不改变同步主路径 |
| → Rollout | [[Slime-SGLang-Engine]] | engine `init_weights_update_group` / reload |
| → 启动工具 | [[Slime-数据准备工具]] | HF ↔ torch_dist 与 `--load` / `--ref-load` |
| → SGLang 对照 | [[SGLang-CheckpointEngine]] | SGLang 侧 checkpoint 热更新 |

---

## 验证建议（零基础可试）

1. **transport 矩阵：** 对照 [[Slime-磁盘权重同步-核心概念]]，列出 colocate + nccl vs 分离 + disk 的组合。
2. **delta 约束：** 确认 `--update-weight-mode=delta` 必须 `--update-weight-transport=disk` 的原因（见 [[Slime-磁盘权重同步-排障指南]]）。
3. **save 路径：** 追踪 [[Slime-Megatron到HF转换-数据流]] 中 `--save-hf` 产出目录结构与 SGLang `--model-path` 的对应关系。
4. **失败状态账：** 为四条路径分别列出 trainer、engine、version、disk/base 和 pause/lock 状态，标出哪些修改没有 `finally` 或 rollback。

---

## 专题导航

| 目录 | 状态 |
| ------ | ------ |
| [[Slime-分布式权重同步|分布式 NCCL 同步]] | ✅ |
| [[Slime-磁盘权重同步|磁盘与 Tensor 同步]] | ✅ |
| [[Slime-Megatron到HF转换|Megatron 与 HF 转换]] | ✅ |

← [[Slime-训练后端]] · → [[Slime-高级特性]]
