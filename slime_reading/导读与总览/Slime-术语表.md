---
title: "Slime 术语表"
type: reference
framework: slime
topic: "导读与总览"
learning_role: reference
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/reference
  - source-reading
updated: 2026-07-10
---
# Slime 术语表

本页用来把 Slime 文档中的 RL、Ray、Megatron、rollout 和权重同步术语对齐到源码入口。读完后，你应该能在遇到 `advantage`、`async_train`、`colocate`、`RolloutDataSource` 等词时知道先去哪个专题继续读。

> RL + Slime 专有术语 · 首次出现代码出处

---

## A

### advantage / advantage_estimator

**定义：** 策略梯度中 baseline 校正后的回报信号；Slime 支持 grpo、gspo、ppo、cispo 等。

**代码：**

```python
## 来源：slime/backends/megatron_utils/loss.py L661-L667
def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Supported methods: "grpo", "gspo", "cispo", "ppo", ..."""
```

→ [[Slime-Advantage计算-核心概念]]

---

### async_train

**定义：** RayTrainGroup 向各 DP rank 异步提交 `train` remote call。

**代码：**

```python
## 来源：train.py L77
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
```

→ [[Slime-RayTrainGroup-核心概念]]

---

## C

### colocate

**定义：** train GPU 与 rollout GPU 映射到同一物理 GPU，通过 offload 时分复用显存。

→ [[Slime-Ray参数-核心概念]]

---

### critic

**定义：** 可选 value network；PPO 时先 train critic 再 train actor。

**代码：**

```python
## 来源：train.py L74-L79
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
```

---

## D

### DataSource / RolloutDataSource

**定义：** 提供 prompt、管理 rollout buffer 与 dataset 持久化。

→ [[Slime-数据源-核心概念]]

---

### DP（Data Parallel）

**定义：** Megatron 数据并行 rank；RolloutManager `_split_train_data_by_dp` 按 rank 拆分 train_data。

→ [[Slime-训练数据-核心概念]]

---

## G

### GRPO（Group Relative Policy Optimization）

**定义：** 组内相对 reward 的 advantage 估计；`--advantage-estimator grpo`。

→ [[Slime-Advantage计算-排障指南]]

---

### generate / generate_rollout

**定义：** RolloutManager.generate（Ray 层）vs rollout fn generate_rollout（样本生产层）。

→ [[Slime-RolloutManager-核心概念]] · [[Slime-SGLang-Rollout-核心概念]]

---

## L

### loss_masks

**定义：** 标记哪些 token 参与 policy loss（通常 response 部分为 1）。

**代码：** 见 [[Slime-Sample数据契约-核心概念]] 中 `Sample` 字段。

---

## O

### offload / offload_rollout / offload_train

**定义：** 训练或推理权重/KV 卸载到 CPU，释放 GPU 给另一角色。

→ [[Slime-Ray参数-排障指南]]

---

### on-policy / off-policy

**定义：** rollout_log_probs 与 train log_probs 一致为 on-policy；TIS/ICEPOP 处理 off-policy 偏差。

→ [[Slime-Advantage计算-排障指南]]

---

## P

### PlacementGroup（PG）

**定义：** Ray 资源 bundle；Slime 为 train/rollout/critic 分别创建 PG。

**代码：**

```python
## 来源：slime/ray/placement_group.py L47-L48
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
```

---

### PPO（Proximal Policy Optimization）

**定义：** clipped policy gradient；`policy_loss_function` 实现。

→ [[Slime-Policy-Loss-核心概念]]

---

## R

### rollout_id

**定义：** 外层训练迭代索引；贯穿 generate/train/save/eval/trace。

→ [[Slime-关键概念#1 · rollout_id]]

---

### rollout_log_probs

**定义：** SGLang 采样时记录的 log prob；用于 importance sampling 与 KL。

→ [[Slime-Sample数据契约-核心概念]]

---

### RM / Reward Model

**定义：** `rm_hub.async_rm` 对 Sample 打分；可替换为 rule-based verifier。

→ [[Slime-Reward与过滤-核心概念]]

---

## S

### Sample

**定义：** Rollout 最小单元 dataclass；含 tokens、rewards、metadata 等。

→ [[Slime-Sample数据契约-核心概念]]

---

### ServerGroup

**定义：** 多 SGLang engine + router 的拓扑单元；支持 PD 分离。

→ [[Slime-引擎拓扑-核心概念]]

---

## T

### trace / trace_span

**定义：** Sample 级执行 span；`trace_utils.trace_span` 记录耗时段。

→ [[Slime-可观测性与CI]]

---

### TIS（Truncated Importance Sampling）

**定义：** off-policy 校正；`vanilla_tis_function` in loss.py。

→ [[Slime-Advantage计算-排障指南]]

---

## U

### update_weights

**定义：** 训练后将 Megatron 权重推送到 SGLang engine；闭环最后一步。

**代码：**

```python
## 来源：train.py L89
        actor_model.update_weights()
```

→ [[Slime-分布式权重同步-核心概念]]

---

## 与 SGLang 术语对照

| Slime | SGLang 对应 |
|-------|------------|
| SGLangEngine | `run_server` / HTTP Server |
| generate HTTP | `/generate` → Scheduler |
| update_weights reload | 权重热更新 / CheckpointEngine |
| PD 拓扑 | Disaggregation P/D |

→ [[Slime与SGLang-阅读对照]]

---

## 导航

- [[Slime-关键概念]]
- [[Slime学习指南]]
