---
title: "分布式 · 学习检查"
type: exercise
framework: sglang
topic: "分布式"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# 分布式 · 学习检查

## 你为什么要做这些练习

分布式知识最容易停留在缩写背诵。本页要求你亲手计算 group、定位 alias、按对象分流，并写出一张可执行排障卡。完成后，你应当能用源码证明：

- 当前 `world_size` 属于哪个 scheduler 模型 WORLD，为什么外层 `dp_size` 不直接进入 `TP × PP` 校验；
- 同一 global rank 在 TP、PP、Attention、MoE、DCP 中为何有不同 `rank_in_group`；
- group 语义名与 coordinator 对象为何不是一一对应；
- model tensor、DP request、PD poll status、Elastic EP membership 分别走哪条通道；
- 一个 timeout/错路由/recovery 症状应落到哪个源码入口。

---

## 练习 1：手算 TP 与 PP group

设 `world_size=8, tp_size=4, pp_size=2`。

1. 按 TP 连续切分公式写出所有 TP groups。
2. 计算 `num_pipeline_model_parallel_groups = world_size // pp_size`。
3. 按 `range(pp_group_idx, world_size, num_pipeline_model_parallel_groups)` 写出所有 PP groups。

预期答案：

- TP：`[0,1,2,3]`、`[4,5,6,7]`；
- PP：`[0,4]`、`[1,5]`、`[2,6]`、`[3,7]`。

用 PowerShell 验算：

```powershell
$world=8; $pp=2; $stride=$world/$pp
0..($stride-1) | ForEach-Object { "[$_, $($_+$stride)]" }
```

预期输出四行，不能得到 `[0,2,4,6]`；后者对应 `tp=2, pp=4` 的另一组参数。

---

## 练习 2：找出 alias，而不是只数 group 名

```powershell
rg -n "_ATTN_CP = _TP|_ATTN_TP = _TP|_MOE_DP = _ATTN_CP|_MOE_DP = _TP|_MOE_EP = _TP|_MOE_TP = _TP" `
  sglang/python/sglang/srt/distributed/parallel_state.py
```

预期：至少命中 Attention 与 MoE 的多条复用分支。

回答：若 `_MOE_EP is _TP`，能否仍然断言 MoE EP 一定关闭 custom all-reduce？正确答案是否定的；别名分支继承 TP coordinator，只有新建 MoE EP 的分支显式传入关闭参数。

---

## 练习 3：按对象给调用链分类

把下面入口分别归入 tensor、request、poll status、membership：

```powershell
rg -n "def tensor_model_parallel_all_reduce|def maybe_external_dp_rank_routing|def poll_and_all_reduce\(|def try_recover_ranks" `
  sglang/python/sglang/srt/distributed/communication_op.py `
  sglang/python/sglang/srt/managers/data_parallel_controller.py `
  sglang/python/sglang/srt/disaggregation/utils.py `
  sglang/python/sglang/srt/elastic_ep/elastic_ep.py
```

预期映射：

| 入口 | 对象 | 主要通道 |
|---|---|---|
| TP helper | GPU/设备 tensor | getter → `GroupCoordinator` → communicator |
| external DP routing | request object | ZMQ worker socket |
| PD poll | CPU uint8 status tensor | 调用者提供的 coordination group + `MIN` |
| Elastic recovery | global/group-local rank membership | Mooncake backend recovery |

---

## 练习 4：解释 coordination group 的条件边界

```powershell
rg -n 'backend="mooncake-cpu"|backend="gloo"|if input_\.is_cpu|group=self\.device_group' `
  sglang/python/sglang/srt/distributed/parallel_state.py
```

预期同时看到：

- 普通 group 创建 Gloo `cpu_group`；
- Mooncake group 创建 `mooncake-cpu`；
- `GroupCoordinator.all_reduce` 的 CPU-tensor fallback 使用当前 `device_group`，不是无条件使用 `cpu_group`。

所以“CPU 数据 = Gloo”不是全局不变量，必须看具体 API。

---

## 练习 5：写一张可执行排障卡

从以下场景任选一个：

- eager 正常、CUDA Graph collective 失败；
- `FOLLOW_BOOTSTRAP_ROOM` 请求落到 inactive rank；
- PD 某些 rank 已 Success，其他 rank 仍 Transferring；
- Elastic EP rejoin 后 WORLD ready，但 MoE membership 未刷新。

你的答案必须包含：

1. 症状与作用域；
2. 当前对象；
3. 源码入口；
4. 要记录的 group membership/backend 或 request 字段；
5. 操作；
6. 预期；
7. 环境不足时的静态替代。

合格标准：不能只写“检查 NCCL”“检查网络”或“调小 batch”；必须先证明故障属于哪条分布式流。

---

## 最终自检

1. `tp=4, pp=2, dp=2` 时，为什么当前模型 WORLD 的校验值仍是 8，而整套普通 DP 部署可能使用更多设备？
2. `local_rank`、global rank、`rank_in_group` 分别回答什么问题？
3. 为什么 Attention TP 与 MoE EP 的 backend 策略必须先判断是否 alias TP？
4. 为什么 PD poll 使用 CPU tensor，却不能无条件写成“Gloo all-reduce”？
5. 为什么 `routed_dp_rank` 与 `FOLLOW_BOOTSTRAP_ROOM` 需要上层保证目标健康？
6. Elastic EP recovery 为什么先恢复 WORLD，再将 global ranks 映射到每个 live group 的 local ranks？

六题都能结合源码对象回答，才算真正掌握本专题。
