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
updated: 2026-07-10
---
# 分布式 · 学习检查

## 读者能做什么

- [ ] 能画出 `WORLD → TP/PP/DCP/Attention/MoE group → GroupCoordinator → communication_op` 的主线。
- [ ] 能解释为什么 `world_size` 在 `initialize_model_parallel` 中只校验 `tp_size * pp_size`。
- [ ] 能区分模型 tensor collective、DP 请求路由、PD CPU poll、Elastic EP recovery 四条链路。
- [ ] 能说出 `local_rank`、global rank、`rank_in_group` 的差异。
- [ ] 能指出 layer 代码为什么应调用 `communication_op.py` helper。
- [ ] 能从一个症状反推出源码入口：启动校验、getter assert、`GroupCoordinator.all_reduce`、DP dispatch、PD poll 或 Mooncake recovery。

## 最小验证

| 验证目标 | 操作 | 预期现象 |
|----------|------|----------|
| TP/PP 配置合法性 | 对照启动进程数与 `tp_size * pp_size` | 不相等时在模型并行初始化前失败 |
| helper 入口 | 在 `tensor_model_parallel_all_reduce` 打断点 | 模型层进入 `get_tp_group().all_reduce` |
| backend 选路 | 在 `GroupCoordinator.all_reduce` 观察 tensor 与 communicator 状态 | 单卡 bypass；CPU tensor 走 CPU 路径；GPU tensor 按 communicator 和 graph 条件选路 |
| DP 路由 | 构造带 `routed_dp_rank` 或 `bootstrap_room` 的请求 | 直接路由优先；`FOLLOW_BOOTSTRAP_ROOM` 按 room 取模 |
| PD poll | 观察 `poll_and_all_reduce` 的 group 与 tensor device | 状态以 CPU tensor 在 gloo group 上取 `MIN` |
| Elastic EP | 观察 `try_recover_ranks` 返回值 | peer 未 ready 返回 `False`；ready 后恢复 WORLD 和 live groups |

## 自检题

1. 如果 `tp_size=4, pp_size=2, dp_size=2`，单个模型并行 worker 的 `world_size` 应该是多少？为什么不是 16？
2. `rank=3` 的 `local_rank`、TP `rank_in_group`、PP `rank_in_group` 可以相同吗？什么时候不同？
3. `FOLLOW_BOOTSTRAP_ROOM` 为什么要求请求不能直接打到 prefill 或 decode 实例？
4. PD poll 为什么使用 CPU/gloo group，而不是 TP NCCL group？
5. Elastic EP recovery 为什么要先恢复 WORLD，再恢复每个 live `GroupCoordinator`？

## 迁移结论

- 分布式文档不能只罗列 TP、PP、DP、EP 缩写；要先建立 rank 坐标系。
- 源码证据要服务一条执行链，而不是按文件顺序堆叠。
- 排障入口要按症状分流：启动、collective、路由、poll、recovery。
- 对读者来说，最重要的不变量是“对象决定链路”：tensor、request、poll status、active rank mask 对应不同源码入口。
