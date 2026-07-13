---
title: "上下文并行与路由重放 · 学习检查"
type: exercise
framework: slime
topic: "上下文并行与路由重放"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-13
---
# 上下文并行与路由重放 · 学习检查

## 读者能做什么

- [ ] 能画出 zigzag CP 中 rank `r` 拿两段 chunk 的布局。
- [ ] 能解释 response token 与 logits 行之间的 `-1` 对齐关系。
- [ ] 能说明 `slice_log_prob_with_cp` 与 `all_gather_with_cp` 分别承担 full→local 与 local→full 的相反方向，但不把它们误写成无条件数学逆函数。
- [ ] 能解释 `get_sum_of_sample_mean` 为什么使用 full rollout denominator。
- [ ] 能区分 zigzag CP 与 allgather-CP 的输入布局。
- [ ] 能列出 RoutingReplay 的 `fallthrough`、`record`、`replay_forward`、`replay_backward` 四个 stage。
- [ ] 能说明 replay 固定 expert ids，但不冻结当前 router scores。
- [ ] 能解释 ref/teacher forward 为什么不能 replay rollout experts。
- [ ] 能说明 allgather-CP 模型输入为何会被重分布成 zigzag-local response list。
- [ ] 能指出 allgather-CP 与 rollout routing replay 当前缺少同布局实现和参数互斥。
- [ ] 能把 stage、全局 replay 指针、forward/backward 游标和 buffer 作为一组失败状态检查。

## 源码入口自测

- [ ] CP offset/slice/gather/reducer：`slime/backends/megatron_utils/cp_utils.py` L9-L344。
- [ ] CP-ready batch：`slime/backends/megatron_utils/data.py` L28-L148。
- [ ] actor routing replay 编排：`slime/backends/megatron_utils/actor.py` L284-L539。
- [ ] training forward stage 临时切换：`slime/backends/megatron_utils/model.py` L602-L638。
- [ ] routing replay 状态机：`slime/utils/routing_replay.py` L13-L92。
- [ ] rollout routed experts 开关：`slime/backends/sglang_utils/sglang_engine.py` L625-L627，`slime/rollout/sglang_rollout.py` L174-L182。

## 可执行验证

- [ ] 从知识库根目录执行 `Set-Location slime`，再运行 `python -m pytest tests/test_cp_utils.py`，确认 reducer 与 CP chunk 契约。
- [ ] 运行 `python -m pytest tests/test_logprob_response_spans.py`，确认 top-p mask 与 CP response row 对齐。
- [ ] 运行 `python -m pytest tests/test_loss_cp_invariance.py`，确认 CP/DP 切分不改变 loss/grad 语义。
- [ ] 用 `rg -n 'slice_log_prob_with_cp|all_gather_with_cp|RoutingReplay|replay_forward|replay_backward' slime/slime` 定位切分/还原与 replay 状态机。

预期：测试覆盖 reducer、response span 和 loss/grad 不变性；静态结果必须能分出 CP 数据布局与 routing replay 控制流。

## 排障演练

- [ ] 构造一个 `response_length` 很短的样本，能说明某个 CP rank 空 chunk 为什么不是异常。
- [ ] 构造 `use_rollout_routing_replay=True` 的路径，能说出 `rollout_routed_experts` 从哪里来、在哪里被删除。
- [ ] 构造 ref forward 路径，能说出为什么 stage 是 `fallthrough`。
- [ ] 构造 GSPO 路径，能说出哪里从 local logprob 还原 full response。
- [ ] 同时打开 allgather-CP 与 rollout routing replay，能证明 expert ids 与 token 的局部布局一致；证明不了时应判定为未支持组合。
- [ ] 在 old actor forward 或 policy train 中注入异常，能检查下一 step 前 stage、两个游标和 buffer 是否已恢复。

## 迁移结论

这组文档读懂后，再回看 [[Slime-Advantage计算]] 和 [[Slime-Policy-Loss]] 的 CP 分支会更清楚：它们不是额外算法，而是在同一条 response 被切分后，保持 token、mask、expert 和 metric 语义一致。
