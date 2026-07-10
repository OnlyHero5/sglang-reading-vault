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
updated: 2026-07-10
---
# 上下文并行与路由重放 · 学习检查

## 读者能做什么

- [ ] 能画出 zigzag CP 中 rank `r` 拿两段 chunk 的布局。
- [ ] 能解释 response token 与 logits 行之间的 `-1` 对齐关系。
- [ ] 能说明 `slice_log_prob_with_cp` 与 `all_gather_with_cp` 是互逆方向的工具。
- [ ] 能解释 `get_sum_of_sample_mean` 为什么使用 full rollout denominator。
- [ ] 能区分 zigzag CP 与 allgather-CP 的输入布局。
- [ ] 能列出 RoutingReplay 的 `fallthrough`、`record`、`replay_forward`、`replay_backward` 四个 stage。
- [ ] 能说明 replay 固定 expert ids，但不冻结当前 router scores。
- [ ] 能解释 ref/teacher forward 为什么不能 replay rollout experts。

## 源码入口自测

- [ ] CP offset/slice/gather/reducer：`slime/backends/megatron_utils/cp_utils.py` L9-L344。
- [ ] CP-ready batch：`slime/backends/megatron_utils/data.py` L28-L148。
- [ ] actor routing replay 编排：`slime/backends/megatron_utils/actor.py` L284-L539。
- [ ] training forward stage 临时切换：`slime/backends/megatron_utils/model.py` L602-L638。
- [ ] routing replay 状态机：`slime/utils/routing_replay.py` L13-L92。
- [ ] rollout routed experts 开关：`slime/backends/sglang_utils/sglang_engine.py` L625-L627，`slime/rollout/sglang_rollout.py` L174-L182。

## 可执行验证

- [ ] 运行 `python -m pytest slime/tests/test_cp_utils.py`，确认 reducer 与 CP chunk 契约。
- [ ] 运行 `python -m pytest slime/tests/test_logprob_response_spans.py`，确认 top-p mask 与 CP response row 对齐。
- [ ] 运行 `python -m pytest slime/tests/test_loss_cp_invariance.py`，确认 CP/DP 切分不改变 loss/grad 语义。
- [ ] 运行 `node maintenance/audit_source_evidence.mjs --note slime_reading/训练后端/上下文并行与路由重放/Slime-上下文并行与路由重放-源码走读.md`，确认源码引用可追踪。
- [ ] 运行 `node maintenance/audit_wikilinks.mjs`，确认双链无断链。

## 排障演练

- [ ] 构造一个 `response_length` 很短的样本，能说明某个 CP rank 空 chunk 为什么不是异常。
- [ ] 构造 `use_rollout_routing_replay=True` 的路径，能说出 `rollout_routed_experts` 从哪里来、在哪里被删除。
- [ ] 构造 ref forward 路径，能说出为什么 stage 是 `fallthrough`。
- [ ] 构造 GSPO 路径，能说出哪里从 local logprob 还原 full response。

## 迁移结论

这组文档读懂后，再回看 [[Slime-Advantage计算]] 和 [[Slime-Policy-Loss]] 的 CP 分支会更清楚：它们不是额外算法，而是在同一条 response 被切分后，保持 token、mask、expert 和 metric 语义一致。
