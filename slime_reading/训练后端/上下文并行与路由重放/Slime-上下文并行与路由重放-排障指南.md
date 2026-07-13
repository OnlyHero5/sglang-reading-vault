---
title: "上下文并行与路由重放 · 排障指南"
type: troubleshooting
framework: slime
topic: "上下文并行与路由重放"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-13
---
# 上下文并行与路由重放 · 排障指南

本篇按症状排障。先判断问题属于 CP 坐标、loss reducer、allgather-CP，还是 RoutingReplay stage。

## 症状速查

| 症状 | 最可能原因 | 源码入口 | 验证方法 |
|------|------------|----------|----------|
| `log_prob length mismatch` | full response 与 CP local chunk 混用 | `cp_utils.py` L320-L344 | 打印 `response_length` 和 `logits_offset` |
| `full_loss_masks.shape != tokens.shape` | `get_batch` 中 token/mask pad 或 slice 不同步 | `data.py` L120-L148 | 对比 allgather-CP 与 zigzag 分支 |
| GSPO/OPSM 数值随 CP 变化 | full response gather 或 loss_mask 分母错误 | `cp_utils.py` L235-L284 | 跑 `test_loss_cp_invariance.py` |
| per-rollout mean 偏大 | denominator 在 micro-batch 内临时算了 | `cp_utils.py` L47-L124 | 跑 `test_cp_utils.py` sibling split 测试 |
| allgather-CP 训练 hang | 空 rank 没有反向图 | `loss.py` L1281-L1287 | 检查 `0 * logits.sum()` 是否保留 |
| `rollout_routed_experts` missing | 开了 rollout routing replay，但 rollout 没返回 experts | `actor.py` L284-L288 | 检查 rollout payload 和 SGLang server args |
| replay shape mismatch | CP/SP 切片和训练 batch 不一致 | `actor.py` L313-L355 | 对比 `slice_with_cp`、TP/SP 切分 |
| ref logprob 与 actor routing 不匹配 | ref 应走 `fallthrough`，不应 replay rollout experts | `actor.py` L441-L456 | 打印 `ROUTING_REPLAY_STAGE` |
| allgather-CP + rollout replay token/expert 错位 | expert ids 固定走 zigzag，tokens 走 contiguous global chunk | `actor.py` L313-L331、`data.py` L69-L88 | 禁用其中一项做 A/B，并逐 token 对齐局部 ids |
| 一次异常后下一 step 游标越界 | stage/buffer/游标无统一 finally 清理 | `actor.py` L436-L539、`model.py` L602-L638 | 异常注入后检查 env、两个 index 与 list 长度 |
| reducer 少算样本但未报错 | `zip(strict=False)` 静默截短 | `cp_utils.py` L47-L124 | 比较 lengths、masks、denoms 与 split 数量 |

## 为什么 `all_gather_with_cp` 用 all_reduce？

每个 CP rank 把自己负责的 response 区间填到 full tensor，其余位置为零。各 rank 的有效区间互斥，所以 sum reduce 后就是完整 response。

源码入口：来源：slime/backends/megatron_utils/cp_utils.py L235-L284

这里用 `dist.nn.all_reduce` 是为了保留 autograd 路径；不是为了偷懒替代普通 gather。

## 为什么 `cu_seqlens` 要乘 `cp_size`？

zigzag CP 中物理 tensor 已经被切成 rank local chunk，但 Megatron THD packed attention 需要逻辑序列长度。`get_batch` 在 zigzag 分支把 local `cu_seqlens` 乘 `cp_size`，让 attention 看到逻辑长度。

源码入口：来源：slime/backends/megatron_utils/data.py L88-L105

allgather-CP 分支不同：它先构造全局 `cu_seqlens`，再对 token stream chunk，所以不走同样的乘法。

## 为什么 per-rollout mean 不用本地 mask sum 做分母？

因为同一个 rollout 的 sibling samples 可能被分到不同 micro-batch。若每个 micro-batch 用自己的局部 mask sum，会把同一个 rollout 切成多个平均数，最终贡献偏大。

`tests/test_cp_utils.py` 专门固定这个契约：

源码入口：来源：tests/test_cp_utils.py L64-L126

排查方法：检查 `rollout_mask_sums` 是否来自 whole step，而不是当前 micro-batch。

## `use_rollout_routing_replay` 与 `use_routing_replay` 是什么关系？

`use_routing_replay` 是基础能力：patch MoE routing，让 `compute_topk` 支持 record/replay/fallthrough。

`use_rollout_routing_replay` 是更强的模式：要求 rollout server 返回 expert ids，并在 actor 训练前预填 replay buffer。

参数层会自动提升：

```python
# 来源：slime/utils/arguments.py L1950-L1952
if args.use_rollout_routing_replay:
    args.use_routing_replay = True
```

## ref/teacher 为什么走 `fallthrough`？

ref/teacher 是 KL 或 OPD 的对照分布。让它们 replay rollout actor 的 experts 会污染对照语义。actor 在 ref/teacher `compute_log_prob` 前显式设置 `fallthrough`。

源码入口：来源：slime/backends/megatron_utils/actor.py L441-L456

## `replay_forward` 和 `replay_backward` 为什么要两个游标？

Megatron pipeline 训练中 forward 和 backward 消费 replay buffer 的时机不同。一个游标会在 forward 消费后让 backward 读错位置；所以 `RoutingReplay` 分别维护 `forward_index` 和 `backward_index`。

源码入口：来源：slime/utils/routing_replay.py L13-L45

当 `use_rollout_routing_replay` 时，old actor logprob 之后会 `clear_all_forward()`，让正式 policy forward 能从同一组 prefilled experts 的开头重放。

源码入口：来源：slime/backends/megatron_utils/actor.py L482-L495

注意 `clear_all_forward()` 只重置 forward 游标，不重置 backward 游标或 buffer；step 末尾才 `clear_all()`。而 actor/model 的 stage 切换和最终清理没有包在统一 `try/finally` 中，异常后不能假设下一 step 会自动从干净状态开始。

## 为什么 replay 仍然有梯度？

replay 只固定 expert indices。概率仍然从当前 router `scores` 里 gather，所以梯度仍流向当前 scores。

源码入口：来源：slime/utils/routing_replay.py L66-L78

如果你看到 router 梯度消失，先确认代码是否错误地复用了旧 probs，而不是只复用 `top_indices`。

还要检查全局指针：每个 MoE module 的 forward pre-hook 会把自己的 replay buffer 写到进程级 `ROUTING_REPLAY`。这依赖 module forward 与 patched `compute_topk` 紧邻且不被并发/重入打断；它不是线程隔离的上下文对象。

## allgather-CP 和 zigzag CP 怎么选？

`allgather_cp` 是 DSA 相关路径，`get_batch` 会走全局 cat 后 contiguous chunk。普通 CP 走 zigzag `slice_with_cp`。两者都还要处理 response logprob 对齐，但坐标来源不同。

源码入口：

- 来源：slime/backends/megatron_utils/data.py L69-L148
- 来源：slime/backends/megatron_utils/loss.py L151-L227

参数校验可看 `tests/test_megatron_argument_validation.py` 中 allgather-CP 相关测试。

但这些校验只限制 allgather-CP 的模型架构，没有禁止它与 rollout routing replay 同开。当前 `fill_routing_replay` 又没有 allgather-CP 分支，所以在补齐实现或组合测试前，应把这组配置视为未证明兼容，而不是推荐组合。

## 改 CP 或 RoutingReplay 前跑什么？

首选：

```powershell
Set-Location slime
python -m pytest tests/test_cp_utils.py
python -m pytest tests/test_logprob_response_spans.py
```

涉及 loss scaling 或 CP/DP 等价时：

```powershell
python -m pytest tests/test_loss_cp_invariance.py
```

文档侧检查：

```powershell
node maintenance/audit_source_evidence.mjs
node maintenance/audit_wikilinks.mjs
```
