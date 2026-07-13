---
title: "SchedulePolicy · 学习检查"
type: exercise
framework: sglang
topic: "SchedulePolicy"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-11
---
# SchedulePolicy · 学习检查

## 读者能做什么

- [ ] 能画出 `waiting_queue → calc_priority → PrefillAdder → can_run_list → ScheduleBatch` 的主线。
- [ ] 能解释 `SchedulePolicy` 只排序，`PrefillAdder` 才做本轮准入。
- [ ] 能说出 `prefix_indices`、`num_matched_prefix_tokens`、`extend_range` 三个字段的关系。
- [ ] 能解释 `NO_TOKEN` 和 `OTHER` 都会停止本轮扫描、但只有前者驱动 `batch_is_full` 更新，并能定位具体返回点。
- [ ] 能解释为什么 chunked prefill 进行中不能被普通 slot delay 打断。
- [ ] 能说明 `PrefillDelayer` 是一轮 prefill pass 的跨 rank 协商，不是单请求采样逻辑。
- [ ] 能解释未完成 chunk 保存在 `Scheduler.chunked_req` 专用槽位，而不是回到普通 `waiting_queue`。

## 源码定位自测

| 问题 | 应定位到 |
|------|----------|
| 为什么 `lpm` 没排序 | `SchedulePolicy.calc_priority`、`_determine_active_policy` |
| prefix cache 命中写到了哪里 | `match_prefix_for_req` |
| 本轮为什么只准入一个请求 | `PrefillAdder.add_one_req`、`budget_state` |
| chunked prefill 下一轮怎么继续 | `add_chunked_req`、Scheduler 的 `self.chunked_req` 更新 |
| prefill 为什么整轮被推迟 | `MinFreeSlotsDelayer.should_delay` 或 `PrefillDelayerSinglePassExecutor` |
| 优先级抢占为什么没发生 | `preempt_to_schedule` |

## 运行验证自测

- [ ] 能设计 `lpm` 对比 `fcfs` 的共享 system prompt 实验，先验证排序与 prefix-hit 变化，再解释为何不能预设 TTFT 一定改善。
- [ ] 能用 `SGLANG_RADIX_FORCE_MISS=1` 解释 prefix cache 收益是否真实。
- [ ] 能通过断点或日志观察 `AddReqResult.NO_TOKEN` 与 `AddReqResult.OTHER`。
- [ ] 能观察一次长 prompt 的 `Scheduler.chunked_req` 从非空到清空，并区分“本轮未推进”与“已经提交最后一块”。
- [ ] 能在 delayer metrics 或 debug 日志中解释 `delay`、`wait_timeout`、`token_watermark`。

## 最小静态验收

操作：在仓库根目录运行下面的定位命令，按“排序 → 普通 waiting 准入 → 专用 chunk 续跑 → single-pass 协商”顺序阅读命中。

```powershell
rg -n 'def calc_priority|def add_one_req\(|def add_chunked_req|class PrefillDelayerSinglePassExecutor|def preempt_to_schedule' sglang/python/sglang/srt/managers/schedule_policy.py sglang/python/sglang/srt/managers/prefill_delayer.py
```

预期：五个入口都存在；`calc_priority` 不创建 `ScheduleBatch`，`add_one_req` 返回 `AddReqResult`，`add_chunked_req` 返回 `req`/`None`，single-pass executor 缓存第一次协商结果，`preempt_to_schedule` 只负责抢占 commit 而不绕过后续准入检查。若任一契约变化，先更新源码主线，再修改本页答案。

## 复盘问题

1. 如果 waiting queue 很长，`lpm` 为什么会临时退化，退化后是否一定执行 priority+FCFS 排序？
2. 如果 `prefix_indices` 很长但 `extend_range` 仍覆盖完整 prompt，最可能是哪层字段契约被破坏？
3. 如果 `can_run_list` 为空但 waiting queue 不空，哪些门可能在 `ScheduleBatch.init_new` 之前把本轮挡掉？
4. 如果开启 prefill delayer 后 TTFT 上升但 decode throughput 变好，这是否一定是 bug？
5. 如果要改一个新 policy，你会把排序逻辑放在哪里，把准入预算逻辑放在哪里？
6. 为什么 `PrefillDelayer` all-gather 的是低水位布尔量而不是原始 token usage？`skip_first_delayer`、pass 超时和墙钟超时分别保护什么？
