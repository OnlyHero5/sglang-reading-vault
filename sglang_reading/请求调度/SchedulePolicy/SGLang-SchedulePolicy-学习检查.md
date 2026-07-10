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
updated: 2026-07-10
---
# SchedulePolicy · 学习检查

## 读者能做什么

- [ ] 能画出 `waiting_queue → calc_priority → PrefillAdder → can_run_list → ScheduleBatch` 的主线。
- [ ] 能解释 `SchedulePolicy` 只排序，`PrefillAdder` 才做本轮准入。
- [ ] 能说出 `prefix_indices`、`num_matched_prefix_tokens`、`extend_range` 三个字段的关系。
- [ ] 能区分 `NO_TOKEN` 和 `OTHER`，并给出各自的源码排查入口。
- [ ] 能解释为什么 chunked prefill 进行中不能被普通 slot delay 打断。
- [ ] 能说明 `PrefillDelayer` 是一轮 prefill pass 的跨 rank 协商，不是单请求采样逻辑。

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

- [ ] 能设计 `lpm` 对比 `fcfs` 的共享 system prompt 实验，并说出预期 TTFT 与 prefix hit 变化。
- [ ] 能用 `SGLANG_RADIX_FORCE_MISS=1` 解释 prefix cache 收益是否真实。
- [ ] 能通过断点或日志观察 `AddReqResult.NO_TOKEN` 与 `AddReqResult.OTHER`。
- [ ] 能观察一次长 prompt 的 `new_chunked_req` 从非空到清空。
- [ ] 能在 delayer metrics 或 debug 日志中解释 `delay`、`wait_timeout`、`token_watermark`。

## 复盘问题

1. 如果 waiting queue 很长，`lpm` 为什么会临时退化，退化后是否一定执行 priority+FCFS 排序？
2. 如果 `prefix_indices` 很长但 `extend_range` 仍覆盖完整 prompt，最可能是哪层字段契约被破坏？
3. 如果 `can_run_list` 为空但 waiting queue 不空，哪些门可能在 `ScheduleBatch.init_new` 之前把本轮挡掉？
4. 如果开启 prefill delayer 后 TTFT 上升但 decode throughput 变好，这是否一定是 bug？
5. 如果要改一个新 policy，你会把排序逻辑放在哪里，把准入预算逻辑放在哪里？
