---
title: "PD分离 · 学习检查"
type: exercise
framework: sglang
topic: "PD分离"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-11
---
# PD分离 · 学习检查

这份清单用来判断你是否真的读懂 PD 分离，而不是只看过 Prefill、Decode、Mooncake 这些名词。验收标准是：能沿一条请求解释状态变化，能把症状映射到源码入口，能说出修改代码前必须守住的不变量。

## 读者能做什么

- [ ] 能画出 Gateway 一次 attempt 并行双发：`Prefill HTTP → Prefill TM → Prefill Scheduler` 与 `Decode HTTP → Decode TM → Decode Scheduler`，再用 room、KV transfer 和 metadata 会合。
- [ ] 能说明 `bootstrap_room` 为什么同时影响 sender/receiver、DP 路由和 metadata 校验。
- [ ] 能区分 Prefill 三队列和 Decode 四队列，并说出每个队列等待的资源。
- [ ] 能解释为什么 `KVPoll.Success` 之后还要经过 metadata gate、all-reduce、staging 或 HiCache restore。
- [ ] 能说明 `ForwardMode.PREBUILT` 为什么不是普通 prefill，也不是普通 decode。
- [ ] 能区分 Gateway 整对 retry 与 Prefill Scheduler optimistic retry。

## 闭卷复述

用 2 分钟复述这条链：

1. Gateway 选择一对 worker、生成 room，并行向 Prefill、Decode 两个独立服务发请求；两侧各有自己的 TokenizerManager 和 Scheduler。
2. 默认稳态路径中，Decode 创建 receiver，解析 Prefill DP rank，预分配 KV slot 和 metadata slot，并把 page indices 发给 receiver。
3. Prefill 创建 sender；默认等 bootstrap 完成后拿到 decode prefix 长度，再计算只需发送的 KV page。
4. 若显式开启 optimistic prefill，`Bootstrapping` 状态也可先 forward；握手没有及时完成时必须释放本轮 KV、reset 并 requeue，不能把这轮结果直接提交。
5. Prefill extend forward 产生首 token，写入 metadata buffer，再发送 KV chunk。
6. Decode poll receiver，并用 metadata 中的 `bootstrap_room` 确认 metadata 已落地。
7. Decode commit 首 token 和 cached token 统计，释放 metadata slot 前把 room 重置为 0。
8. Decode 构造 `PREBUILT` batch，把请求合入 running batch 继续逐 token decode。

## 断点验证

| 验证目标 | 断点入口 | 预期现象 |
|----------|----------|----------|
| room 进入 Scheduler | `handle_generate_request` | `recv_req.bootstrap_room` 写入 `Req.bootstrap_room` |
| 稳态 Decode 占位 | `_create_receiver_and_enqueue` | `kv_receiver` 使用同一个 `bootstrap_room` |
| Prefill rank 解析 | `_resolve_pending_reqs` | 注入 rank、本地 `room % dp_size` 或远程 query 最终初始化 receiver |
| metadata slot 建立 | `DecodePreallocQueue.pop_preallocated` | `metadata_buffer_index` 非空并传给 receiver |
| Prefill 计算发送范围 | `PrefillBootstrapQueue.finalize_bootstrap` | `start_send_idx` 等于 `decode_prefix_len` |
| optimistic 失败回收 | `optimistic_release_and_requeue` | KV 被释放、请求 reset、retry count 增加并重新排队 |
| Prefill 写 metadata | `MetadataBuffers.set_buf` | `bootstrap_room[idx, 0]` 从 0 变成当前 room |
| Decode gate 生效 | `_apply_metadata_gate` | room 为 0 时 Success 被降为 Transferring |
| commit 成功 | `_commit_transfer_to_req` | `Req.output_ids` 追加 Prefill 首 token |
| 进入 running | `get_next_disagg_decode_batch_to_run` | `new_prebuilt_batch` 被过滤后 merge 到 running |

## 失败模式

- [ ] room 缺失：能定位到 Scheduler bad request 或 DP `follow_bootstrap_room_scheduler` 断言。
- [ ] Prefill bootstrap 堵住：能先查 Decode receiver 和 `send_metadata`，而不是只看 Prefill GPU。
- [ ] Prefill GPU 有计算但没有传输：能检查是否启用了 optimistic prefill、`pending_bootstrap` 是否仍为真，以及是否发生 release/requeue。
- [ ] Decode 长时间不进入 prealloc：能检查 `pending_reqs` 是否在等 Prefill parallel info 或 room→DP rank query。
- [ ] transfer Success 但 Decode 不动：能检查 metadata gate、staging、HiCache restore 和 all-reduce。
- [ ] metadata mismatch：能解释为什么可能是 room 复用、metadata slot 复用或 buffer 未归零。
- [ ] decode waiting 堆积：能区分 running 空位不足、grammar 未 ready 和 prebuilt batch 未合入。
- [ ] 启动参数失败：能从 radix cache、fake backend、speculative、HiSparse、staging backend 的互斥关系解释。

## 修改代码前

- [ ] 改请求路由前，确认 batch room normalize、DP follow room 和 `Req.bootstrap_room` 三处一致。
- [ ] 改 transfer backend 前，确认 `KVPoll` 数值顺序仍满足 MIN all-reduce 语义。
- [ ] 改 metadata buffer 前，确认 0 仍表示未写入，commit 后仍会归零并释放 slot。
- [ ] 改 prealloc 容量前，确认 running 上限和 in-transfer 额外槽位没有混淆。
- [ ] 改 HiCache 或 staging 前，确认 raw receiver Success 不会绕过本地 restore/scatter gate。
- [ ] 改 retry 前，确认 Gateway retry 会换 pair/room 并整对重放，而 optimistic retry 只回收 Prefill Scheduler 内部这轮 speculative 计算。

## 下一步

如果这份清单都能闭卷通过，继续读 [[SGLang-分布式]] 看 rank group 与 all-reduce 事实来源；读 [[SGLang-KV-Cache]] 看 KV slot 与物理 tensor 生命周期；读 [[SGLang-Speculative]] 看 PD `PREBUILT` 如何接 speculative draft input。
