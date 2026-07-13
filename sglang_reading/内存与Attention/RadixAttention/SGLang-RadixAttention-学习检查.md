---
title: "RadixAttention · 学习检查"
type: exercise
framework: sglang
topic: "RadixAttention"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-11
---
# RadixAttention · 学习检查

## 读者能做什么

- [ ] 能画出 `Req`、`RadixKey`、`TreeNode`、KV pool、`RadixAttention.forward` 五个对象和它们之间的四条边。
- [ ] 能沿共享 2k token system prompt 的请求复述：match → 写 `prefix_indices` → lock → extend tail → attention → unfinished/finished cache。
- [ ] 能说明 `extra_key`、`page_size`、`cache_protected_len`、`lock_ref` 各自保护什么不变量。
- [ ] 能区分 match 刚结束时的 device-hit `prefix_indices`，与 chunk commit 后可能附带私有 tail 的 `prefix_indices`。
- [ ] 能解释为什么 `RadixAttention.forward` 不负责 tree match。
- [ ] 能说出 classic `RadixCache` 和 `UnifiedRadixCache` 的边界差异。

## 场景推演

1. 两个请求 token 完全相同，但 LoRA id 不同。你应该判断为不能共享，并能指出 `extra_key` 和 `child_key` 两个源码入口。
2. page size 是 16，请求当前有 2047 个 token。你应该判断 tree 只能接管完整 page，未对齐 tail 需要由 `cache_protected_len` 相关逻辑处理。
3. chunked prefill 中途完成一个 chunk。你应该能说明为什么要 insert 后再 match，以及为什么要改写 `req_to_token_pool`。
4. 显存压力触发 evict。你应该能说明 classic cache 为什么从 leaf 开始，为什么释放 token 数可能超过目标。
5. piecewise CUDA Graph 打开后 attention 输出异常。你应该先检查 `real_num_tokens`、query 切片和 `out_cache_loc` 切片是否一致。

## 排障入口

| 症状 | 优先入口 | 预期观察 |
|------|----------|----------|
| 第二次相同 prompt 没变快 | `match_prefix_for_req` | `len(req.prefix_indices)` 是否增长 |
| 同 prompt 部分命中 | `RadixKey.page_aligned` | 命中长度是否按 page 向下取整 |
| LoRA 场景串 cache | `Req.__init__`、`RadixKey.child_key` | `extra_key` 是否不同 |
| chunked prefill 显存上涨 | `cache_unfinished_req` | duplicate 和 tail 是否被释放 |
| 请求运行中 KV 被释放 | `inc_lock_ref` / `dec_lock_ref` | lock 是否成对、node 是否同 tree |
| attention 写错 slot | `ScheduleBatch.prepare_for_extend`、`RadixAttention.forward` | `out_cache_loc` 是否与真实 token 数一致 |

## 验证实验

- [ ] 用同一 system prompt 连续请求两次，记录 TTFT 和 prefill token 数。
- [ ] 设置 `SGLANG_RADIX_FORCE_MISS=1` 重复实验，先验证第二次请求的 prefix hit/extend-token 优势消失，再解释 TTFT 变化。
- [ ] 启动时使用 `--disable-radix-cache` 做全局关闭对照。
- [ ] 在 `match_prefix_for_req`、`prepare_for_extend`、`RadixAttention.forward` 三处断点，确认 tree match 与 attention forward 不在同一层发生。

## 能力标准

达到通过标准时，你应该能用一句话说清本专题：

> `RadixCache` 在调度侧把 token 前缀映射到可复用 KV pool indices，`ScheduleBatch` 用 `prefix_indices` 只构造尚未计算的 extend 区间，`RadixAttention` 再把这些 tensor 转交给 attention backend 读写 paged KV；tree ownership 则由 `cache_protected_len` 单独界定。

## 后续阅读

- [[SGLang-KV-Cache]]：继续看 KV pool、page allocator、req pool 的物理内存管理。
- [[SGLang-Attention]]：继续看 attention backend 如何消费 `ForwardBatch` metadata。
- [[SGLang-Scheduler]]：回到调度层，看 waiting queue、admission budget 与 prefix cache policy。
