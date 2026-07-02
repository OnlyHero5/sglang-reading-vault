---
type: batch-doc
module: 08-SchedulePolicy
batch: "08"
doc_type: faq
title: "调度策略：关键问题"
tags:
 - sglang/batch/08
 - sglang/module/schedule-policy
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# 调度策略：关键问题

## Q1：LPM 和 FCFS 该怎么选？

**Explain：** 默认 `--schedule-policy lpm` 在 prefix 共享明显的场景（相同 system prompt、RAG 文档前缀）能显著提高 cache hit。但当等待队列 >128 或 tree cache 禁用时，系统自动降级 FCFS。

| 场景 | 推荐 | 原因 |
|------|------|------|
| 多租户共享长 system prompt | `lpm` | 优先调度高 prefix 命中请求 |
| 队列经常 >128 | 接受自动 FCFS 或显式 `fcfs` | 避免每轮 O(n) 前缀匹配 |
| ChunkCache / disable radix | `fcfs` | cache-aware 策略会被强制降级 |
| 需要严格 FIFO 公平 | `fcfs` + priority | 时间戳排序 |
| PD 路由亲和 | `routing-key` | 与 running batch 同 key 优先 |

---

## Q2：`AddReqResult.NO_TOKEN` 和 `OTHER` 有什么区别？

**Explain：** Scheduler 对两者的处理略有不同——`NO_TOKEN` 通常意味着 KV 真的不够，会标记 `batch_is_full`；`OTHER` 可能是 delayer、chunk 上限、prefill_max_requests 等「软限制」。

**Code — Scheduler 对返回值的处理：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L2884-L2907
            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (not self.running_batch.is_empty())
                    else:
                        self.running_batch.batch_is_full = True
                # revert matched mamba idx to avoid memory leak, if req is not added.
                # Only free if the slot was freshly allocated in this batch (not
                # pre-existing from a session). Session-held slots have their own
                # lifecycle and freeing them here causes double-free.
                added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
                if (
                    not added
                    and req.mamba_pool_idx is not None
                    and not getattr(req, "session", None)
                ):
                    self.tree_cache.req_to_token_pool.mamba_allocator.free(
                        req.mamba_pool_idx.unsqueeze(-1)
                    )
                    req.mamba_pool_idx = None
                break
```

**Comment：**

- PrefillDelayer 拒绝时 `add_one_req` 返回 `OTHER`，**不会**因单个 delayer 就把 batch 标满——但若 `can_run_list` 为空则整轮返回 `None`。
- 运维上：频繁 `OTHER` + 空 batch 可能是 delayer 过 aggressive；频繁 `NO_TOKEN` 可能是 KV 容量或 `max_prefill_tokens` 过小。

---

## Q3：PrefillDelayer 与 MinFreeSlotsDelayer 会冲突吗？

**Explain：** 不会互斥，但叠加 delay 可能增加 TTFT。二者触发条件不同：

| 延迟器 | 触发条件 | 作用域 |
|--------|----------|--------|
| MinFreeSlots | `allocatable_reqs < min_free_slots` 且 `running_bs > 0` | 单 rank，prefill 轮次入口 |
| PrefillDelayer | slot 或 queue 条件 + 全局 all_gather | 跨 DP rank，`add_one_req` 内 |

**易错理解 vs 正确理解：**

```python
# ❌ 易错：以为 enable_prefill_delayer 会自动启用 MinFreeSlots
# PrefillDelayer 需要 --enable-prefill-delayer
# MinFreeSlots 需要 --min-free-slots-delay 或 DFlash 自动公式

# ✅ 正确：Scheduler 中的独立初始化
# 来源：scheduler.py L887-L896（MinFreeSlots）
min_free_slots = resolve_min_free_slots(
 self.server_args.min_free_slots_delay,
 self.max_running_requests,
 is_dflash=self.spec_algorithm.is_dflash(),
)

# 来源：scheduler.py L1049-L1056（PrefillDelayer）
if self.server_args.enable_prefill_delayer:
 if self.server_args.disaggregation_mode != "decode":
 self.prefill_delayer = PrefillDelayer(...)
```

---

## Q4：为什么 LPM 要对「批内共享前缀」的请求 deprioritize？

**Explain：** 假设 10 个请求共享 100 token 前缀，但 global radix 只命中 5 token。若 10 个同时 prefill，会重复计算 95 token × 10。策略让**第一个**请求先跑并 insert 到模拟树，其余同前缀请求被 deprioritize，下一轮 global radix 命中更高。

**Code — deprioritize 排序 key：**

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L303-L314
    @staticmethod
    def _sort_by_longest_prefix(
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        waiting_queue.sort(
            key=lambda r: (
                -r.num_matched_prefix_tokens
                if r.rid not in temporary_deprioritized
                else float("inf")
            )
        )
```

**Comment：** 阈值可通过 `IN_BATCH_PREFIX_CACHING_*` 环境变量调节；设为 `-1` 可禁用批内检查。

---

## Q5：`CLIP_MAX_NEW_TOKENS` 会影响生成长度吗？

**Explain：** **不会**。它只影响 Scheduler **估算** decode 占用多少 KV，防止 `max_new_tokens=100000` 的请求让 prefill 过度保守。实际 stop 仍由 sampling params 控制。

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L64-L70
# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)
```

---

## Q6：分块 prefill 与 `PrefillAdder.new_chunked_req` 的关系？

**Explain：** 当单次 extend 超过 `rem_chunk_tokens` 时，`add_one_req` 只提交第一块，设置 `new_chunked_req`，Scheduler 将其存到 `self.chunked_req`。下一轮 prefill **跳过** MinFreeSlots 和空队列检查，直接 `add_chunked_req` 续传。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L797-L835（add_chunked_req 节选）
    def add_chunked_req(self, req: Req):
        if self.dllm_config is not None:
            _rem_tokens = self._get_dllm_remain_tokens()
        else:
            _rem_tokens = min(self.rem_chunk_tokens, int(self.rem_total_tokens))
            if self.is_hybrid_swa:
                # alloc_extend needs extend_num_tokens + page_size per request,
                # so reserve one page here to avoid OOM
                _rem_tokens = min(
                    _rem_tokens, int(self.rem_swa_tokens) - self.page_size
                )
            # The chunked_req must be added to the list; otherwise, it will cause a memory leak.
            # Therefore, in certain cases where _rem_tokens <= 0, it should be replaced with rem_chunk_tokens.
            if _rem_tokens <= 0:
                if self.is_hybrid_swa:
                    return req
                _rem_tokens = self.rem_chunk_tokens

        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
        req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)
        self.can_run_list.append(req)
        self._update_prefill_budget(
            0,
            req.extend_range.length,
            (
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
                if not truncated
                else 0
            ),
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
        )

        # Return if chunked prefill not finished
        return req if truncated else None
```

**Comment：** 分块期间 `max_new_tokens` 预算为 0，最后一块才计入 decode 预留——避免重复扣减。

---

## Q7：与 vLLM 调度策略的对比

| 维度 | SGLang（本模块） | vLLM（典型） |
|------|----------------|--------------|
| 排序策略 | 可插拔 LPM/DFS/LOF/routing-key | 主要是 FCFS + priority |
| 前缀感知 | 原生 Radix + 批内前缀优化 | PagedAttention prefix（实现不同） |
| 预算模型 | PrefillAdder 多层 rem_* + 抢占 | Scheduler 内嵌 budget |
| 延迟 prefill | PrefillDelayer + MinFreeSlots | 无直接等价（continuous batching 默认立即 merge） |
| 分块 prefill | `rem_chunk_tokens` + `chunked_req` 状态机 | Chunked prefill 类似概念 |

SGLang 的差异化在于 **Radix 树深度集成到排序** 以及 **overlap + DP 下的 PrefillDelayer**。

---

## Q8：启用 PrefillDelayer 的前置条件

**Explain：** 必须 **overlap scheduling 开启**（`disable_overlap_schedule=False`）。否则构造时 assert 失败。decode-only 引擎即使传 flag 也会被忽略。

```python
# 来源：python/sglang/srt/managers/prefill_delayer.py L110-L112
        assert (
            not server_args.disable_overlap_schedule
        ), "To use PrefillDelayer, disable_overlap_schedule must be False."
```

此外 token 低水位 `prefill_delayer_token_usage_low_watermark` 可在 GPU 闲置时 bypass delay，避免 throughput 过度下降。

---

## 验证建议（零基础可试）

1. **操作：** 准备固定 512 token system prompt + 短 user message，启动 A：`--schedule-policy fcfs`；B：`--schedule-policy lpm`。各发 50 条并发请求，对比 `/metrics` 或日志中的 TTFT / `cache_hit_rate`。 
 **预期现象：** LPM 在共享 system prompt 场景下 TTFT 更低、cache hit 更高；FCFS 更公平但 prefix 复用弱。 
 **对应文档节：** [[08-SchedulePolicy-01-核心概念|01-核心概念 § 用户故事]]、Q1 LPM vs FCFS

2. **操作：** 设 `SGLANG_RADIX_FORCE_MISS=1` 重启服务，用相同 workload 再测 TTFT。 
 **预期现象：** 强制 miss 后 TTFT 明显上升（每条都全长 prefill），验证 prefix 排序收益来自 Radix 而非偶然 batching。 
 **对应文档节：** §3 `match_prefix_for_req`、Q4 批内 deprioritize

3. **操作：** 显式 `--schedule-policy lpm` 并加 `--disable-radix-cache`（或 ChunkCache 模式），观察启动日志与调度行为。 
 **预期现象：** cache-aware 策略静默降级为 FCFS，LPM 排序 key 失效；与 Q1 表「disable radix → fcfs」一致。 
 **对应文档节：** §2 调度策略枚举、Q1 场景表
