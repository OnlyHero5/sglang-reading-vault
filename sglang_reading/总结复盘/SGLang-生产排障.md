---
title: "SGLang 生产排障"
type: troubleshooting
framework: sglang
topic: "总结复盘"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# SGLang 生产排障

## 你为什么要读

本页是一页式生产排障索引，用来从线上症状反查指标、CLI、日志关键词和专题文档。读完后，你应该能把 OOM、TTFT、decode 吞吐、CUDA Graph、PD、LoRA、量化和可观测性问题快速归到第一检查项。

> 常见症状 → 可能原因 → 检查项 → 相关文档 
> 面向生产运维 / 推理工程师的一页式手册

---

## 1. 显存 OOM / KV 不足

**症状：** 请求 abort、日志出现 `retract_decode` 或 `alloc` 返回 None；Prometheus `sglang:kv_pool_usage` 接近 1.0。

**优先检查**

- 指标：`sglang:kv_pool_usage`、`sglang:num_retracted_reqs`
- CLI：`--mem-fraction-static`、`--max-running-requests`、`--max-total-tokens`
- 日志关键字：`retract_decode`、`check_decode_mem`、`alloc` 返回 None

**常见根因**

- `max_running_requests × avg_seq_len` 超出 KV pool 预算
- 长 prompt 突发 prefill 占满 slot，decode 无空间
- HiCache 主机内存不足（启动阶段即失败）
- Paged allocator `page_size` 与序列长度不对齐导致碎片

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L3040-L3056
        # Check if decode out of memory
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            old_available_tokens = self.token_to_kv_pool_allocator.available_size()
            old_ratio = self.new_token_ratio_tracker.current
            mamba_allocator = getattr(
                self.tree_cache.req_to_token_pool, "mamba_allocator", None
            )
            old_mamba_available = (
                mamba_allocator.available_size()
                if mamba_allocator is not None
                else None
            )
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
                self.server_args
            )
```

**延伸阅读**

- [[SGLang-KV-Cache-排障指南]]
- [[SGLang-Scheduler-排障指南]]（Q3 Retract）
- [[SGLang-框架对比与设计决策]]（追问 9）

---

## 2. TTFT 过高 / 首 token 慢

**症状：** P99 TTFT 超标；首条 SSE chunk 延迟数秒；prefill batch 排队明显。

**优先检查**

- 指标：`sglang:time_to_first_token_seconds`、`sglang:cache_hit_rate`、`num_matched_prefix_tokens`
- CLI：`--max-prefill-tokens`、chunked prefill、`--disable-overlap-schedule`
- 日志关键字：`AddReqResult.NO_TOKEN`、`PrefillDelayer`、`is_disable_overlap_for_batch`

**常见根因**

- 长 prompt 全量 prefill，未启用 chunked prefill
- RadixCache 未命中，每请求重算完整 system prompt
- PrefillDelayer / MinFreeSlots 延迟 prefill 保 decode 槽位
- 连续两个 prefill batch 触发 overlap disable，牺牲吞吐换首包
- PD 模式下 prefill 网络 + KV 传输叠加

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/schedule_policy.py L91-L131
def match_prefix_for_req(
    tree_cache: BasePrefixCache,
    req: Req,
    token_ids: Optional[array[int]] = None,
    *,
    cow_mamba: bool = False,
    include_req: bool = False,
):
    if token_ids is None:
        token_ids = req.origin_input_ids + req.output_ids

    match_result = tree_cache.match_prefix(
        MatchPrefixParams(
            key=RadixKey(token_ids=token_ids, extra_key=req.extra_key),
            cow_mamba=cow_mamba,
            req=req if include_req else None,
        )
    )
    if envs.SGLANG_RADIX_FORCE_MISS.get():
        match_result = zero_match_result(tree_cache, match_result)
    (
        req.prefix_indices,
        req.last_node,
        req.last_host_node,
        req.best_match_node,
        req.host_hit_length,
        req.swa_host_hit_length,
        req.mamba_host_hit_length,
    ) = (
        match_result.device_indices,
        match_result.last_device_node,
        match_result.last_host_node,
        match_result.best_match_node,
        match_result.host_hit_length,
        match_result.swa_host_hit_length,
        match_result.mamba_host_hit_length,
    )
    max_len = req._compute_max_prefix_len(len(token_ids))
    req.num_matched_prefix_tokens = min(
        len(req.prefix_indices) + req.host_hit_length, max_len
    )
```

**延伸阅读**

- [[SGLang-SchedulePolicy-排障指南]]
- [[SGLang-RadixAttention-排障指南]]
- [[SGLang-用户场景]]

---

## 3. TPOT 高 / decode 吞吐差

**症状：** 流式输出 token 间隔大；`tokens_per_second` 低于预期；decode batch 利用率低。

**优先检查**

- 指标：decode 阶段 latency histogram、`sglang:gen_throughput`、`running_batch_size`
- CLI：`--max-running-requests`、TP/DP 配置、`--disable-overlap-schedule`
- 日志关键字：`batch_is_full`、`running_bs`、spec verify 耗时

**常见根因**

- `max_running_requests` 过小，GPU batch 未满
- prefill 持续抢占 decode slot（prefill-first 策略）
- PP 模式不走 overlap loop，stage 间 bubble
- Speculative decoding accept rate 低，verify 双倍 forward 白跑
- Grammar + spec 组合触发 `need_grammar_sync`，单轮 disable overlap

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L4168-L4178
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp()
        elif scheduler.enable_overlap_mlx:
            scheduler.event_loop_overlap_mlx()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
```

**延伸阅读**

- [[SGLang-Scheduler-排障指南]]（Q4 PP/Overlap）
- [[SGLang-SchedulePolicy-核心概念]]
- [[SGLang-分布式-排障指南]]

---

## 4. RadixCache 命中率低

**症状：** `sglang:cache_hit_rate` 明显低于该 workload 的重复前缀预期；TTFT 无改善；`num_matched_prefix_tokens` 接近 0。

**优先检查**

- 指标：`sglang:cache_hit_rate` 与基线、`num_matched_prefix_tokens` 分布
- CLI / 环境：`extra_key` 一致性、`SGLANG_RADIX_FORCE_MISS`
- 日志关键字：`extra_key` mismatch、`positional_embed_overrides`、`force_miss`

**常见根因**

- system prompt 含动态字段（timestamp、request id、用户 id）
- 各租户 LoRA adapter 不同导致 `extra_key` 隔离
- `positional_embed_overrides` 非空强制禁用 prefix cache
- 权重热更新 flush cache 后尚未重建（骤降属预期）
- page_size > 1 时 partial page 未进树

**源码锚点：**

```python
## 来源：python/sglang/srt/mem_cache/radix_cache.py L155-L160
    def _check_compatible(self, other: RadixKey) -> None:
        if self.extra_key != other.extra_key:
            raise ValueError(
                f"RadixKey operations require matching extra_key, but got "
                f"{self.extra_key=} != {other.extra_key=}"
            )
```

**延伸阅读**

- [[SGLang-RadixAttention-排障指南]]
- [[SGLang-用户场景]]（故事 A 调试表）
- [[SGLang-框架对比与设计决策]]

---

## 5. PD 分离卡住 / KV 传输超时

**症状：** Prefill 完成但 decode 无首 token；请求长时间停在 transfer 状态；Gateway PD 路由 502。

**优先检查**

- 指标：`kv_transfer_*` 延迟、`sglang:disagg_*` 分段耗时
- CLI：`--disaggregation-mode`、`--disaggregation-transfer-backend`
- 日志关键字：`KVPoll.Transferring`、`bootstrap_room == 0`、`PreallocQueue`

**常见根因**

- metadata 未就绪时过早判定 Success（room 仍为 0）
- Prefill / Decode bootstrap 握手失败，KV sender 无目标 room
- 跨 AZ KV 传输 RTT 高于本地 prefill+decode 合并成本
- `pre_alloc_size` 过小，transfer 队列阻塞
- Transfer backend 与硬件栈不匹配（Mooncake / NIXL / Ascend）

**源码锚点：**

```python
## 来源：python/sglang/srt/disaggregation/utils.py L114-L118
            actual_room = metadata_buffers.bootstrap_room[
                decode_req.metadata_buffer_index, 0
            ].item()
            if actual_room == 0:
                polls[i] = int(KVPoll.Transferring)
```

**延伸阅读**

- [[SGLang-PD分离-排障指南]]
- [[SGLang-model-gateway-源码走读]]
- [[SGLang-用户场景]]（故事 C）
- [[SGLang-框架对比与设计决策]]

---

## 6. Speculative decoding 反而变慢

**症状：** 开启 EAGLE/NGRAM 后 TPOT 上升；GPU 利用率升高但吞吐下降。

**优先检查**

- 指标：`sglang:spec_accept_rate`（或日志 `accept_lens`）
- CLI：`--speculative-algorithm`、`--speculative-num-steps`、`--speculative-draft-model-path`
- 日志关键字：`does not support overlap`、`reject_sampling`、verify batch 大小

**常见根因**

- accept length/rate 相对基线下降，实测 draft/verify 成本大于节省的 target decode 成本
- draft 模型额外占显存，挤压 running batch
- 域偏移导致 draft/target 分布不对齐
- 插件算法 `supports_overlap=False` 与 overlap 冲突
- verify 阶段 dtype cast 在 plan stream 内引发竞态

**源码锚点：**

```python
## 来源：python/sglang/srt/speculative/spec_registry.py L96-L99
        if not server_args.disable_overlap_schedule and not self.supports_overlap:
            raise ValueError(
                f"Speculative algorithm {self.name} does not support overlap scheduling."
            )
```

**延伸阅读**

- [[SGLang-Speculative-排障指南]]
- [[SGLang-用户场景]]（故事 B）
- [[SGLang-框架对比与设计决策]]

---

## 7. Grammar / JSON schema 排队

**症状：** 带 `json_schema` 的请求 TTFT 尖刺；`sglang:num_grammar_queue_reqs` 持续升高；部分请求 abort "Grammar preprocessing timed out"。

**优先检查**

- 指标：`sglang:num_grammar_queue_reqs`、grammar 编译耗时
- CLI：`--grammar-backend`（xgrammar / outlines / none）
- 日志关键字：`grammar_queue`、`grammar_wait_ct`、`InvalidGrammarObject`

**常见根因**

- 启动时 `--grammar-backend none` 但请求携带约束 → 直接 abort
- 复杂 schema 首次编译慢，cache miss 排队
- grammar + spec + decode 触发 overlap sync，单轮吞吐下降
- 多 rank DP 下 grammar 编译 all_gather 同步放大延迟
- 超时阈值 `SGLANG_GRAMMAR_MAX_POLL_ITERATIONS` 过小

**源码锚点：**

```python
## 来源：python/sglang/srt/constrained/grammar_manager.py L176-L235
        for i, req in enumerate(self.grammar_queue):
            if i not in ready_req_idxs:
                self.grammar_queue[i].grammar_wait_ct += 1
                if (
                    self.grammar_queue[i].grammar_wait_ct
                    >= self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS
                ):
                    # Timeout after max poll iterations
                    # The actual waiting time is SGLANG_GRAMMAR_MAX_POLL_ITERATIONS * max(SGLANG_GRAMMAR_POLL_INTERVAL, GPU_forward_batch_latency)
                    failed_req_idxs.add(i)

        # Sync ready and failed requests across all ranks
        if self.grammar_sync_size == 1:
            synced_ready_req_idxs = ready_req_idxs
            synced_failed_req_idxs = failed_req_idxs
        else:
            all_gather_output = [None] * self.grammar_sync_size
            torch.distributed.all_gather_object(
                all_gather_output,
                (ready_req_idxs, failed_req_idxs),
                group=self.grammar_sync_group,
            )
            synced_ready_req_idxs = set.intersection(*[x[0] for x in all_gather_output])
            synced_failed_req_idxs = set.union(*[x[1] for x in all_gather_output])

        # Return ready requests
        return_reqs: List[Req] = []
        for i in synced_ready_req_idxs:
            req = self.grammar_queue[i]
            return_reqs.append(req)
            if req.finished() or req.grammar is None:  # It is aborted by AbortReq
                continue

            assert isinstance(req.grammar, futures.Future) and req.grammar_key
            try:
                req.grammar = req.grammar.result()
            except Exception as e:
                logger.error(
                    f"Grammar compilation raised an exception: {e}, "
                    f"grammar_key={req.grammar_key}"
                )
                req.grammar = InvalidGrammarObject(f"Grammar compilation failed: {e}")
            self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())
            self._apply_request_reasoning_budget(req)
            if isinstance(req.grammar, InvalidGrammarObject):
                error_msg = f"Failed to compile {req.grammar_key[0]} grammar: {req.grammar.error_message}"
                req.set_finish_with_abort(error_msg)

        # Return failed requests
        for i in synced_failed_req_idxs:
            req = self.grammar_queue[i]
            return_reqs.append(req)

            assert isinstance(req.grammar, futures.Future) and req.grammar_key
            req.grammar.cancel()
            self.grammar_backend.set_cache(
                req.grammar_key, InvalidGrammarObject("Grammar preprocessing timed out")
            )
            error_msg = f"Grammar preprocessing timed out: {req.grammar_key=}"
            req.set_finish_with_abort(error_msg)
```

**延伸阅读**

- [[SGLang-Sampling-排障指南]]
- [[SGLang-Scheduler-排障指南]]（grammar sync）
- [[SGLang-可观测性-排障指南]]

---

## 8. 权重热更新失败

**症状：** `update_weights_from_ipc` 返回失败；`sglang:num_paused_reqs` 尖刺后不恢复；热更新后输出 token 错乱。

**优先检查**

- 指标：`sglang:num_paused_reqs`、`sglang:weight_load_duration_seconds{source="ipc"}`、`sglang:cache_hit_rate`（flush 后骤降属预期）
- CLI：`--checkpoint-engine-wait-weights-before-ready`、`--load-format`
- 日志关键字：`checkpoint-engine is not installed`、`flush_cache`、`ImportError`

**常见根因**

- 未安装 `sglang[checkpoint-engine]` 依赖
- `inference_parallel_size` 与 server TP 不一致
- 热更新未 flush cache，旧 prefix KV 与新权重不一致
- `wait_weights_before_ready` 超时，HTTP 已 up 但 warmup 未完成
- broadcast / p2p update_method 配置错误

**源码锚点：**

```python
## 来源：python/sglang/srt/checkpoint_engine/update.py L122-L128
                    f"{endpoint}/update_weights_from_ipc",
                    json={
                        "zmq_handles": dict(
                            socket_paths[src : src + inference_parallel_size]
                        ),
                        "flush_cache": True,
                        "weight_version": weight_version,
```

**延伸阅读**

- [[SGLang-CheckpointEngine-排障指南]]
- [[SGLang-可观测性-排障指南]]（Q4–Q5）
- [[SGLang-框架对比与设计决策]]（追问 15）

---

## 快速索引

| 症状关键词 | 跳转章节 |
|-----------|---------|
| OOM / retract / KV 满 | §1 |
| 首 token 慢 / TTFT | §2 |
| decode 慢 / TPOT / 吞吐 | §3 |
| cache_hit_rate 低 | §4 |
| PD / transfer / bootstrap | §5 |
| spec / EAGLE / accept | §6 |
| json_schema / grammar 排队 | §7 |
| 热更新 / checkpoint | §8 |
