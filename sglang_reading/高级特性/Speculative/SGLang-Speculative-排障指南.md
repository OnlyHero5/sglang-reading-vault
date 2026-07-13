---
title: "Speculative · 排障指南"
type: troubleshooting
framework: sglang
topic: "Speculative"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# Speculative · 排障指南

这篇按症状排障，不按算法名罗列。投机解码的问题通常不是“某个函数错了”，而是控制账、阶段账、KV 账、验收账中的一处语义错位。

## 快速定位表

| 症状 | 先查 | 源码入口 | 验证方法 |
|------|------|----------|----------|
| 打开投机后吞吐下降 | accept rate 和 draft/verify/commit 成本 | 算法 worker、接受长度统计；EAGLE 再看 adaptive | 对比关闭投机与较小 steps/topk/block 后 tokens/s |
| 启动时报 unknown algorithm | 控制账 | `SpeculativeAlgorithm.from_string` | 检查算法名是否内置或已注册 |
| 自定义插件 overlap 报错 | 控制账 | `CustomSpecAlgo.create_worker` | 检查 `supports_overlap` 与 `disable_overlap_schedule` |
| EAGLE/NGRAM 随机验收跨 TP 不一致 | 验收账 | `eagle_sample` stochastic branch | 先确认不是 greedy/HIP/NPU 分支，再看三个结果是否 rank 0 broadcast |
| topk 大于 1 后输出错位 | KV 账 | `_finalize_accept_tree_path`、KV mover | 检查 `accept_index.shape[1]` 与 compact path |
| NGRAM 命中异常或串请求 | corpus 状态 | `NGRAMWorker` departed rid 清理 | 检查请求离批后 match state 是否 erase |
| plan stream 下偶发竞态 | 阶段账和 stream 边界 | `prepare_for_draft_extend` | 确认 caller 在进 plan stream 前完成 dtype cast |
| adaptive 不切步或频繁抖动 | 验收反馈 | `AdaptiveController` | 检查候选 state 是否注册、accepted drafts 是否回传 |

## Q1：EAGLE 和 NGRAM 怎么选？

先看 workload，而不是先看算法名。EAGLE 用额外 draft model 换更强候选；NGRAM 用历史/corpus 匹配换零模型成本。二者都需要 target verify，但 draft 侧成本完全不同。

| 维度 | EAGLE family | NGRAM |
|------|--------------|-------|
| draft 来源 | draft model / hidden states | corpus match |
| draft KV | 有 | 无 |
| verify 输入 | `EagleVerifyInput` | `NgramVerifyInput` |
| 适合场景 | 有匹配 draft checkpoint、模型分布稳定 | 模板文本、代码补全、重复上下文 |
| 首要风险 | draft 与 target 分布不匹配 | corpus 状态污染、重复度不足 |

源码上最小分界是 `has_draft_kv`：

```python
# 来源：python/sglang/srt/speculative/spec_info.py L121-L130
    def has_draft_kv(self) -> bool:
        """Whether the draft phase writes KV chains. NGRAM does not (its tree
        lives only in the verify mask), so per-decode KV sizing needs no
        per-topk page rounding; see get_alloc_len_per_decode."""
        return not self.is_ngram()

    def carries_draft_hidden_states(self) -> bool:
        """Whether the disagg prefill->decode transfer carries draft hidden
        states (EAGLE-family only; STANDALONE's vanilla draft ignores them)."""
        return self.is_eagle()
```

排障抓手：如果你选 NGRAM，就不要沿 draft model 权重和 draft hidden transfer 排查；如果你选 EAGLE，就要同时看 draft checkpoint 质量、hidden states、draft KV 和 target verify。

## Q2：自定义算法为什么启动期就失败？

插件注册允许外部算法接入，但 overlap 调度能力必须和当前 server args 一致。源码在创建 worker 前做 fail fast。

```python
# 来源：python/sglang/srt/speculative/spec_registry.py L92-L111
    def handle_server_args(self, server_args: ServerArgs) -> None:
        pass

    def create_worker(self, server_args: ServerArgs) -> Type:
        if not server_args.disable_overlap_schedule and not self.supports_overlap:
            raise ValueError(
                f"Speculative algorithm {self.name} does not support overlap scheduling."
            )
        if not self.supports_overlap:
            # Reached only when overlap is disabled, so the algorithm really
            # does run synchronously on the V2 schema below.
            logger.warning(
                "Speculative algorithm %s is registered with "
                "supports_overlap=False, which is deprecated: the spec V1 "
                "worker path has been removed, and the algorithm now runs on "
                "the V2 scheduler schema with overlap disabled (synchronous). "
                "Migrate the plugin worker to support overlap scheduling.",
                self.name,
            )
        return self.factory(server_args)
```

判断：

- overlap 开启且插件声明不支持，启动直接失败。
- overlap 关闭时仍会跑 V2 schema 的同步路径，但会 warning。
- 这不是性能提示，而是调度协议边界。

## Q3：为什么 plan stream 中不能临时 cast `predict`？

`prepare_for_draft_extend` 可能运行在 plan stream 下。这里若对 `predict` 做 dtype cast，会引入跨 stream dependency，进而破坏 MTP acceptance 的时序。

```python
# 来源：python/sglang/srt/speculative/base_spec_worker.py L114-L125
        batch.spec_info = draft_extend_input
        # Do NOT cast predict dtype here. The caller (e.g., _draft_extend_for_decode)
        # may run this under a plan stream; casting inside the plan stream creates a
        # cross-stream dependency that can lead to data races and break MTP acceptance.
        # The caller should cast to int64 before entering the plan stream context.
        batch.input_ids = predict
        maybe_detect_oob(
            batch.input_ids,
            0,
            batch.model_config.vocab_size,
            "v2 prepare_for_draft_extend input_ids",
        )
```

验证方法：在 caller 处检查 `predict.dtype`，确认进入 plan stream 前已经是目标 dtype；不要在这个函数里“顺手修正”。

## Q4：哪些 verify 分支会同步 TP 采样结果？

在 CUDA/MUSA 等平台的非全 greedy stochastic 分支中，target logits 经过 softmax、top-k、top-p 后，TP rank 间可能因浮点微差采到不同 token。spec decode 的后果比普通 sampling 更重，因为 `accept_index` 还会驱动 KV 写回。

```python
# 来源：python/sglang/srt/speculative/eagle_utils.py L596-L606
        # Sync sampling results across TP ranks: different GPUs may
        # produce slightly different target_probs due to floating-point
        # non-determinism in softmax/top_k/top_p, causing different
        # sampled tokens. Broadcast from rank 0 to ensure consistency.
        tp_group = (
            get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()
        )
        if tp_group.world_size > 1:
            tp_group.broadcast(predict, src=0)
            tp_group.broadcast(accept_index, src=0)
            tp_group.broadcast(num_correct_drafts, src=0)
```

排障抓手：这张源码卡位于 stochastic `else` 内。greedy 分支不 broadcast；HIP/NPU 即使请求不是 greedy，也由平台条件强制 argmax，并跳过该 broadcast。只有确认实际进入 stochastic 分支后，才要求 `predict`、`accept_index`、`num_correct_drafts` 三者同源。

## Q5：topk 大于 1 为什么容易 KV 写回错位？

topk 树的 verify block 比最终接受链更宽。accepted path 要搬回每个请求的连续前部，KV mover 的宽度必须按 `accept_index.shape[1]` 算。

```python
# 来源：python/sglang/srt/speculative/spec_utils.py L541-L578
    bs = len(batch.seq_lens)
    device = batch.seq_lens.device
    # accept_index element count, NOT bs * num_draft_tokens: for topk > 1 the
    # tree exceeds the accepted chain, over-reading accept_index (illegal memory).
    size = bs * accept_index.shape[1]

    # fill_accept_out_cache_loc reads out_cache_loc[accept_index]; -1 sentinel ok.
    maybe_detect_oob(
        accept_index,
        -1,
        batch.out_cache_loc.size(0),
        "spec v2 move_accept_tokens accept_index",
    )

    tgt_cache_loc = torch.zeros(
        size,
        dtype=torch.int64,
        device=device,
    )
    accept_out_cache_loc = torch.zeros(size, dtype=torch.int64, device=device)
    assign_extend_cache_locs[(bs,)](
        batch.req_pool_indices,
        batch.req_to_token_pool.req_to_token,
        batch.seq_lens,
        batch.seq_lens + num_correct_drafts + 1,
        tgt_cache_loc,
        batch.req_to_token_pool.req_to_token.shape[1],
        next_power_of_2(bs),
    )
    fill_accept_out_cache_loc[(size,)](
        accept_index,
        batch.out_cache_loc,
        accept_out_cache_loc,
        next_power_of_2(size),
    )
    token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
        tgt_cache_loc, accept_out_cache_loc
    )
```

验证方法：在 topk > 1 的请求上断点检查 `accept_index.shape`、`batch.out_cache_loc.shape`、`accept_lens - 1`。如果有人按 `num_draft_tokens` 推 mover 宽度，就是错误方向。

## Q6：NGRAM 为什么会串请求或命中异常？

NGRAM 的 match state 跟请求 rid 绑定。overlap 下请求是否 finished 的状态可能滞后一轮，所以源码用当前 batch rid 集合与上一轮 rid 集合求差，清理离开 decode batch 的请求。

```python
# 来源：python/sglang/srt/speculative/ngram_worker.py L472-L481
            self._update_ngram_corpus(batch)
            # Erase match state of requests that left the decode batch.
            # req.finished() is unusable here: under overlap it flips at result
            # processing, one iteration after the request left the batch.
            # The last batch's entries persist while idle (bounded, small).
            cur_rids = {req.rid for req in batch.reqs}
            departed_rids = self._prev_decode_rids - cur_rids
            if departed_rids:
                self.ngram_corpus.erase_match_state(list(departed_rids))
            self._prev_decode_rids = cur_rids
```

排障抓手：如果 NGRAM 在多请求交错下表现异常，先看 rid 清理，不要先怀疑 tokenizer 或 target model。

## Q7：DFLASH 与普通 EAGLE 的边界在哪里？

DFLASH 不只是“能在 draft 阶段 target verify”。它是固定 block 的独立验收协议：target verify 用标准 causal mask；非 greedy 且专用 kernel 可用时走 DFLASH sampling verify，否则走 argmax block accept；它自己构造 `commit_lens/out_tokens`，不调用 `eagle_sample` 或通用 target KV mover。

```python
# 来源：python/sglang/srt/speculative/spec_info.py L118-L119
    def supports_target_verify_for_draft(self) -> bool:
        return self.is_dflash()
```

因此遇到以下现象时应查 DFLASH worker，而不是 EAGLE 树路径：`return_logprob=True` 启动/运行报错；非 greedy 配置却出现 greedy 回退警告；`eagle_sample`、`accept_index` 或 KV mover 断点不命中。标准 causal mask 也是预期行为，不是 tree mask 丢失。

```python
# 来源：python/sglang/srt/speculative/dflash_worker_v2.py L1215-L1229
    def _validate_phase1_sampling_support(self, batch: ScheduleBatch) -> None:
        sampling_info = batch.sampling_info
        if sampling_info is None or sampling_info.is_all_greedy:
            return

        if (
            not is_dflash_sampling_verify_available()
            and not self._warned_sampling_fallback
            and self.tp_rank == 0
        ):
            logger.warning(
                "DFLASH non-greedy verification is unavailable on this build/device; "
                "falling back to greedy argmax verification."
            )
            self._warned_sampling_fallback = True
```

```python
# 来源：python/sglang/srt/speculative/dflash_worker_v2.py L1263-L1272
    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DFLASH speculative decoding does not support return_logprob yet."
            )
        self._validate_phase1_sampling_support(batch)
```

## Q8：adaptive spec 为什么没切换，或频繁切换？

EAGLE adaptive controller 要先为候选 steps 构建 runtime state，然后在 batch size 或 verify 结果触发时 `_activate`。verify 反馈发生在 `accept_lens` 已到 CPU 的结果处理阶段，不在 GPU worker 热路径同步复制。如果 state 未注册，切换会失败；如果 accept 反馈抖动，可能频繁切换。NGRAM 当前把 controller 固定为 `None`，不要把“不切步”当作 NGRAM adaptive 故障。

```python
# 来源：python/sglang/srt/speculative/adaptive_runtime_state.py L118-L134
    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> None:
        """Feed verify results; switch runtime state if EMA warrants it."""
        new_step = self.params.on_verify_complete(
            num_correct_drafts_per_req, batch_size
        )
        if new_step is not None:
            self._activate(new_step)

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)
```

验证方法：

- 检查 `candidate_steps` 是否包含目标 step。
- 检查 verify 完成后是否把 drafts-only accepted counts 传给 controller。
- 对比固定 steps 与 adaptive steps 的吞吐和延迟抖动。

## 复盘：排障顺序

1. 先判断算法分支：EAGLE family、NGRAM、DFLASH、Frozen-KV MTP、STANDALONE。
2. 再判断阶段：draft、draft extend、verify、accept writeback。
3. 接着看验收结果：EAGLE/NGRAM 检查 `predict/accept_lens/accept_index`；DFLASH 检查 `out_tokens/commit_lens/bonus`。
4. 最后看算法专用提交：EAGLE 树/NGRAM 是否需要移动 KV，DFLASH block 是否正确 commit，EAGLE adaptive 是否切换。

不要把 accept rate 当成唯一指标。投机净收益还取决于 draft 成本、verify 并行度、topk 树宽度、KV 写回成本和 workload 可预测性。
