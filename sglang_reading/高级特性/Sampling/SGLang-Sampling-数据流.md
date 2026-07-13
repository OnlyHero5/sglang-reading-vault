---
title: "Sampling · 数据流"
type: dataflow
framework: sglang
topic: "Sampling"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/dataflow
  - source-reading
updated: 2026-07-12
---
# Sampling · 数据流

## 先回答为什么读

源码走读解释函数顺序，本篇解释对象形状。Sampling 的排障重点通常不是“有没有这个参数”，而是参数在不同阶段变成了什么：

```text
SamplingParams
  -> Req.grammar / grammar_queue
  -> SamplingBatchInfo tensors
  -> penalized / masked / biased / custom-processed logits
  -> batch_next_token_ids
  -> Req output / grammar state
  -> next-step penalty state
```

这是一张生命周期图，不是无条件的调用链：普通 decode、overlap、speculative verify、Ascend 和 RL on-policy 会在中间分叉。

如果你能沿这条数据流说出每一步对象长什么样，就能判断问题发生在参数、队列、mask、采样还是状态推进。

## 1. 单请求参数：CPU 侧语义

`SamplingParams` 是单个请求的采样意图。它包含 API 字段和内部字段；`__post_init__` 先处理特殊值，`verify` 校验，`normalize` 再收束 stop/tokenizer 相关字段。

| 字段类别 | 例子 | 后续消费者 |
|----------|------|------------|
| 长度和停止 | `max_new_tokens`、`stop_strs`、`stop_regex_strs` | result processor / stop checker |
| 分布控制 | `temperature`、`top_p`、`top_k`、`min_p` | `SamplingBatchInfo` / `Sampler` |
| 惩罚 | `frequency_penalty`、`presence_penalty`、`repetition_penalty` | penalty orchestrator |
| 约束 | `json_schema`、`regex`、`ebnf`、`structural_tag` | `GrammarManager` |
| 可观测 | `logit_bias`、`sampling_seed` | logits preprocess / deterministic sampling |

`temperature=0` 的对象变化尤其关键：构造后会成为 `temperature=1.0、top_k=1`，所以 batch tensor 中不会保留零温度。

无 tokenizer 时，字符串 stop、regex stop、`min_new_tokens` 会直接被拒绝，因为它们依赖 decode 或 eos token id。

```python
# 来源：python/sglang/srt/sampling/sampling_params.py L45-L72
def raise_if_tokenizer_required(
    tokenizer, stop_strs, stop_regex_strs, min_new_tokens=0
):
    """Raise ValueError if tokenizer-dependent features are used without a tokenizer.

    String-based stop conditions (stop_strs, stop_regex_strs) require tokenizer.decode()
    to convert output token IDs to text for matching. min_new_tokens requires the
    tokenizer's eos_token_id to penalize. When skip_tokenizer_init=True, these cannot
    be used.
    """
    if tokenizer is not None:
        return

    if stop_strs:
        raise ValueError(
            f"stop={stop_strs!r} is unavailable when skip_tokenizer_init=True "
            "(requires tokenizer to decode tokens to text for matching)."
        )
    if stop_regex_strs:
        raise ValueError(
            f"stop_regex={stop_regex_strs!r} is unavailable when skip_tokenizer_init=True "
            "(requires tokenizer to decode tokens to text for matching)."
        )
    if min_new_tokens > 0:
        raise ValueError(
            f"min_new_tokens={min_new_tokens} is unavailable when skip_tokenizer_init=True "
            "(requires tokenizer for eos_token_id)."
        )
```

## 2. Grammar queue：请求可能暂不进 batch

带 grammar 的请求会得到 `req.grammar`。cache miss 时它是一个 Future，并且请求被留在 `grammar_queue`；cache hit 时它是可直接用于 mask 的 grammar object。

```text
Req
  sampling_params.json_schema != None
  grammar = Future | GrammarObject | InvalidGrammarObject
  grammar_key = ("json", schema)
  grammar_wait_ct = number
```

队列轮询时，未 ready 的请求会累加 `grammar_wait_ct`，超时后进入 failed 集合。

```python
# 来源：python/sglang/srt/constrained/grammar_manager.py L175-L199
        # Check failed requests
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
```

DP 下 ready 取交集，failed 取并集：所有 rank 都 ready 才准入，任一 rank 失败就失败。

## 3. Batch tensor：每行参数对齐 batch 行

进入 batch 后，采样参数变成设备上的张量：

| 字段 | 形状 | 含义 |
|------|------|------|
| `temperatures` | `[bs, 1]` | 对 `[bs, vocab]` logits 广播除法 |
| `top_ps` | `[bs]` | 每行 top-p 阈值 |
| `top_ks` | `[bs]` | 每行 top-k 阈值 |
| `min_ps` | `[bs]` | 每行 min-p 阈值 |
| `sampling_seed` | `[bs]` 或 `None` | 仅 deterministic inference 开启时存在；未显式设置的行填 `42` |
| `logit_bias` | `[bs, vocab]` | 可选逐 token bias |
| `vocab_mask` | backend 相关 | grammar 合法 token mask |

`SamplingBatchInfo` 还会记录 batch 级开关：是否全 greedy，是否需要 top-p、top-k、min-p。这些标志在构造和 merge 时维护，但当前 `filter_batch` 不会重算；它们可能保守地让剩余请求继续走较慢路径。

```python
# 来源：python/sglang/srt/sampling/sampling_batch_info.py L184-L203
        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            sampling_seed=sampling_seed,
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
            need_top_p_sampling=any(r.sampling_params.top_p != 1.0 for r in reqs),
            need_top_k_sampling=any(r.sampling_params.top_k != TOP_K_ALL for r in reqs),
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
            vocab_size=vocab_size,
            penalizer_orchestrator=penalizer_orchestrator,
            has_custom_logit_processor=has_custom_logit_processor,
            custom_params=custom_params,
            custom_logit_processor=merged_custom_logit_processor,
            device=device,
            logit_bias=logit_bias,
        )
        ret.adjusted_from_schedule_batch(batch, vocab_size)
        return ret
```

## 4. Logits：从模型分数变成可采样分布

模型输出的 `next_token_logits` 会被原地改写。`apply_logits_bias` 的名字容易误导，它实际上按顺序处理 overlap additive penalty、overlap scaling penalty、non-overlap penalty、grammar mask 和 logit bias。

```python
# 来源：python/sglang/srt/sampling/sampling_batch_info.py L266-L283
    def apply_logits_bias(self, logits: torch.Tensor):
        if self.acc_additive_penalties is not None:
            # Used in the overlap mode
            logits.add_(self.acc_additive_penalties)

        if self.acc_scaling_penalties is not None:
            # Used in the overlap mode
            apply_scaling_penalties(logits, self.acc_scaling_penalties)

        if self.penalizer_orchestrator and self.penalizer_orchestrator.is_required:
            # Used in the non-overlap mode
            self.penalizer_orchestrator.apply(logits)

        if self.vocab_mask is not None:
            self.apply_mask_func(logits=logits, vocab_mask=self.vocab_mask)

        if self.logit_bias is not None:
            logits.add_(self.logit_bias)
```

`xgrammar` 用 bitmask kernel 原地改 logits。

```python
# 来源：python/sglang/srt/constrained/xgrammar_backend.py L238-L245
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if logits.device.type in {"cuda", "npu", "xpu", "musa"}:
            if _is_hip:
                apply_token_bitmask_inplace_cuda(logits, vocab_mask)
            else:
                apply_token_bitmask_inplace_triton(logits, vocab_mask)
        else:
            raise RuntimeError(f"Unsupported device: {logits.device.type}")
```

`outlines` 则直接把非法位置填成负无穷。

```python
# 来源：python/sglang/srt/constrained/outlines_backend.py L73-L75
    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor):
        logits.masked_fill_(vocab_mask, float("-inf"))
```

这两个 backend 的 mask 形态不同；如果后续没有扩展再次改写 logits，效果相同：非法 token 在采样中不可选。可选的 custom logit processor 在 `Sampler._preprocess_logits` 中更晚执行，因此它有能力覆盖这个结果，属于必须单独审计的信任边界。

## 5. Penalty：跨 step 的历史状态

Penalty 不是静态参数。每一步生成后，orchestrator 会记住输出 token；下一步再把 frequency、presence、repetition、min-new-token 等影响施加到 logits。

```python
# 来源：python/sglang/srt/sampling/penaltylib/orchestrator.py L45-L87
    def cumulate_output_tokens(self, output_ids: torch.Tensor):
        """
        Feed the output tokens to the penalizers.

        Args:
            output_ids (torch.Tensor): The output tokens.
        """
        for penalizer in self.penalizers.values():
            penalizer.cumulate_output_tokens(output_ids=output_ids)

    def apply(self, logits: torch.Tensor, repeat: Optional[int] = None):
        """
        Apply all penalizers to the logits in-place.

        Args:
            logits: The logits tensor to apply penalties to.
            repeat: If set (speculative decoding), per-request penalties are
                expanded via repeat_interleave to match the draft token layout.
                Additive penalties are captured into a zeros tensor, expanded,
                then added; scaling penalties are accumulated, expanded, then
                applied directly.
        """
        if repeat is None:
            for penalizer in self.penalizers.values():
                penalizer.apply(logits)
        else:
            # Additive: capture into zeros, expand, add
            bs = logits.shape[0] // repeat
            additive = torch.zeros(
                (bs, logits.shape[1]), dtype=torch.float32, device=logits.device
            )
            self.accumulate_additive_penalties(additive)
            logits.add_(torch.repeat_interleave(additive, repeat, dim=0))
            # Scaling: accumulate, expand, apply
            accumulated = self.accumulate_scaling_penalties()
            if accumulated is not None:
                from sglang.srt.sampling.penaltylib.repetition_penalty import (
                    apply_scaling_penalties,
                )

                expanded = torch.repeat_interleave(accumulated, repeat, dim=0)
                apply_scaling_penalties(logits, expanded)
```

所以重复惩罚是否生效，要看两个时刻：上一步有没有 cumulate token，本步有没有 apply logits。

## 6. 输出：`batch_next_token_ids`

Sampler 输出的是 `[bs]` 的 token id。约束解码下，SGLang 会同步 token id，避免 TP rank 因采样差异进入不同 grammar 状态。同步不是“rank 0 广播”，而是对 token id 做 `MIN all_reduce`：所有 rank 最终采用各 rank 局部结果中的最小 id。

```python
# 来源：python/sglang/srt/layers/sampler.py L379-L395
    def _sync_token_ids_across_tp(
        self, batch_next_token_ids: torch.Tensor, sampling_info: SamplingBatchInfo
    ):
        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )
```

这是约束解码下 TP hang 的关键边界：token id 一旦不同，后续 grammar FSM 也会不同。它保证 rank 一致，但不证明某个特定 rank 的局部采样结果被保留。

## 7. 状态推进：提交 token、grammar 前进、penalty 延后一拍

结果处理器先把接受的 token 提交到 `Req.output_ids`，再推进 grammar FSM。普通 decode 每步一个 token；speculative verify 可能一次接受一段，并在 grammar 结束时截断 overdrafted suffix，避免非法后缀进入 KV 或输出。

```python
# 来源：python/sglang/srt/managers/scheduler_components/batch_result_processor.py L586-L615
    def _accept_grammar_tokens(
        self, req: Req, tokens: Union[int, List[int]]
    ) -> List[int]:
        """Advance the grammar over the accepted token(s), stopping at the token
        that terminates it.

        ``tokens`` is a single sampled token (normal decode) or the whole
        verified run (spec decode). Returns the retained prefix; for spec the
        suffix past grammar completion is dropped so it is never committed to KV
        nor emitted. Advances the grammar FSM only -- ``grammar.finished`` is
        synced by the caller once the finish state is updated.
        """
        if isinstance(tokens, int):
            tokens = [tokens]
        retained = []
        try:
            for token_id in tokens:
                req.grammar.accept_token(token_id)
                retained.append(token_id)
                if req.grammar.is_terminated():
                    break
        except ValueError as e:
            # accept_token raises ValueError if the token is not in the grammar
            # (misconfigured grammar or invalid token); abort the request.
            logger.error(
                f"Grammar accept_token failed for req {req.rid} with token "
                f"{tokens}: {e}"
            )
            self.abort_request(AbortReq(rid=req.rid))
        return retained
```

Penalty 的“推进”不在这个函数里直接完成。下一轮 `ScheduleBatch.prepare_for_decode()` 会从已经更新的 `Req.output_ids` 取最新 token，调用 orchestrator 的 `cumulate_output_tokens()`；然后下一次 logits 预处理才应用累计状态。因而 result processor 是历史的生产者，ScheduleBatch 是 penalty 状态的实际搬运者。

## 数据流总表

| 阶段 | 输入 | 输出 | 排障信号 |
|------|------|------|----------|
| 参数规范化 | API sampling fields | normalized `SamplingParams` | alias 是否清掉、tokenizer 依赖是否报错 |
| Grammar 准入 | `Req` + constraint field | queue 或 waiting queue | `grammar_key`、`grammar_wait_ct` |
| Batch 构造 | `ScheduleBatch.reqs` | `SamplingBatchInfo` | batch 开关和 tensor 形状 |
| Logits 预处理 | `[bs, vocab]` logits | penalty/mask/bias/custom 改写后的 logits | 顺序与扩展点是否符合预期 |
| Sampler | logits/probs | `batch_next_token_ids` | greedy vs backend 分支 |
| 结果处理 | token ids | updated `Req.output_ids` / grammar | token 提交、grammar accept |
| 下一轮 decode 准备 | 最新 `Req.output_ids` | updated penalizer state | `cumulate_output_tokens` 是否执行 |

## 运行验证

这篇可以先不启动模型，用源码检索验证三条主线是否仍然成立：参数构造与规范化在 `SamplingParams`，batch 侧状态在 `SamplingBatchInfo`，token/grammar 提交与下一步 penalty 累积分别落在 `BatchResultProcessor` 和 `ScheduleBatch`。

```powershell
rg -n 'class SamplingParams|def __post_init__|def normalize|class GrammarManager|class SamplingBatchInfo|def update_regex_vocab_mask|def apply_logits_bias|def apply_custom_logit_processor|BatchedPenalizerOrchestrator|def cumulate_penalty_output_tokens|def _sync_token_ids_across_tp|def _accept_grammar_tokens' sglang/python/sglang/srt/sampling/sampling_params.py sglang/python/sglang/srt/constrained/grammar_manager.py sglang/python/sglang/srt/sampling/sampling_batch_info.py sglang/python/sglang/srt/sampling/penaltylib/orchestrator.py sglang/python/sglang/srt/layers/sampler.py sglang/python/sglang/srt/managers/schedule_batch.py sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py
```

读输出时重点看三件事：

- `__post_init__()` 与 `normalize()` 仍然分别负责特殊值改写，以及 API alias/stop/tokenizer 依赖收束。
- `SamplingBatchInfo` 仍然同时承载 grammar mask、penalty、logit bias、custom processor 元数据和可选 seed tensor。
- `_sync_token_ids_across_tp()` 与 `_accept_grammar_tokens()` 仍然分别守住 TP 一致性和 grammar FSM 推进。
