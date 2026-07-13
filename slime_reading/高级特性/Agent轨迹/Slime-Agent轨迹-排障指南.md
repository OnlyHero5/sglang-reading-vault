---
title: "Agent轨迹 · 排障指南"
type: troubleshooting
framework: slime
topic: "Agent轨迹"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-13
---
# Agent轨迹 · 排障指南

本篇按症状排障。先判断问题发生在 wire 翻译、SGLang token 捕获、message tree 挂载、drift 线性化，还是 session 生命周期。

## 症状速查

| 症状 | 可能原因 | 源码入口 | 验证方式 |
|------|----------|----------|----------|
| loss 和 rollout logprob 对不齐 | 用文本重新 tokenize，或 SGLang 没返回 output logprobs | `call_sglang_generate` | 检查 `output_token_logprobs` 与 `Sample.loss_mask` 长度 |
| 每轮都 fork 成新 sample | manager message 不稳定，tool call id 或 arguments 字符串破坏 dict equality | OpenAI/Anthropic translate 和 reply build | 对比下一轮 prompt replay 的 assistant dict |
| 同一 assistant 响应重复训练 | `response_trained` 没生效或绕过 manager 自己拼 sample | `_split_chain_into_builders` | 看 shared leaf 的 loss_mask 是否只有第一次为 1 |
| 短 assistant rewrite 生成废弃样本 | rewrite merge 阈值太低或 assistant child 不唯一 | `_try_merge_assistant_rewrite` | 跑 branching rewrite merge 测试 |
| finish 后第二次取不到样本 | `get_trajectory` 是一次性消费 | `get_trajectory` | 第二次调用应返回空列表 |
| 客户端断连后仍训练 | 在响应 flush 前手动 record，或绕过 `_run_turn` | `_run_turn` | 断连路径应返回 499 且不 record |
| 多轮 prefix cache 命中差 | session id 不稳定或 router policy 未设置 | `X-SMG-Routing-Key` | 检查请求头和 `--router-policy consistent_hashing` |
| runaway agent 无限请求 | 没设置 turn cap 或 harness 超时 | `_check_turn_cap` | 超限应返回 429 |
| 多个无鉴权 agent 轨迹混在一起 | 都回退到 sid=`default` | `_request_session_id` | 检查 Bearer/API key/body metadata |
| 并行工具只执行了一个 | OpenAI reply builder 只保留第一个 tool call | `_build_reply_parts` | 对比 parser `tool_uses` 与 wire `tool_calls` 数量 |
| context cap 后 `response_length` 异常 | `max_sample_tokens` 切入首轮 prompt | `_SampleBuilder.to_sample` | 比较 cap 与 `leading_prompt_len` |
| finish 失败重试后训练 leaf 变了 | `response_trained` 在 pop tree 前已原地修改 | `_split_chain_into_builders` | 比较首次失败前后 node 标记 |
| fan-out 后 reward 总影响放大 | 每个 sample 都拿完整 reward | `get_trajectory` | 核对 sibling 数量与 loss aggregation 口径 |
| 按官方示例调 `finish_session` 报缺参数 | 文档示例省略必需 `base_sample` | `BaseAdapter.finish_session` | 以当前函数签名为准 |
| Anthropic client 不提前做 context 管理 | count-tokens 端点固定返 0 | `_count_tokens` | 检查 client 是否把该值当硬预算 |

## Q1：为什么不能 decode 再 tokenize 输出？

训练目标必须和 rollout 采样 token id 一致。decode 后再 tokenize 会受 chat template、special token、空格影响，导致 loss mask 和 logprob 对不上。

```python
# 定位骨架（基于 `slime/agent/adapters/common.py` L475-L518；只展示 SGLang 回包解析）
        meta = data.get("meta_info") or {}
        output_token_logprobs = meta.get("output_token_logprobs") or []
        output_ids = [x[1] for x in output_token_logprobs]
        output_log_probs = [float(x[0]) for x in output_token_logprobs]
        finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
```

排障动作：在 custom generate 里不要用 `output["text"]` 重新 tokenizer；应传递模型真实 `output_ids`、`output_log_probs`。

## Q2：REALIGN 和 FORK 怎么判断？

`classify_token_drift` 用 held tokens 和新 turn `prompt_ids` 的 common prefix 判断 drift 位置。drift 在最近 response 内且新输出短于阈值，才 realign；其他情况 fork。

```python
# 定位骨架（基于 `slime/agent/trajectory.py` L169-L191；只展示 drift 分类入口）
def classify_token_drift(self, turn: TurnRecord) -> DriftKind:
    """Decide how this builder should absorb ``turn``'s prompt.

    The incoming turn's prompt is expected to match the tokens this builder
    already holds as an exact prefix. When token drift has occurred -- the
    prompt diverges from the held tokens -- we decide whether to REALIGN
    (heal a short divergence inside the most-recent response span) or to FORK
    (``len(turn.output_ids) >= fork_threshold``, or the divergence sits too
    early to absorb). With no drift the turn is handled the CLEAN way -- a
    plain prefix extension.
    """
    realign_at = _common_prefix_len(self.tokens, turn.prompt_ids)
```

排障动作：调低 `fork_threshold_tokens` 会产生更多 sample；调高会把更多短 rewrite 吸收到同一 sample。

参数语义要纠正：REALIGN 的阈值比较对象是“当前 turn 的整个 `output_ids` 长度”，不是 drift tail 长度。且 `_align_to_prompt` 从最近 response 起点起整段覆盖为上下文，不只遮掉 common-prefix 之后的少量差异 token。

## Q3：tool call 为什么要 canonicalize？

树匹配用 dict equality。OpenAI 的 wire `arguments` 是 JSON string，且 `tool_calls[].id` 每次可能变化；如果保留这些 wire 细节，下一轮 replay 就无法命中原节点。

```python
# 定位骨架（基于 `slime/agent/adapters/openai.py` L102-L164；只展示 canonicalization 契约）
def _translate_messages(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages -> tokenizer chat-template messages.

    Mirrors anthropic._translate_messages so a replayed assistant turn compares
    equal (dict equality) to the leaf the manager appended on the previous
    request. Two invariants must hold:

      * tool_calls[i].function.arguments is a dict (not a JSON string): the chat
        template needs a mapping, and the manager matches history by dict
        equality regardless of key order.
      * Wire-only correlation ids are dropped (tool_call_id on tool messages,
        tool_calls[i].id on echoed assistant messages). Fresh ids are minted on
```

Anthropic 用共享 `tool_call_dict` 做同一件事。

```python
# 定位骨架（基于 `slime/agent/adapters/common.py` L110-L118；省略返回行）
def tool_call_dict(name: str, arguments: dict | None) -> dict:
    """Canonical OpenAI-shape tool call stored on manager_message.

    arguments stays a dict (not a JSON string): the chat template needs a
    mapping, and the trajectory manager matches history by dict equality, so a
    sampled leaf and its replayed echo compare equal regardless of key order.
    The wire-only tool-call id is dropped for the same reason.
    """
```

## Q4：同一个 assistant 响应为何只训练一次？

当多个 leaf 共享 generated assistant 前缀时，第一次经过该 node 会把 `response_trained` 设为 True；后续 leaf 再经过它时，builder 用 `trained=False` 把它作为上下文重放。

```python
# 定位骨架（基于 `slime/agent/trajectory.py` L456-L500；只展示 cross-leaf 去重入口）
def _split_chain_into_builders(self, chain: list[MessageNode]) -> list[_SampleBuilder]:
    """Pack the chain's generated turns into per-Sample token builders.

    Turns flow into the current builder until one can't extend it as an
    exact prefix (re-tokenization drift past what we can drop); that turn
    opens a new builder -- a fork. A generated turn shared by sibling leaves
    is trained only on the first leaf to claim it; later leaves re-emit it
    as loss_mask=0 context so the shared prefix isn't double-counted.
    """
```

验证入口：`tests/test_agent/test_trajectory_manager_branching.py` 中 cross-leaf dedup、deep multi-leaf dedup 覆盖这个不变量。

## Q5：客户端断连会怎样？

`_run_turn` 先 `_respond`，再 `record_turn`。如果 `_respond` 发现客户端断连，返回 499 或传播取消，不会记录该 turn。

```python
# 定位骨架（基于 `slime/agent/adapters/common.py` L359-L391；只展示断连处理）
            try:
                response = await self._respond(request, body, reply, in_tok, out_tok, stream)
            except (ConnectionResetError, asyncio.CancelledError) as e:
                self.logger.warning(
                    "[%s] sid=%s client disconnected before response flush: %s after %.1fs",
                    self.log_prefix,
                    sid,
                    type(e).__name__,
                    time.monotonic() - t0,
                )
                if isinstance(e, asyncio.CancelledError):
                    raise
                return web.Response(status=499, text="client disconnected")
```

排障动作：如果你自己写 adapter 或 custom generate，不要在 response flush 前记录训练轨迹。

## Q6：为什么第二次 `finish_session` 返回空？

`get_trajectory` 是 destructive read，成功线性化后会删除 sid 的 tree 和 turn count。

```python
# 定位骨架（基于 `slime/agent/trajectory.py` L307-L344；只展示消费入口）
def get_trajectory(
    self,
    sid: str,
    *,
    base_sample: Sample,
    reward: float = 0.0,
    extra_metadata: dict[str, Any] | None = None,
    max_sample_tokens: int = 0,
) -> list[Sample]:
    """Linearize this sid's routing tree into slime ``Sample`` objects and
    consume the session.
```

排障动作：custom generate 应保存第一次返回的 samples；不要期望 manager 能重复读取同一个 session。

## Q7：max turns 上限在哪里生效？

turn cap 是 adapter 层的 serving 保险丝，不属于 `TrajectoryManager`。超限时返回 429。

```python
# 定位骨架（基于 `slime/agent/adapters/common.py` L285-L306；只展示 turn cap 主干）
def _check_turn_cap(self, sid: str) -> web.Response | None:
    """Enforce max_turns_per_sid, returning a 429 response once exceeded.

    Increments the per-sid counter as a side effect when under the cap.
    """
    cap = self.max_turns_per_sid
    if cap is None:
        return None
    prior = self._sid_turn_count.get(sid, 0)
    if prior >= cap:
```

验证入口：`tests/test_agent/test_adapters.py` 的 `test_max_turns_per_sid_returns_429`。

## Q8：应该用 adapter 还是手写 custom generate？

| 场景 | 推荐 |
|------|------|
| 复用已有 OpenAI/Anthropic SDK agent | adapter |
| 简单工具循环或 RAG | 手写 `--custom-generate-function-path` |
| 一个 prompt 拆多个训练段 | custom generate 返回 `list[Sample]` |
| 跨样本调度、后台队列、fully async | `--rollout-function-path` |

官方文档的边界：

```markdown
# 定位骨架（基于 `docs/en/get_started/agent.md` L11-L26；只展示集成选型表）
| Run a custom agent loop, tool calls, RAG, browser/terminal/sandbox interaction for each sample | [`--custom-generate-function-path`](customization.md#2-custom-generate-function---custom-generate-function-path), [writing a custom generation function](quick_start.md#writing-custom-generation-function) |
| Implement verifier rewards, test-based rewards, environment success checks, or an external reward service | [`--custom-rm-path`](customization.md#3-reward-model---custom-rm-path), [writing a custom reward function](quick_start.md#writing-custom-reward-function) |
| Return multiple training samples from one prompt, such as subagent, multi-agent, or context-compaction segments | [fan-out return from custom generate](customization.md#returning-multiple-training-samples-for-one-prompt), [`examples/multi_agent`](../_examples_synced/multi_agent/README.md) |
```

## Q9：为什么不能用缺省 sid？

OpenAI 无 Bearer、`metadata.session_id`、`user` 时，Anthropic 无 Bearer、`X-Api-Key` 时，都返回 `"default"`。这不是每请求自动生成的唯一 id。同一 adapter 实例中的多个这类客户端会共享 session store、closed state、turn cap、trajectory tree，而 `default` 又不会作为 SGLang routing header 发送。

排障动作：要求每个 agent run 传稳定且全局唯一 sid；线上日志显式记录 sid，并对 `default` 设告警或直接拒绝。

## Q10：OpenAI 为什么丢了并行 tool calls？

`_build_reply_parts` 可以先构造多个 call，但最终对 `wire_tool_calls[:1]` 和 `manager_tool_calls[:1]` 切片。这是为兼容会丢额外并行 call 的客户端，代价是 parser 识别到的其余 call 不会到达 wire 或 trajectory manager。OpenAI adapter 也只注册 `/v1/chat/completions`，不实现 `/v1/responses`。

排障动作：若 agent 必须并行调用工具，不要依赖当前 OpenAI adapter 自动保真；选择串行工具策略，或明确扩展 wire/manager canonicalization 并补 round-trip 测试。

## Q11：context cap 为什么可能产生非法 sample？

`max_context_tokens` 同时限制每轮生成长度和最终 `max_sample_tokens`。`to_sample` 从右侧截断 tokens/loss/logprobs，却仍用未变的 `leading_prompt_len` 计算 response length。cap 小于首轮 prompt 时可得负 `response_length`，或得到没有训练 token 的截断结果。当前 branching 测试没有覆盖 `max_sample_tokens`。

```python
# 来源：slime/agent/trajectory.py L239-L261
        start = self.leading_prompt_len  # first-turn prompt stripped; response region starts here
        tokens = list(self.tokens)
        loss_mask = self.loss_mask
        logprobs = self.logprobs
        if max_sample_tokens and len(tokens) > max_sample_tokens:
            tokens = tokens[:max_sample_tokens]
            loss_mask = loss_mask[:max_sample_tokens]
            logprobs = logprobs[:max_sample_tokens]
        md = dict(extra_metadata or {})
        return Sample(
            index=base_sample.index,
            group_index=base_sample.group_index,
            rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
            prompt=base_sample.prompt,
            label=base_sample.label,
            tokens=tokens,
            response_length=len(loss_mask) - start,
            loss_mask=loss_mask[start:],
            rollout_log_probs=logprobs[start:],
            reward=0.0,
            status=Sample.Status.COMPLETED,
            metadata=md,
        )
```

排障动作：记录 `leading_prompt_len`、cap、截断后 token 数、`response_length` 和 `sum(loss_mask)`；发布前强制 cap 至少容纳 prompt 加最小 response。

## Q12：`finish_session` 失败后能否直接重试？

不能假设等价。adapter 先把 sid 加入 `closed` 并 pop session store，manager 又在 DFS 中就原地设 `response_trained=True`；只有全部 sample 成功建完后才 pop tree。中途异常可留下“store 已丢、tree 还在、部分 node 已 claimed”的混合状态。

排障动作：将 finish 视为一次性提交；异常时保留原始 turn/tree dump，重建 manager 而不是在原对象上盲目重试。

## Q13：fan-out reward 应该怎么核算？

manager 把输入 reward 完整赋给每个 sample；测试也明确断言“no split”。这是一个口径选择，不是 reward 保守。如果下游按 sample 直接求和而不按共享 `rollout_id` 归并，分支越多，该 rollout 的总权重越大。

排障动作：从 custom generate 追到 train-step split/loss aggregation，确认 sibling sample 是否按 rollout group 聚合；若业务需要保守 reward，由自定义逻辑显式分配。

## Q14：该跑哪些测试？

优先跑：

```powershell
python -m pytest tests/test_agent/test_trajectory_manager_branching.py tests/test_agent/test_adapters.py -q
```

更完整的 CPU agent 侧验收：

```powershell
python -m pytest tests/test_agent/ -q
```

如果环境缺少某些外部依赖，至少保留前两个文件作为 trajectory 和 adapter 的核心回归。
