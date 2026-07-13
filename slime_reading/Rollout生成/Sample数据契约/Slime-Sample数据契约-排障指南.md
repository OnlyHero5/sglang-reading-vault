---
title: "Sample数据契约 · 排障指南"
type: troubleshooting
framework: slime
topic: "Sample数据契约"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# Sample数据契约 · 排障指南

这篇按症状排障。遇到 Sample 问题时，先判断是哪条边界坏了：response 时间轴对齐、rollout 输出形状、reward 标量化、debug/Ray 序列化，还是插件动态加载。

## 快速定位表

| 症状 | 可能原因 | 源码入口 | 验证方法 |
|------|----------|----------|----------|
| PPO/GRPO ratio 缺 old logprob | trainable token 没传 `log_probs` | `Sample.append_response_tokens` | 跑 `test_append_response_tokens_requires_trainable_log_probs` |
| tool token 参与了 loss | 非训练 token 没用 `trainable=False` | `append_response_tokens`、`loss_mask` | 看该 token 对应 mask 是否为 0 |
| top-p replay 报 offsets 错 | ids/offsets 缺一或长度不等于 `response_length + 1` | `_extract_rollout_top_p_token_data`、`_validate_response_metadata_lengths` | 检查 offsets 首尾和 token id 总数 |
| routed experts reshape 失败 | 误按 response 长度检查，或只传当前 chunk | `_apply_meta_info`、`fill_routing_replay` | 检查首维为 `len(tokens)-1`，meta 为完整快照 |
| compact rollout loss 被重复计数 | sibling 没共享 `rollout_id` | `_validate_rollout_id_annotated` | 看嵌套输出 flatten 前的 leaf list |
| filter 后样本还在 batch 里 | `remove_sample` 只置零 mask，不删除行 | `_convert_samples_to_train_data` | 看 `loss_masks` 是否全 0 |
| reward 是 dict 后训练报类型错 | 没设置 `args.reward_key` | `Sample.get_reward_value` | 打印 `sample.reward` 和 `args.reward_key` |
| debug load 后状态或扩展字段丢失 | `to_dict/from_dict` 兼容边界坏 | `Sample.from_dict` | 跑 `test_round_trip_preserves_every_field` |
| 插件路径加载失败 | `"module.attr"` 路径错误或函数不在顶层 | `load_function` | 在 Python 中单独 import 该路径 |

## 1. 什么时候必须显式设置 rollout_id？

默认 rollout 形状是 `list[list[Sample]]`，leaf 在 depth 1，Slime 保持兼容，不强制 `rollout_id`。compact/subagent 形状多一层，表示一次 rollout execution 拆出多条 training sample，这些 sibling 必须共享同一个 `rollout_id`。

```python
# 定位骨架（据 `slime/ray/rollout.py` L898-L925 删节）：
if node and isinstance(node[0], Sample):
    if depth >= 2 and len(node) > 1:
        rids = [s.rollout_id for s in node]
        missing = [i for i, r in enumerate(rids) if r is None]
        assert not missing, (
            f"Compact rollout returned {len(node)} samples but rollout_id is unset on "
            f"positions {missing}. Set Sample.rollout_id on every sibling so the loss "
            "reducer can aggregate them as one rollout instead of N."
        )
        assert len(set(rids)) == 1, f"Sibling samples from one compact rollout must share rollout_id; got {rids}."
    return
```

排查：在 rollout 函数返回前打印嵌套结构，不要等 RolloutManager flatten 后再看。

## 2. loss_mask 全 0 的 sample 会被删掉吗？

不会。`remove_sample=True` 会把这条 sample 的 `loss_mask` 改成全 0，但对象仍留在 batch 里，便于保持分组、日志和指标形态。

```python
# 定位骨架（据 `slime/ray/rollout.py` L747-L778 删节）：
for sample in samples:
    if sample.loss_mask is None:
        sample.loss_mask = [1] * sample.response_length

    assert (
        len(sample.loss_mask) == sample.response_length
    ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
    if sample.remove_sample:
        sample.loss_mask = [0] * sample.response_length
    loss_masks.append(sample.loss_mask)
train_data["loss_masks"] = loss_masks
```

验证：转换后看 `train_data["loss_masks"][i]`，而不是只看样本是否还在列表里。

## 3. trainable token 为什么必须带 log_probs？

PPO/GRPO 需要 rollout engine 侧的 old logprob 计算 ratio 或 KL。Slime 不允许可训练 token 没有 logprob，因为这种错误拖到 loss 阶段会更难定位。

```python
# 定位骨架（据 `slime/utils/types.py` L253-L303 删节）：
tokens = _to_int_list(tokens)
log_probs = _to_float_list(log_probs)
if log_probs is not None and len(log_probs) != len(tokens):
    raise ValueError(f"log_probs length {len(log_probs)} != tokens length {len(tokens)}")
if tokens and trainable and log_probs is None:
    raise ValueError("trainable response tokens require rollout log probabilities.")
if tokens and not trainable:
    if log_probs is not None:
        raise ValueError("non-trainable response tokens should not pass rollout log probabilities.")
    log_probs = [0.0] * len(tokens)
```

验证：自定义 generate 里模型生成 token 应调用 `append_response_tokens(tokens=tokens, trainable=True, log_probs=log_probs)`；tool/environment token 应调用 `trainable=False` 且不要传真实 logprob。

## 4. top-p replay 只有 ids 或只有 offsets 会怎样？

会直接报错。top-p replay 是 ragged 表，ids 与 offsets 必须成对出现；offsets 长度必须等于生成 token 数 + 1。

```python
# 定位骨架（据 `slime/utils/types.py` L13-L36 删节）：
token_ids = decode_int32_meta_array(meta_info, _TOP_P_TOKEN_ID_META_KEYS)
offsets = decode_int32_meta_array(meta_info, _TOP_P_TOKEN_OFFSET_META_KEYS)
if token_ids is None and offsets is None:
    return None
if token_ids is None or offsets is None:
    raise ValueError("SGLang top-p token replay must include both token ids and offsets.")
if offsets.numel() == 0 or int(offsets[0]) != 0:
    raise ValueError(f"SGLang top-p token offsets must start with 0, got {offsets[:1].tolist()}.")
if int(offsets[-1]) != token_ids.numel():
    raise ValueError(
        "SGLang top-p token ids/offsets mismatch: "
        f"offsets[-1]={int(offsets[-1])}, len(token_ids)={token_ids.numel()}."
    )
```

验证：对每条 sample 检查 `len(offsets) == response_length + 1` 和 `offsets[-1] == len(token_ids)`。

## 5. streaming 中间 chunk 为什么不要更新终态？

中间 chunk 还不是完整 response。`append_response_tokens` 提供 `update_terminal_info=False`，让 tokens、mask、logprob、top-p 先增长，但不改变 `Sample.Status`、prefix cache 统计和 `weight_versions` 等终态信息。

```python
# 定位骨架（据 `slime/utils/types.py` L316-L381 删节）：
if not update_terminal_info or "finish_reason" not in meta_info:
    return

if getattr(args, "sglang_speculative_algorithm", False):
    self.spec_info.add(meta_info=meta_info)

self.prefix_cache_info.add(meta_info=meta_info)

if "weight_version" in meta_info:
    self.weight_versions.append(meta_info["weight_version"])
```

验证：`tests/test_rollout_metrics.py` 的 `test_append_response_tokens_can_skip_terminal_status_for_streaming_chunks` 覆盖了中间 chunk 不改 status 的行为。

## 6. routed experts 为什么不能按 response_length 检查？

它的首维对应整条 `tokens` 的 next-token 位置，契约是 `len(tokens)-1`。默认 SGLang rollout 先把 prompt ids 写进 `sample.tokens`；训练侧 `fill_routing_replay` 也再次断言 `experts.shape[0] == tokens.shape[0]-1`。此外 `_apply_meta_info` 每次直接覆盖该字段，不做 chunk merge。

验证：元数据元素数应等于 `(len(sample.tokens)-1) * num_layers * moe_router_topk`。streaming 或多轮路径若只返回增量 chunk，会在 reshape 时失败或覆盖历史。

## 7. reward 为 dict 时如何取标量？

通过 `args.reward_key`。如果 reward 是 dict 但没有配置 key，后续把 dict 当 float 用时会出错。

```python
# 来源：slime/utils/types.py L246-L251
def get_reward_value(self, args) -> float:
    return self.reward if not args.reward_key else self.reward[args.reward_key]

@property
def effective_response_length(self):
    return sum(self.loss_mask) if self.loss_mask is not None else self.response_length
```

验证：打印 `sample.reward` 的类型。如果是 dict，确认训练和评估使用的 `reward_key` / `eval_reward_key` 指向同一个语义字段。

## 8. legacy rollout 返回裸 list 还能用吗？

能，但建议新代码返回 `RolloutFnTrainOutput`。`call_rollout_fn` 会把裸训练输出包装成 `RolloutFnTrainOutput(samples=output)`，把裸评估输出包装成 `RolloutFnEvalOutput(data=output)`。

```python
# 来源：slime/rollout/base_types.py L19-L26
def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    return output
```

验证：插件合同测试 `tests/plugin_contracts/test_plugin_rollout_contracts.py` 覆盖了 legacy wrapper。

## 9. load_function 路径写错会怎样？

直接抛 import 或 attribute 异常，不会 fallback。路径必须是模块顶层可 import 的 `"package.module.function"` 或类路径。

```python
# 来源：slime/utils/misc.py L37-L45
def load_function(path):
    """
    Load a function from a module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
```

验证：在同一个 `PYTHONPATH` 下执行最小 import；如果函数定义在闭包或 notebook cell 里，`load_function` 找不到。

## 10. ABORTED 和 FAILED 如何区分？

`ABORTED` 是终态映射的一部分，来自 SGLang `finish_reason.type == "abort"`；`FAILED` 是 rollout 逻辑显式设置的可恢复失败，可能仍有部分输出。

```python
# 定位骨架（据 `slime/utils/types.py` L130-L140 选取枚举）：
class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"

status: Status = Status.PENDING
```

排查：如果是用户取消、engine abort 或 partial rollout abort，优先用 `ABORTED`；如果是工具/API/解析失败且上层希望保留部分上下文，可显式设 `FAILED`。未知 finish reason 不会自动变成 `FAILED`，而是保持原 status。

## 11. debug load 后字段丢了怎么办？

先看 `to_dict/from_dict`。它会转换 enum 和嵌套统计对象，并保留未知字段。如果字段是不可序列化对象，问题通常不在 `Sample.from_dict`，而在保存前的数据类型。

```python
# 定位骨架（据 `slime/utils/types.py` L222-L244 删节）：
def to_dict(self):
    value = self.__dict__.copy()
    value["status"] = self.status.value
    value["spec_info"] = self.spec_info.to_dict()
    value["prefix_cache_info"] = self.prefix_cache_info.to_dict()
    return value

@staticmethod
def from_dict(data: dict):
    data = dict(data)
    data["status"] = Sample.Status(data["status"])
    data["spec_info"] = Sample.SpecInfo.from_dict(data.get("spec_info", {}))
    data["prefix_cache_info"] = Sample.PrefixCacheInfo.from_dict(data.get("prefix_cache_info", {}))
```

验证：round-trip 只证明字段保真；还要单独调用 `_validate_response_metadata_lengths()`，或通过 `append_response_tokens` 触发校验。测试中本就长度不一致的对象也能原样 round-trip。扩展字段若是外部对象，还要确认后续持久化格式是否支持。
