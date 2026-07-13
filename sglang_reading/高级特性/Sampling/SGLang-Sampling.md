---
title: "Sampling"
type: map
framework: sglang
topic: "Sampling"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-12
---
# Sampling

## 你为什么要读

读 Sampling，不是为了背 `temperature/top_p/top_k` 参数表，而是为了回答一个更靠近源码的问题：模型已经算出下一 token 的 logits 后，SGLang 怎样把“用户想要的风格、格式和限制”变成一个真正的 token id。

读完本专题，你应该能解决三类问题：

1. OpenAI 或原生请求里的采样字段，在哪里被规范化成内部 `SamplingParams`。
2. JSON schema、regex，以及由 tool/reasoning 派生的约束，为什么可能让请求先排队而不是马上进 batch。
3. 一个 decode step 中，logits 会按什么顺序经历 penalty、grammar mask、logit bias、custom processor、greedy 或概率采样。

## 源码主线

把 Sampling 看成“下一 token 生产线”：

```text
SamplingParams
  -> GrammarManager queue
  -> SamplingBatchInfo
  -> ModelRunner._preprocess_logits
  -> Sampler.forward
  -> BatchResultProcessor commits token / advances grammar
  -> ScheduleBatch prepares the next step's penalty state
```

这不是每个请求都严格经过的单线：无 grammar 的请求跳过编译队列；speculative decode 在 verify 后一次提交多个 token；overlap 调度会提前快照 penalty；Ascend、RL on-policy 和普通 CUDA/CPU 采样也会在 `Sampler` 内部分叉。

`SamplingParams` 是第一站。它把 API 字段和内部字段放在同一个结构里，但 `stop`、`stop_regex` 这类 API alias 会在 normalize 后清掉。

```python
# 来源：python/sglang/srt/sampling/sampling_params.py L75-L120
class SamplingParams(msgspec.Struct, kw_only=True, omit_defaults=True):
    """
    The sampling parameters.

    See docs/backend/sampling_params.md or
    https://docs.sglang.io/backend/sampling_params.html
    for the documentation.
    """

    # --- API parameters (set by callers) ---
    max_new_tokens: Optional[int] = 128
    stop: Optional[Union[str, List[str]]] = (
        None  # API input alias, copied to stop_strs then cleared in normalize()
    )
    stop_token_ids: Optional[Set[int]] = None
    stop_regex: Optional[Union[str, List[str]]] = (
        None  # API input alias, copied to stop_regex_strs then cleared in normalize()
    )
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = TOP_K_ALL
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    n: int = 1
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    structural_tag: Optional[str] = None
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    no_stop_trim: bool = False
    custom_params: Optional[Dict[str, CustomParamValue]] = None
    stream_interval: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    sampling_seed: Optional[int] = None

    # --- Internal fields (populated by __post_init__ or normalize(), not API-facing) ---
    stop_strs: Optional[Union[str, List[str]]] = None  # from stop
    stop_regex_strs: Optional[Union[str, List[str]]] = None  # from stop_regex
    stop_str_max_len: int = 0  # set by normalize()
    stop_regex_max_len: int = 0  # set by normalize()
    is_normalized: bool = False  # set by normalize()
```

## 阅读路径

| 顺序 | 文档 | 读完要拿到什么 |
|------|------|----------------|
| 1 | [[SGLang-Sampling-核心概念]] | 下一 token 生产线的心理模型 |
| 2 | [[SGLang-Sampling-源码走读]] | 从请求参数到 `Sampler.forward` 的源码主线 |
| 3 | [[SGLang-Sampling-数据流]] | 每一步对象形状如何变化 |
| 4 | [[SGLang-Sampling-排障指南]] | 结构化输出、greedy、grammar timeout、penalty、determinism 排障 |
| 5 | [[SGLang-Sampling-学习检查]] | 能否不用源码复述主线，打开源码后能否定位证据 |

## 首次阅读建议

第一次只抓四个对象：

| 对象 | 作用 |
|------|------|
| `SamplingParams` | 单请求采样意图 |
| `GrammarManager` | 约束解码的异步编译和排队口 |
| `SamplingBatchInfo` | 把每个请求的采样意图变成 batch tensor |
| `Sampler.forward` | 从 logits 选择下一 token |

如果你从 OpenAI API 读过来，要注意边界：OpenAI handler 只负责把 `response_format` 等字段翻译进 `sampling_params`；Sampling 专题解释的是这些内部参数怎样真正改变 logits 和 token 选择。

再记住两个反直觉边界：`temperature=0` 不是把 logits 除以零，而是在 `SamplingParams.__post_init__` 中改写为 `temperature=1.0、top_k=1`；`sampling_seed` 只有服务器开启 deterministic inference 后才会变成 batch tensor，且不同 backend 的支持范围并不相同。
