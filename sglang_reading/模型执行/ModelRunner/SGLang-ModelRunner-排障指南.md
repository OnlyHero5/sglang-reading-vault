---
title: "ModelRunner · 排障指南"
type: troubleshooting
framework: sglang
topic: "ModelRunner"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# ModelRunner · 排障指南

## 你为什么要读

本页是 ModelRunner 排障入口。读完后，你应该能把“没有走 CUDA Graph”“prefill 仍是 eager”“PP rank 没 logits”“structured output 延迟采样”等症状分别落到 Worker 门面、ForwardBatch、runner 选路或结果处理边界。

## 1. Scheduler 为什么不直接调用 ModelRunner？

**症状：** 调用链看起来多绕了一层 `TpModelWorker`。

**原因：** Worker 是 Scheduler 和 rank 本地执行之间的稳定门面。它屏蔽 TP/PP、draft worker、embedding、在线权重更新、HiCache consumer、WAR barrier 等差异。Scheduler 面向 batch 语义，ModelRunner 面向 rank 执行现场。

源码入口：来源：python/sglang/srt/managers/tp_worker.py L63-L101

源码入口：来源：python/sglang/srt/managers/tp_worker.py L225-L315

**验证：** 看 Worker 上的方法不只有 generation forward，还包含 embedding forward、weight update、memory pool、tokenizer/processor 和 PP/world group 初始化。

## 2. `ScheduleBatch` 和 `ForwardBatch` 到底差在哪？

**症状：** 调试时同一个字段在 Scheduler 和 ModelRunner 两边都能看到，不清楚谁负责更新。

**原因：** `ScheduleBatch` 是调度态，可能继续被 merge、filter、retract；`ForwardBatch` 是本次 forward 的执行快照，关注 tensor、索引、长度、KV 写入位置和采样信息。

源码入口：来源：python/sglang/srt/model_executor/forward_batch_info.py L14-L26

源码入口：来源：python/sglang/srt/model_executor/forward_batch_info.py L613-L722

**验证：** 在 `ForwardBatch.init_new` 看 `seq_lens_cpu_cache` 的 shape 断言。如果 Scheduler 修改了 batch size 却复用旧 CPU mirror，这里会直接暴露。

## 3. decode 为什么没有走 CUDA Graph？

**症状：** decode 吞吐低，profiling 里 launch 开销明显，结果包里 `can_run_cuda_graph=False`。

**可能原因：**

- `forward_mode` 不在 graph 支持集合里。
- `decode_cuda_graph_runner` 没有初始化。
- 当前 batch shape 不能匹配 capture bucket。
- DP/attention TP/CP padding 约束过滤了目标 batch size。
- 开了某些 backend 或高级模式后 runner 判定不能 replay。

源码入口：来源：python/sglang/srt/model_executor/model_runner.py L3048-L3141

源码入口：来源：python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py L58-L100

源码入口：来源：python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py L930-L1045

**验证：** 启动时找 `Capture target decode CUDA graph begin` 的 bs 列表，再对照运行时 batch size。若 bs 不在 capture 集合或被并行约束过滤，decode 会退到 eager。

## 4. prefill 为什么还是 eager？

**症状：** 以为 prefill graph 会生效，但 `_forward_raw` 最后进入 `eager_runner.execute`。

**可能原因：**

- prefill graph runner 实际是 `EagerRunner`。
- `cuda_graph_config.prefill.bs` 为空。
- 模型不是标准 language model 或找不到 layers。
- EAGLE target 在特定 backend 下主动禁用 prefill graph。
- 当前 batch 不能通过 prefill graph runner 的 `can_run_graph`。
- 当前启用了 CP strategy，源码条件会阻止 prefill graph 分支。

源码入口：来源：python/sglang/srt/model_executor/model_runner.py L2670-L2812

源码入口：来源：python/sglang/srt/model_executor/model_runner.py L3093-L3133

**验证：** 启动日志应出现 `Capture target prefill CUDA graph begin/end`。若出现 `Disable prefill CUDA graph`，按日志原因回到上面的条件排查。

## 5. PP 非末 rank 为什么没有 logits？

**症状：** 某个 PP rank forward 返回了结果，但没有 `next_token_ids`。

**原因：** PP 非末 rank 只负责传 hidden states 给下一 stage。只有 `pp_group.is_last_rank` 的 Worker 才把输出当 logits，并调用 `ModelRunner.sample`。

源码入口：来源：python/sglang/srt/managers/tp_worker.py L506-L572

**验证：** 在非末 rank 看 `GenerationBatchResult.pp_hidden_states_proxy_tensors`，在末 rank 看 `logits_output` 与 `next_token_ids`。

## 6. structured output 下为什么延迟采样？

**症状：** `forward_batch_generation` 返回后暂时没有 `next_token_ids`，但请求没有失败。

**原因：** overlap + grammar 下，Worker 把采样封装成 `delay_sample_func`。Scheduler 稍后在 forward stream 上执行闭包，再做 relay 和 D2H copy。这样可以避免和调度流互相阻塞，同时在采样后释放 vocab mask 和 logits tensor。

源码入口：来源：python/sglang/srt/managers/tp_worker.py L524-L537

源码入口：来源：python/sglang/srt/managers/scheduler.py L3404-L3432

源码入口：来源：python/sglang/srt/model_executor/model_runner.py L3143-L3159

**验证：** 观察 `delay_sample_func` 被执行后应填入 `next_token_ids`，随后闭包被清空，`logits_output.next_token_logits` 被置空以释放显存。

## 7. 在线更新权重后 graph 要不要重建？

**症状：** 权重热更新成功，但后续 decode graph 行为不符合预期。

**原因：** Worker 把更新请求转给 ModelRunner。对磁盘更新路径，只有请求参数 `recapture_cuda_graph` 为真且设备支持 graph 时，ModelRunner 才会重新初始化 decode graph。

源码入口：来源：python/sglang/srt/managers/tp_worker.py L103-L108

源码入口：来源：python/sglang/srt/model_executor/model_runner.py L1804-L1876

**验证：** 更新请求里确认 `recapture_cuda_graph`，更新后看是否再次出现 decode graph capture 日志。

## 8. embedding 请求为什么没有采样？

**症状：** embedding 请求走了 ModelRunner，但没有 token 输出。

**原因：** embedding 路径调用 `forward_batch_embedding`，它只做 `ForwardBatch.init_new` 和 `model_runner.forward`，返回 `EmbeddingPoolerOutput`。它不走 `GenerationBatchResult.next_token_ids`。

源码入口：来源：python/sglang/srt/managers/tp_worker.py L219-L222

源码入口：来源：python/sglang/srt/managers/scheduler.py L3350-L3368

**验证：** 看 `EmbeddingBatchResult.embeddings` 与 `pooled_hidden_states`，不要追 generation 的采样字段。

## 9. HiCache 或 HiSparse 问题该从哪里切入？

**症状：** KV 分层、host cache 或 sparse KV 相关行为和 forward 对不上。

**原因：** HiCache consumer index 在 Worker 构造 `ForwardBatch` 前同步；HiSparse coordinator 在 `_forward_raw` 的 decode 分支中挂到 `forward_batch`，并等待 pending backup。

源码入口：来源：python/sglang/srt/managers/tp_worker.py L440-L443

源码入口：来源：python/sglang/srt/managers/tp_worker.py L490-L495

源码入口：来源：python/sglang/srt/model_executor/model_runner.py L3071-L3078

**验证：** 先确认 batch 的 `hicache_consumer_index` 是否传入 Worker，再看 decode 分支是否设置 `forward_batch.hisparse_coordinator`。

## 排障顺序

1. 先看 `forward_mode`。
2. 再看 PP rank 是否是末 rank。
3. 再看 `_forward_raw` 的 graph 条件。
4. 再看 Worker 是否延迟采样。
5. 最后看 Scheduler result processor 是否按 mode 消费结果。

## 运行验证

ModelRunner FAQ 的最小验证要覆盖 `forward_mode`、graph 条件、PP 输出、延迟采样、embedding 分支和 HiCache/HiSparse 入口。

```powershell
rg -n 'forward_mode|_forward_raw|can_run_cuda_graph|PPProxyTensors|is_generation|sample\(|EmbeddingBatchResult|forward_batch_embedding|hicache_consumer_index|hisparse_coordinator|process_batch_result|GenerationBatchResult' sglang/python/sglang/srt/model_executor/model_runner.py sglang/python/sglang/srt/model_executor/forward_batch_info.py sglang/python/sglang/srt/managers/tp_worker.py sglang/python/sglang/srt/managers/scheduler.py
```

读输出时先按 `forward_mode` 判断 prefill/decode/embedding；再看 `_forward_raw` 是否满足 graph replay 条件。PP 问题看 `PPProxyTensors`，采样问题看 `tp_worker.py` 里何时调用 `sample`，embedding 问题看 `EmbeddingBatchResult`，KV 分层问题看 `hicache_consumer_index` 和 `hisparse_coordinator`。
