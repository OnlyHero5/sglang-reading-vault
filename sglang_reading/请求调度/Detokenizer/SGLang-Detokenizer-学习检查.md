---
title: "Detokenizer · 学习检查"
type: exercise
framework: sglang
topic: "Detokenizer"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-11
---
# Detokenizer · 学习检查

## 读者能做什么

- [ ] 能画出 `SchedulerOutputStreamer → BatchTokenIDOutput → DetokenizerManager → BatchStrOutput → TokenizerManager ReqState`。
- [ ] 能解释为什么 Detokenizer 是独立 CPU 进程，而不是 Scheduler 内部函数。
- [ ] 能说清 `surr_offset`、`read_offset`、`sent_offset` 的区别。
- [ ] 能解释 streaming 时 `output_strs[i]` 为什么是增量文本。
- [ ] 能区分 `decode_ids` 窗口片段、`output_ids` 客户端 token delta、`output_strs` 文本 delta。
- [ ] 能区分 Detokenizer 内部文本 delta 与 TokenizerManager 对客户端暴露的 incremental/累积模式。
- [ ] 能说明 UTF-8 不完整时为什么不推进 token offset。
- [ ] 能区分普通文本回程、embedding 透传回程和 `skip_tokenizer_init` token-id 回程。
- [ ] 能说明多 detokenizer worker 为什么按 `http_worker_ipc` 做粗粒度亲和，以及它可能怎样造成负载倾斜。

## 源码定位自测

| 问题 | 应定位到 |
|------|----------|
| Detokenizer 子进程在哪里启动 | `Engine._launch_detokenizer_subprocesses` |
| Scheduler 如何打包输出 token | `SchedulerOutputStreamer._stream_output_generation`、`_GenerationStreamAccumulator.accept` |
| 增量 decode 状态在哪里 | `DecodeStatus` |
| replacement char 如何处理 | `_decode_batch_token_id_output` streaming 分支 |
| `BatchStrOutput` 如何组装 | `handle_batch_token_id_out` |
| skip 模式如何绕过 Detokenizer | `SchedulerIpcChannels.create` |
| TokenizerManager 如何消费文本 | `TokenizerManager._handle_batch_output` |
| 多 worker 如何切分回包 | `MultiDetokenizerRouter`、`multi_http_worker_event_loop` |

## 运行验证自测

- [ ] 能启动服务后确认存在 `sglang::detokenizer` 进程。
- [ ] 能做一个 streaming 中文或 emoji 请求，并解释为什么不会永久输出 replacement char。
- [ ] 能观察普通模式下 `BatchStrOutput.output_strs` 进入 TokenizerManager。
- [ ] 能分别记录 `decode_ids`、`output_ids`、`output_strs`，说明首包三者长度为什么可以不同。
- [ ] 能开启 `--skip-tokenizer-init` 并解释为什么输出不含文本增量。
- [ ] 能调小 `SGLANG_DETOKENIZER_MAX_STATES` 说明状态容量错误的触发条件。

## 复盘问题

1. 为什么 Detokenizer 不能只 decode 最新 token？
2. 为什么 `sent_offset` 可能大于 `decoded_text_len`？
3. 为什么 finished 分支要删除 `decode_status[rid]` 后再发送最后一段增量？
4. 如果多 worker 下 `http_worker_ipcs` 缺失，会在哪一层失败？
5. 如果要给输出增加一个 per-request tensor 字段，应在 Scheduler 输出、Detokenizer 编码、TokenizerManager 消费三处分别检查什么？
6. 为什么两个不同 rid 可能固定落到同一个 Detokenizer worker？这对状态正确性和负载均衡分别意味着什么？
