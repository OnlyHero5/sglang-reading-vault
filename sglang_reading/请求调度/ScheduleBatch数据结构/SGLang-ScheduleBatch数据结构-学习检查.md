---
title: "ScheduleBatch数据结构 · 学习检查"
type: exercise
framework: sglang
topic: "ScheduleBatch数据结构"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-11
---
# ScheduleBatch数据结构 · 学习检查

## 读者能做什么

- [ ] 能画出主请求链，并把普通生成、`skip_tokenizer_init`、embedding 三种回程分开。
- [ ] 能解释 `Req` 为什么是生命周期对象，而 `TokenizedGenerateReqInput` 只是 IPC 输入。
- [ ] 能解释 live `running_batch` 为什么可跨 decode 轮次演化，overlap 的 `batch.copy()` 为什么只是结果处理快照。
- [ ] 能说出 `ScheduleBatch` 和 `ForwardBatch` 的分工，并指出 `ForwardBatch.init_new` 会消费三项 one-shot override。
- [ ] 能用 `prefix_indices`、`prefix_lens`、`extend_lens`、`seq_lens` 复述一次 prefill 的输入切分。
- [ ] 能解释 decode 轮次为什么要重新分配 `out_cache_loc`，并推进 `seq_lens`。
- [ ] 能说明 `filter_batch` 和 `merge_batch` 维护哪些 per-request 对齐关系。
- [ ] 能判断一个新增 IPC 字段应走 msgspec 类型、`enc_hook`，还是 `PickleWrapper`。
- [ ] 能严格区分 `decode_ids`（detokenize 窗口片段）、`output_ids`（客户端 token 增量）、`output_strs`（文本增量）。

## 最小源码定位

| 任务 | 入口 |
|------|------|
| 看 tokenized 请求如何发给 Scheduler | `tokenizer_manager.py::_send_one_request` |
| 看 IPC 消息如何编码 | `io_struct.py::sock_send` / `sock_recv` |
| 看 tokenized 请求如何变成 `Req` | `scheduler.py::handle_generate_request` |
| 看 prefill batch 如何创建 | `scheduler.py::get_new_batch_prefill` |
| 看本轮 extend token 如何确定 | `schedule_batch.py::prepare_for_extend` |
| 看 decode 如何推进 | `schedule_batch.py::prepare_for_decode` |
| 看 batch 如何缩小和扩大 | `schedule_batch.py::filter_batch` / `merge_batch` |
| 看执行视图如何生成、one-shot 如何消费 | `forward_batch_info.py::ForwardBatch.init_new` |
| 看 token 窗口与客户端 token 如何分别切片 | `output_streamer.py::_GenerationStreamAccumulator.accept` |
| 看 token 输出如何转字符串 | `detokenizer_manager.py::handle_batch_token_id_out` |

## 验证实验

选择一个小模型和单请求，做六类观察；没有可运行 GPU 环境时，逐项静态追踪赋值点与消费者，并保存源码位置作为替代证据：

- [ ] 在 TokenizerManager 发送前确认对象类型是 `TokenizedGenerateReqInput`，且 opaque 字段已执行 wrap。
- [ ] 在 Scheduler 构造 `Req` 后确认 `output_ids` 为空，`origin_input_ids` 已存在。
- [ ] 在第一次 prefill 后记录 `prefix_lens`、`extend_lens`、`extend_num_tokens`。
- [ ] 重复相同长 prompt 前，确认首请求 prefix 已入 cache、cache 未禁用/驱逐，且 `extra_key`、LoRA、embedding override、多模态 hash 等匹配；再观察 `prefix_lens` 是否增大、`extend_num_tokens` 是否下降，不要求完整 prompt 全命中。
- [ ] 在 `ForwardBatch.init_new` 前后观察三项 one-shot override，确认执行视图拿到旧值而原 batch 已复位。
- [ ] 在 Scheduler 输出与 Detokenizer 入口同时比较 `decode_ids`、`output_ids`、`output_strs`，解释三者长度为何可以不同。
- [ ] 分别检查普通、skip-tokenizer、embedding 回程的对象类型和路径。

## 失败模式自检

- [ ] 如果请求串输出，能先检查 `filter_batch` / `merge_batch` 是否保持 `reqs` 与张量顺序一致。
- [ ] 如果 prefix cache 没命中，能检查 `extra_key`、LoRA、embedding 覆盖、多模态 pad value。
- [ ] 如果 msgpack 报错，能判断字段是否应该显式 `wrap_as_pickle`。
- [ ] 如果 decode 阶段输入异常，能检查 `prepare_for_decode` 是否清掉 stale `input_embeds` 并重建 `out_cache_loc`。
- [ ] 如果请求结束但无响应，能检查是否中途直接写了 `finished_reason`。
- [ ] 如果 `decode_ids` 比本轮新 token 多，不会先判定重复生成，而会检查 surrounding/read 窗口与 `output_ids`。
- [ ] 如果某个 override 只生效一轮或泄漏到下一轮，能检查 `ForwardBatch.init_new` 的消费与复位。
- [ ] 如果 overlap 才出现结果错位，能区分 live running batch 与 `batch.copy()` 结果快照。

## 迁移到下一篇

读完本专题后，继续读 [[SGLang-Detokenizer]] 时关注 token 到字符串的增量 decode；继续读 [[SGLang-ModelRunner]] 时从 `ForwardBatch` 开始看执行栈；回读 [[SGLang-Scheduler]] 时重点看 `Req` 如何在 waiting/running/retracted 队列之间移动。
