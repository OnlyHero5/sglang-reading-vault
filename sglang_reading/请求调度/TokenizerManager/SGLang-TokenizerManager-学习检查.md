---
title: "TokenizerManager · 学习检查"
type: exercise
framework: sglang
topic: "TokenizerManager"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# TokenizerManager · 学习检查

## 读者能做什么

- [ ] 能画出 `GenerateReqInput -> ReqState -> TokenizedGenerateReqInput -> BatchStrOutput/BatchTokenIDOutput -> out_dict` 的生命周期。
- [ ] 能解释前台 `_wait_one_response` 和后台 `handle_loop` 如何通过 `ReqState.event` 协作。
- [ ] 能说出 `rid` 与 `http_worker_ipc` 的区别，并解释多 HTTP worker 为什么需要二者。
- [ ] 能指出 `skip_tokenizer_init=True` 对输入和输出两侧分别改变了什么。
- [ ] 能解释权重更新、pause、reader/writer lock 如何阻止新请求进入 Scheduler。
- [ ] 能说明 `text=None`、chunk coalesce、`batch_notify_size` 三者和 streaming 输出的关系。
- [ ] 能把 score API、flush cache、多 worker router 归类为主线分叉，而不是新后端。

## 最小复述

不看正文，尝试用 5 句话复述：

1. HTTP route 不直接分词或读 ZMQ，而是消费 `TokenizerManager.generate_request` async generator。
2. `generate_request` 为每个 `rid` 创建 `ReqState`，再等待 pause、进入 reader lock、分词并发送 tokenized IPC object。
3. 后台 `handle_loop` 收到批量输出后，按 `rid` 写回对应 `ReqState.out_list` 并 set event。
4. `_wait_one_response` 被唤醒后 drain 输出，按 streaming 配置决定返回 delta、完整文本或 token ids。
5. 控制面请求走 `FanOutCommunicator`，多 worker 回包靠 `http_worker_ipc` 路由。

## 可执行验证

| 验证 | 操作 | 预期现象 |
|------|------|----------|
| 普通非流式 generate | 在 `generate_request`、`_init_req_state`、`_handle_batch_output`、`_wait_one_response` 加日志，发送一次非流式请求 | 同一 `rid` 先注册，finished 回包后删除，HTTP 只 yield 一次 |
| incremental streaming | 启动 `--incremental-streaming-output`，发送 `stream=True` 请求 | 每个 chunk 是 delta；消费慢时多个 chunk 可能 coalesce |
| skip tokenizer | 启动 `--skip-tokenizer-init`，分别发送 text 和 `input_ids` | text 请求在 TokenizerManager 报错；`input_ids` 请求进入后端并返回 token ids |
| 权重更新互斥 | 在 update weights 期间发送新 generate | 新请求停在 pause/reader lock 之前，不应先进入 Scheduler |
| 多 worker 回包 | `tokenizer_worker_num > 1` 时记录 `http_worker_ipc` | 回包应被 router 拆到 owner worker，非 owner worker 不应持有该 `rid` state |

## 易错判断

| 易错说法 | 正确说法 |
|----------|----------|
| TokenizerManager 负责 batching | 它最多做 tokenization batch 或 IPC batch；GPU continuous batching 是 Scheduler 的职责 |
| Detokenizer 总会参与输出回程 | `skip_tokenizer_init=True` 下主链路可以让 TokenizerManager 直接处理 `BatchTokenIDOutput` |
| `text=None` 表示 Detokenizer 丢文本 | 非 incremental streaming 中间包可能故意延迟完整 text materialize |
| pause 只影响 Scheduler | TokenizerManager 本地也维护 `is_pause`，新请求在分词发送前等待 |
| 多 worker 只要 rid 唯一就够 | 回包还必须知道 owner worker 的 `http_worker_ipc` |

## 下一步

- 想看请求进入 Scheduler 后如何变形，读 [[SGLang-ScheduleBatch数据结构]]。
- 想看文本增量如何从 token ids 产生，读 [[SGLang-Detokenizer]]。
- 想看 API 层如何构造 `GenerateReqInput`，读 [[SGLang-OpenAI-API]]。
