---
title: "SGLang 服务实验"
type: exercise
framework: sglang
topic: "推理 Serving"
learning_role: practice
difficulty: intermediate
estimated_time: "60 到 120 分钟"
prerequisites:
  - "[[推理Serving主线]]"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---

# SGLang 服务实验

## 学习目标

验证 HTTP stream、prefix cache、overlap 和 KV 压力对可观测现象的影响。

## 静态模式

没有可用 GPU 时，依次定位：

```powershell
rg -n 'generate_request|_wait_one_response|event_loop_overlap|event_loop_normal|retract_decode' sglang/python/sglang/srt
```

预期：能把 route、等待状态、两种事件循环和 KV retract 对应到 [[SGLang-HTTP请求全链路]] 的位置。

## 运行模式准备

使用能够在当前 GPU 上运行的小模型启动 SGLang。模型路径和 TP 参数按本机环境调整；不要直接复制生产参数。

发送流式请求：

```bash
curl -N http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain KV cache in one paragraph.","sampling_params":{"max_new_tokens":64},"stream":true}'
```

预期：收到多条 SSE `data:` chunk，最后正常结束；同一请求保持同一个 id。

## Prefix cache 对照

固定 system prompt，连续发送两次仅用户问题不同的请求。记录 TTFT、matched prefix token 和 cache hit 指标。再关闭 radix cache 或强制 miss 重复实验。

预期：命中时第二次请求的 prefill 工作量下降；如果 prompt 含动态时间戳，命中率可能明显降低。

## Overlap 对照

分别使用默认配置和 `--disable-overlap-schedule`，保持 workload 一致。

记录：

- TTFT P50/P99
- TPOT
- output token throughput
- running batch size
- GPU 利用率

预期：关闭 overlap 后断点和调用顺序更直接；性能可能因模型、batch 和硬件而变化。结论必须附带模型、GPU、输入输出长度和并发。

## KV 压力实验

逐步提高并发和输出长度，观察 KV usage、retract 和失败率。不要直接把显存压满；设置明确停止阈值。

预期：KV 接近容量时可能出现 retract 或排队上升；如果有 token id 输出但没有文本，应检查 Detokenizer 路径。

## 通过标准

- [ ] 能解释一次 stream 的每个进程边界。
- [ ] 能给出 prefix hit 与 force miss 的对照结果。
- [ ] 能说明 overlap 对吞吐和调试体验的影响。
- [ ] 能用指标区分 KV 不足、GPU forward 卡住和文本回程卡住。
