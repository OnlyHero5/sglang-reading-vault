---
title: "SGLang 服务实验"
type: exercise
framework: sglang
topic: "推理 Serving"
learning_role: practice
source_baseline: "70df09b"
difficulty: intermediate
estimated_time: "60 到 120 分钟"
prerequisites:
  - "[[推理Serving主线]]"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-13
---

# SGLang 服务实验

## 读者任务

在一台可运行小模型的 GPU 主机上，完成“启动—健康检查—流式请求—prefix 对照—overlap 对照—KV 压力—停止服务”的闭环，并用 response metadata、Scheduler 日志和 Prometheus metrics 区分请求入口、调度、KV 与文本回程问题。

## 静态模式

没有可用 GPU 时，依次定位：

```powershell
rg -n 'generate_request|_wait_one_response|event_loop_overlap|event_loop_normal|retract_decode' sglang/python/sglang/srt
```

预期：能把 route、等待状态、两种事件循环和 KV retract 对应到 [[SGLang-HTTP请求全链路]] 的位置。

再确认实验使用的 CLI 与 metrics 名称仍存在：

```powershell
rg -n 'model_path:|mem_fraction_static:|disable_radix_cache:|disable_overlap_schedule:|enable_metrics:' sglang/python/sglang/srt/server_args.py
rg -n 'sglang:cache_hit_rate|sglang:token_usage|sglang:num_retracted_requests_total' sglang/python/sglang/srt/observability/metrics_collector.py
```

预期：当前参数由 `ServerArgs` dataclass 生成 CLI flag；metrics 能定位 cache hit、KV/token pool 使用和 retract 累计值。

## 运行环境与启动

推荐 Linux 或 WSL2 + NVIDIA GPU。以下命令从知识库根目录执行；模型可替换，但必须在实验记录里写明具体 revision、GPU、dtype 与 TP。

```powershell
$env:PYTHONPATH = (Resolve-Path 'sglang/python').Path
$MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'

sglang serve `
  --model-path $MODEL `
  --host 127.0.0.1 `
  --port 30000 `
  --mem-fraction-static 0.70 `
  --enable-metrics
```

如果当前环境尚未生成 `sglang` console script，可使用仍受支持的兼容入口：

```powershell
python -m sglang.launch_server --model-path $MODEL --host 127.0.0.1 --port 30000 --mem-fraction-static 0.70 --enable-metrics
```

预期：模型加载完成、warmup 成功，`http://127.0.0.1:30000/health` 返回 `200`。端口开始监听但 health 仍为 `503`，表示服务仍在 Starting 或准备退出，不能开始压测。

在第二个终端检查：

```powershell
curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:30000/health
curl.exe -s http://127.0.0.1:30000/model_info
```

## 流式请求

发送流式请求：

```powershell
curl.exe -N http://127.0.0.1:30000/generate `
  -H "Content-Type: application/json" `
  -d '{"text":"Explain KV cache in one paragraph.","sampling_params":{"temperature":0,"max_new_tokens":64},"stream":true}'
```

预期：收到多条 SSE `data:` chunk，最后出现带 `finish_reason` 的终态结果；同一请求的 `meta_info.id` 保持一致。若 Scheduler 已输出 token id 而客户端没有文本，转入 Detokenizer/TokenizerManager 回程排查。

## Prefix cache 对照

用同一个长前缀连续发送两次请求，并直接读取 native response 的 `cached_tokens`：

```powershell
@'
import json
import urllib.request

url = "http://127.0.0.1:30000/generate"
prefix = "You are a systems tutor. Explain with invariants and failure modes. " * 40

def call(question):
    body = json.dumps({
        "text": prefix + question,
        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    meta = out["meta_info"]
    print({k: meta.get(k) for k in (
        "id", "prompt_tokens", "completion_tokens", "cached_tokens",
        "num_retractions", "e2e_latency", "decode_throughput")})

call(" What is KV cache?")
call(" What is continuous batching?")
'@ | python -
```

预期：第二次请求的 `cached_tokens` 明显大于第一次；`prompt_tokens` 仍表示完整 prompt 长度。cache hit 是否改善端到端延迟取决于模型、并发和调度，不能由一次请求推导生产阈值。

随后停止服务，以相同参数增加 `--disable-radix-cache` 重启并重复。预期：`cached_tokens` 不再体现相同前缀复用。若前缀含时间戳、随机数或不一致 chat template，两次请求即使“看起来相似”也可能无法命中。

## Overlap 对照

分别使用默认配置和增加 `--disable-overlap-schedule` 的配置重启服务，其他参数完全相同。每次 warmup 后运行：

```powershell
python -m sglang.benchmark.serving `
  --backend sglang `
  --base-url http://127.0.0.1:30000 `
  --dataset-name random `
  --num-prompts 32 `
  --random-input-len 256 `
  --random-output-len 64 `
  --request-rate 4
```

记录：

- TTFT P50/P99
- TPOT
- output token throughput
- running batch size
- GPU 利用率

预期：关闭 overlap 后调用顺序更适合断点追踪；TTFT、TPOT 和吞吐的方向由当前 workload 决定。报告必须包含完整启动命令与 benchmark 命令，否则对照无效。

## KV 压力实验

先保留输入输出长度，将 `--request-rate` 从 1、2、4 逐级增加；再固定 request rate，把 `--random-output-len` 从 64 增到 128。每档结束读取：

```powershell
curl.exe -s http://127.0.0.1:30000/metrics | Select-String 'sglang:(token_usage|full_token_usage|cache_hit_rate|num_running_reqs|num_queue_reqs|num_retracted_requests_total)'
```

安全停止条件：出现 OOM、错误率上升、`token_usage` 持续高于你预先记录的阈值，或 retract/排队持续增长时立即停止加压。阈值必须按本次硬件和模型记录，不能写成框架通用常数。

预期：压力增加时 running/queue、KV pool usage 和延迟可能上升；资源不足时可能出现 retract。`token_usage` 是多个 pool 中的瓶颈汇总值，混合 SWA 模型还应同时看 `full_token_usage` 与 `swa_token_usage`。

## 收尾

用 `Ctrl+C` 停止前台 server，确认端口释放；不要通过模糊进程名批量杀死共享机器上的其他任务。

## 通过标准

- [ ] 能解释一次 stream 的每个进程边界。
- [ ] 能给出 prefix hit 与 force miss 的对照结果。
- [ ] 能说明 overlap 对吞吐和调试体验的影响。
- [ ] 能用指标区分 KV 不足、GPU forward 卡住和文本回程卡住。
- [ ] 实验记录包含启动命令、模型 revision、GPU、输入输出长度、并发与停止条件。
