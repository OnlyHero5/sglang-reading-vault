---
title: "多模态生成 · 学习检查"
type: exercise
framework: sglang
topic: "多模态生成"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---

# 多模态生成 · 学习检查

## 你为什么要做这组检查

目标是确认你不会把扩散 pipeline 服务与 LLM SRT 混为一谈，并能追踪请求如何进入 Scheduler 和多 GPU PipelineExecutor。

## 能力检查

- [ ] 能说明 `multimodal_gen` 与 LLM SRT 是并列运行时。
- [ ] 能画出 HTTP/broker → Scheduler → GPUWorker → PipelineExecutor。
- [ ] 能说明 `launch_server`、`run_scheduler_process`、`GPUWorker` 的职责。
- [ ] 能解释 master/slave Pipe 与多 GPU pipeline 的关系。
- [ ] 能判断故障来自入口、调度、pipeline 配置还是具体 stage。

## 最小验证

操作：

```powershell
rg -n "def launch_server|run_scheduler_process|class GPUWorker|class PipelineExecutor|ComposedPipeline" sglang/python/sglang/multimodal_gen
```

预期：能找到服务启动、scheduler 进程、GPU worker 和 pipeline 执行器，并说明它们的创建顺序。若 pipeline 已创建但某个模型失败，继续检查对应 pipeline config 和 stage 实现。

## 复盘

主链见 [[SGLang-多模态生成-源码走读]]，跨进程对象见 [[SGLang-多模态生成-数据流]]。
