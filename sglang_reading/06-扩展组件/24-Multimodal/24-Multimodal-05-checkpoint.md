---
type: batch-doc
module: 24-Multimodal
batch: "24"
doc_type: checkpoint
title: "Multimodal 验收清单"
tags:
 - sglang/batch/24
 - sglang/module/multimodal
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# Multimodal 验收清单

## 读者自测（不打开 sglang/）

- [ ] 能说明 Processor 注册与 get_mm_processor 选型流程
- [ ] 能追踪 image 从 Client 到 ViT forward 的数据流
- [ ] 能说出 BaseMultimodalProcessor、MultimodalSpecialTokens、PROCESSOR_MAPPING 的作用
- [ ] 能解释 placeholder 展开与 grid_thw 对齐的重要性
- [ ] 五篇正文 ≥ 15 段内嵌源码

## 核心结论

1. 每个 VLM 架构对应一个 `BaseMultimodalProcessor` 子类，经 `import_processors` 注册到 `PROCESSOR_MAPPING`。
2. Processor 负责媒体加载、resize、special token 展开，输出 `MultimodalProcessorOutput` 供 Scheduler prefill。
3. CUDA IPC 与 ViT CUDA Graph 是跨进程与性能优化可选层，不影响核心语义路径。

## 遗留问题

- 各 40+ Processor 文件仅 Qwen-VL 作代表走读；接入新模型时对照同系列 processor 即可。

## Wave-3 升级（2026-07-02）

- [x] `24-Multimodal-01-核心概念.md` §1 扩展用户故事「用户上传一张图问这是什么」
- [x] 新增 §5 设计追问：`keep_mm_feature_on_device` 取舍、ZMQ vs CUDA IPC 通道对比
- [x] 各节 Explain 扩至 ≥2 句；§6 CUDA IPC 自原 §5 顺延
- [x] 实码引用 server_args.py、base_processor.py L473-L482
