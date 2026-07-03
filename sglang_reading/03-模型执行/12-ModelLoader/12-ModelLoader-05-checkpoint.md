---
type: batch-doc
module: 12-ModelLoader
batch: "12"
doc_type: checkpoint
title: "ModelLoader 验收清单"
tags:
 - sglang/batch/12
 - sglang/module/model-loader
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# ModelLoader 验收清单

## 读者自测

- [ ] 能说明 BaseModelLoader 与 DefaultModelLoader 职责
- [ ] 能列举 3 种 LoadFormat 及场景
- [ ] 能描述 safetensors iterator → load_weights 数据流
- [ ] 能解释 FlattenedTensorBucket 在热更新中的作用
- [ ] 五篇正文 ≥ 15 段内嵌源码

## 核心结论

1. **ModelLoader** 按 `LoadFormat` 策略模式选择实现，核心是 **weight iterator + model.load_weights**。
2. **TP 分片**在模型层 `weight_loader` 完成，Loader 负责 IO 与量化 config 解析。
3. **weight_sync** 用打平 bucket 高效跨进程传 LoRA/权重 tensor。

## 代码块统计

| 文件 | 代码块 | 约行数 |
|------|--------|--------|
| README | 1 | 18 |
| 01 | 5 | 90 |
| 启动链路 | 10 | 145 |
| HTTP Server | 4 | 65 |
| OpenAI API | 6 | 55 |
| **合计** | **26** | **~373** |
