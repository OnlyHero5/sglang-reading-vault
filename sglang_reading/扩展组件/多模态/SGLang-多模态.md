---
title: "多模态"
type: map
framework: sglang
topic: "多模态"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-10
---
# 多模态

> SGLang 扩展组件 · 多模态 | Git：`70df09b`

## 1. 本模块在全局架构中的位置

专题读法：多模态推理位于 TokenizerManager 与 Scheduler 之间：Processor 将 OpenAI messages 中的 image/video/audio 转为 `input_ids` placeholder 与 `MultimodalDataItem` feature tensor。ViT/Audio Tower 在 Scheduler prefill 阶段 forward，embedding 与 text token 拼接后进入 LLM 主干。本模块与 multimodal_gen（扩散生成）不同，聚焦 VLM 对话推理路径。

```
Client (messages + media)
 → TokenizerManager.process_multimodal
 → MultimodalProcessor (qwen_vl / llava / ...)
 → Scheduler (ViT CUDA Graph / IPC transport)
 → LLM RadixAttention + KV Cache
```

---

## 2. 本模块目标

1. 图像/视频/音频如何经 Processor 转为 `MultimodalDataItem` 并嵌入 input_ids？
2. `get_mm_processor` 如何按模型架构选择 Qwen-VL、LLaVA 等 Processor？
3. CUDA IPC 与 ViT CUDA Graph 在 multimodal 路径中的作用？

## 文档导航

| 文件 | 内容 |
|------|------|
| [[SGLang-多模态-核心概念]] | Modality、Processor 注册、Special Tokens |
| [[SGLang-多模态-源码走读]] | base_processor、qwen_vl、multimodal_processor |
| [[SGLang-多模态-数据流]] | TokenizerManager → Scheduler → ViT |
| [[SGLang-多模态-排障指南]] | 新模型接入、IPC、PD+Encoder |
| [[SGLang-多模态-学习检查]] | 验收清单 |

## 源码范围

`srt/multimodal/`（processors/、mm_utils.py、vit_cuda_graph_runner.py）、`srt/managers/multimodal_processor.py`。

## 最关键的一段入口代码

注册读法：启动时 `import_processors` 扫描 processors 包，将 `BaseMultimodalProcessor` 子类注册到 `PROCESSOR_MAPPING`；运行时按 `hf_config.architectures` 选取。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/multimodal_processor.py L44-L68
def get_mm_processor(
    hf_config,
    server_args: ServerArgs,
    processor,
    transport_mode,
    model_config=None,
    **kwargs,
) -> BaseMultimodalProcessor:
    model_impl = str(getattr(server_args, "model_impl", "auto")).lower()
    uses_transformers_backend = model_impl == "transformers"
    if model_impl == "auto" and model_config is not None:
        from sglang.srt.model_loader.utils import get_resolved_model_impl

        uses_transformers_backend = (
            get_resolved_model_impl(model_config) == ModelImpl.TRANSFORMERS
        )

    for model_cls, processor_cls in PROCESSOR_MAPPING.items():
        if model_cls.__name__ not in hf_config.architectures:
            continue
        if not uses_transformers_backend or getattr(
            processor_cls, "supports_transformers_backend", False
        ):
            return processor_cls(
                hf_config, server_args, processor, transport_mode, **kwargs
```

读法：

- 每个 Processor 类声明 `models = [Qwen2VLForConditionalGeneration, ...]`。
- Transformers backend 回退到 `TransformersAutoMultimodalProcessor`。
- `transport_mode` 控制 CUDA IPC 等跨进程 tensor 传递。

## 下一模块

→ [[SGLang-LoRA|LoRA]]
