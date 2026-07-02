---
type: batch-doc
module: 24-Multimodal
batch: "24"
doc_type: faq
title: "多模态 VLM（Multimodal） · 关键问题"
tags:
  - sglang/batch/24
  - sglang/module/multimodal
  - sglang/doc/faq
aliases:
  - "04-关键问题"
updated: 2026-07-02
---
# 多模态 VLM（Multimodal） · 关键问题

## Q1：如何为新 VLM 添加 Processor？

**Explain：** 新建 `processors/my_vl.py`，继承 `BaseMultimodalProcessor`，声明 `models = [MyVLForConditionalGeneration]`，实现 `process()`。启动时自动 `import_processors` 注册。

**Code：**

```python
# 来源：python/sglang/srt/managers/multimodal_processor.py L34-L41
                assert hasattr(cls, "models")
                for arch in getattr(cls, "models"):
                    if overwrite:
                        for model_cls, processor_cls in PROCESSOR_MAPPING.items():
                            if model_cls.__name__ == arch.__name__:
                                del PROCESSOR_MAPPING[model_cls]
                                break
                    PROCESSOR_MAPPING[arch] = cls
```

**Comment：** 确保 model class 已在 `srt/models/` 注册且 `architectures` 名称一致。

---

## Q2：image token 数量不对导致错位？

**Explain：** placeholder 展开数量必须与实际 ViT output token 数一致；Qwen 用 smart_resize + grid 计算。

**Code（正确流程）：**

```python
# 来源：python/sglang/srt/multimodal/processors/qwen_vl.py L41-L44
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()
MAX_RATIO = 200
```

**Code（反模式）：**

```python
# 错误：固定写死 256 个 image token，与真实 grid 不符
input_text = input_text.replace("<image>", "<|image_pad|>" * 256)
```

**Comment：** 症状为乱码或重复短语；检查 Processor 输出的 `input_ids` 长度与 vision forward 的 seq len。

---

## Q3：何时用 TransformersAutoMultimodalProcessor？

**Explain：** `model_impl=transformers` 且无原生 Processor，或 Processor 标记 `supports_transformers_backend=True`。

**Code：**

```python
# 来源：python/sglang/srt/managers/multimodal_processor.py L64-L66
        if not uses_transformers_backend or getattr(
            processor_cls, "supports_transformers_backend", False
        ):
```

**Comment：** 原生 SGLang 模型实现 + 专用 Processor 性能更优。

---

## Q4：多图 / 多视频顺序

**Explain：** `organize_results()` 按 images → videos → audios 顺序；placeholder 在文本中的出现顺序必须与此一致。

**Code：**

```python
# 来源：python/sglang/srt/multimodal/processors/base_processor.py L72-L81
    def organize_results(self) -> List[Tuple[Modality, Any]]:
        """

        :return: a list of results, with their corresponding modalities
        """
        return (
            [(Modality.IMAGE, data) for data in self.images]
            + [(Modality.VIDEO, data) for data in self.videos]
            + [(Modality.AUDIO, data) for data in self.audios]
        )
```

**Comment：** 交错模态的 prompt 需在 process 内按文本顺序重排 mm_items。

---

## Q5：ViT CUDA Graph 不生效？

**Explain：** Graph 要求固定 shape；动态分辨率 batch 可能回退 eager。检查 `vit_cuda_graph_runner` capture 条件与 batch padding。

**Code：**

```python
# 来源：python/sglang/srt/multimodal/vit_cuda_graph_runner.py L115-L117, L246-L249, L371-L379
    def _get_graph_key(self, x_3d: torch.Tensor) -> int:
        # x_3d: [S, B, H], B=1, S as graph_key
        return x_3d.shape[0]
```

**Comment：** 与 LLM decode cuda graph 独立配置。

---

## Q6：AMX CPU 预处理

**Explain：** Qwen2-VL 在支持 AMX 的 CPU 上可 hack HF ImageProcessor 走 fast_preprocess_cpu。

**Code：**

```python
# 来源：python/sglang/srt/multimodal/processors/qwen_vl.py L65-L72
if _is_cpu and _is_cpu_amx_available:
    try:
        import transformers

        from sglang.srt.layers.amx_utils import fast_preprocess_cpu

        transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast._preprocess = (
            fast_preprocess_cpu
```

**Comment：** 仅 CPU 推理路径；GPU 部署可忽略。
