---
title: "多模态 · 排障指南"
type: troubleshooting
framework: sglang
topic: "多模态"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# 多模态 · 排障指南

## 你为什么要读

多模态报错经常离根因很远：模型 forward 报 shape mismatch，问题可能早在 processor 选择、placeholder 展开或 `grid_thw` 对齐时就出现。本文按“架构注册、媒体处理、token 对齐、特征搬运、模型执行”逐层缩小范围。

## Q1：如何为新 VLM 添加 Processor？

**读法：** 新建 `processors/my_vl.py`，继承 `BaseMultimodalProcessor`，声明 `models = [MyVLForConditionalGeneration]`，实现 `process()`。启动时自动 `import_processors` 注册。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/multimodal_processor.py L34-L41
                assert hasattr(cls, "models")
                for arch in getattr(cls, "models"):
                    if overwrite:
                        for model_cls, processor_cls in PROCESSOR_MAPPING.items():
                            if model_cls.__name__ == arch.__name__:
                                del PROCESSOR_MAPPING[model_cls]
                                break
                    PROCESSOR_MAPPING[arch] = cls
```

**要点：** 确保 model class 已在 `srt/models/` 注册且 `architectures` 名称一致。

---

## Q2：image token 数量不对导致错位？

**读法：** placeholder 展开数量必须与实际 ViT output token 数一致；Qwen 用 smart_resize + grid 计算。

**源码锚点（正确流程）：**

```python
## 来源：python/sglang/srt/multimodal/processors/qwen_vl.py L41-L44
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()
MAX_RATIO = 200
```

**源码锚点（反模式）：**

```python
# 错误：固定写死 256 个 image token，与真实 grid 不符
input_text = input_text.replace("<image>", "<|image_pad|>" * 256)
```

**要点：** 症状为乱码或重复短语；检查 Processor 输出的 `input_ids` 长度与 vision forward 的 seq len。

---

## Q3：何时用 TransformersAutoMultimodalProcessor？

**读法：** `model_impl=transformers` 且无原生 Processor，或 Processor 标记 `supports_transformers_backend=True`。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/multimodal_processor.py L64-L66
        if not uses_transformers_backend or getattr(
            processor_cls, "supports_transformers_backend", False
        ):
```

**要点：** 原生 SGLang 模型实现 + 专用 Processor 性能更优。

---

## Q4：多图 / 多视频顺序

**读法：** `organize_results()` 按 images → videos → audios 顺序；placeholder 在文本中的出现顺序必须与此一致。

**源码锚点：**

```python
## 来源：python/sglang/srt/multimodal/processors/base_processor.py L72-L81
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

**要点：** 交错模态的 prompt 需在 process 内按文本顺序重排 mm_items。

---

## Q5：ViT CUDA Graph 不生效？

**读法：** Graph 要求固定 shape；动态分辨率 batch 可能回退 eager。检查 `vit_cuda_graph_runner` capture 条件与 batch padding。

**源码锚点：**

```python
## 来源：python/sglang/srt/multimodal/vit_cuda_graph_runner.py L115-L117, L246-L249, L371-L379
    def _get_graph_key(self, x_3d: torch.Tensor) -> int:
        # x_3d: [S, B, H], B=1, S as graph_key
        return x_3d.shape[0]
```

**要点：** 与 LLM decode cuda graph 独立配置。

---

## Q6：AMX CPU 预处理

**读法：** Qwen2-VL 在支持 AMX 的 CPU 上可 hack HF ImageProcessor 走 fast_preprocess_cpu。

**源码锚点：**

```python
## 来源：python/sglang/srt/multimodal/processors/qwen_vl.py L65-L72
if _is_cpu and _is_cpu_amx_available:
    try:
        import transformers

        from sglang.srt.layers.amx_utils import fast_preprocess_cpu

        transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast._preprocess = (
            fast_preprocess_cpu
```

**要点：** 仅 CPU 推理路径；GPU 部署可忽略。

---

## 验证抓手

FAQ 的排查顺序可以用一条静态命令先校准。它不验证模型输出质量，但能确认六个问题分别落在正确源码边界。

```powershell
rg -n "PROCESSOR_MAPPING|supports_transformers_backend|IMAGE_FACTOR|organize_results|_get_graph_key|fast_preprocess_cpu" `
  sglang/python/sglang/srt/managers/multimodal_processor.py `
  sglang/python/sglang/srt/multimodal/processors/base_processor.py `
  sglang/python/sglang/srt/multimodal/processors/qwen_vl.py `
  sglang/python/sglang/srt/multimodal/vit_cuda_graph_runner.py
```

预期现象：

- 新 VLM 注册问题应命中 `PROCESSOR_MAPPING`。
- Transformers fallback 问题应命中 `supports_transformers_backend`。
- image token 数量问题应先命中 Qwen 的 `IMAGE_FACTOR`，再回到 grid/token 展开路径。
- 多图多视频顺序问题应命中 `organize_results`。
- ViT CUDA Graph 问题应命中 `_get_graph_key`。
- AMX CPU 预处理问题应命中 `fast_preprocess_cpu`。

如果运行期症状与这些入口对不上，例如 token 错位却只改 HTTP 层，说明排查方向已经偏离：应先回到 processor 输出的 `input_ids`、`mm_items` 和视觉 grid 对齐关系。
