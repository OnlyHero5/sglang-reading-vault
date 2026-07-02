---
type: batch-doc
module: 24-Multimodal
batch: "24"
doc_type: walkthrough
title: "Multimodal · 源码走读"
tags:
 - sglang/batch/24
 - sglang/module/multimodal
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# Multimodal · 源码走读

## 走读顺序

1. `multimodal_processor.py` — 注册与工厂
2. `base_processor.py` — 基类与输出结构
3. `processors/qwen_vl.py` — 代表实现
4. `vit_cuda_graph_runner.py` — ViT 图捕获
5. `mm_utils.py` — 通用工具

---

## 1. BaseMultimodalProcessor 抽象

**Explain：** 子类实现 `process()`：接收原始 messages / 媒体路径，返回 `MultimodalProcessorOutput`（含 input_ids、mm_items）。

**Code：**

```python
# 来源：python/sglang/srt/multimodal/processors/base_processor.py L1-L9
import asyncio
import concurrent
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
```

**Comment：** 基类还处理 thread pool 并行 load_image/load_video，避免阻塞 event loop。

---

## 2. Qwen-VL Processor 模型绑定

**Explain：** `qwen_vl.py` 声明支持的 model class 列表，并定义 IMAGE_FACTOR、smart_resize 等 Qwen 特有预处理。

**Code：**

```python
# 来源：python/sglang/srt/multimodal/processors/qwen_vl.py L41-L44
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()
MAX_RATIO = 200
```

**Comment：** 像素对齐 28 倍数，与 Qwen2-VL patch size 一致；环境变量可 cap 最大像素防 OOM。

---

## 3. smart_resize

**Explain：** 按 min/max pixels 与 aspect ratio 约束计算目标高宽，保持总 token 数在预算内。

**Code：**

```python
# 来源：python/sglang/srt/multimodal/processors/qwen_vl.py L80-L95（函数首部示意）
def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
```

**Comment：** 视频路径使用独立的 VIDEO_MIN/MAX_PIXELS 与 FPS 抽帧逻辑。

---

## 4. MRotaryEmbedding 与 multimodal position

**Explain：** Qwen-VL 使用 multimodal rotary（mrope），Processor 需输出 `grid_thw` 供 position id 计算。

**Code：**

```python
# 来源：python/sglang/srt/multimodal/processors/qwen_vl.py L14-L14
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
```

**Comment：** 与模型 forward 中 `get_rope_index` 配合；错误 grid 会导致 vision-text 对齐错乱。

---

## 5. Transformers Auto 回退

**Explain：** 当 `model_impl=transformers` 且无专用 Processor 时，使用 HuggingFace processor 自动路径。

**Code：**

```python
# 来源：python/sglang/srt/managers/multimodal_processor.py L71-L78
    if uses_transformers_backend:
        from sglang.srt.multimodal.processors.transformers_auto import (
            TransformersAutoMultimodalProcessor,
        )

        return TransformersAutoMultimodalProcessor(
            hf_config, server_args, processor, transport_mode, **kwargs
        )
```

**Comment：** 性能通常低于原生 SGLang Processor；适合快速兼容新 HF 模型。

---

## 6. ViT CUDA Graph Runner

**Explain：** 固定 batch 形状的 ViT forward 可 capture CUDA Graph，降低 small batch launch overhead。

**Code：**

```python
# 来源：python/sglang/srt/multimodal/vit_cuda_graph_runner.py L357-L388
    def run(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        output_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [seq_len, hidden] -> [S, B=1, H]
        x_3d = x.unsqueeze(1)
        graph_key = self._get_graph_key(x_3d)

        if graph_key not in self.block_graphs:
            self.create_graph(
                x_3d=x_3d,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                cu_window_seqlens=cu_window_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )

        return self.replay(
            graph_key=graph_key,
            x_3d=x_3d,
            position_embeddings=position_embeddings,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            output_indices=output_indices,
        )
```

**Comment：** InternVL 等有专用 `internvl_vit_cuda_graph_runner.py` 变体。

---

## 7. EVS 模块

**Explain：** `multimodal/evs/` 实现 Efficient Video Streaming 相关 processor，用于长视频流式编码。

**Code：**

```python
# 来源：python/sglang/srt/multimodal/evs/evs_core.py L74-L97
    # Core EVS
    similarity = torch.nn.functional.cosine_similarity(
        video_embeds[1:, ...], video_embeds[:-1, ...], dim=-1
    )
    dissimilarity = 1 - similarity

    # Always ensure we include all tokens from the first frame
    dissimilarity = torch.cat(
        [255 * torch.ones_like(video_embeds[:1, :, :, 0]), dissimilarity], dim=0
    )

    dissimilarity_flat = dissimilarity.view(-1)
    order = torch.argsort(dissimilarity_flat, dim=-1, descending=True, stable=True)
    retain_num_tokens = compute_retained_tokens_count(
        tokens_per_frame=tokens_per_frame, num_frames=T, q=q
    )
    topk_indices = order[:retain_num_tokens]

    retention_mask = torch.zeros_like(dissimilarity_flat, dtype=torch.bool)
    retention_mask[topk_indices] = True
    retention_mask = retention_mask.reshape(dissimilarity.size())

    mask = retention_mask.view(-1)  # "T H W -> (T H W)"
    return mask
```

**Comment：** 见 evs/README.md；与 batch 29 multimodal_gen 扩散路径不同（本模块为 VLM 判别式）。

---

## 8. load_image / load_video 工具

**Explain：** `srt.utils.load_image` 等统一处理 URL、base64、本地路径；Processor 通过基类调用。

**Code：**

```python
# 来源：python/sglang/srt/multimodal/processors/base_processor.py L23-L31
from sglang.srt.utils import (
    envs,
    is_cpu,
    is_npu,
    is_xpu,
    load_audio,
    load_image,
    load_video,
    logger,
```

**Comment：** 支持多帧视频 decode；`VideoDecoderWrapper` 在 qwen_vl 中用于硬件加速解码。

---

## 9. 未注册架构错误

**Explain：** architectures 不在 mapping 且非 transformers backend 时抛 ValueError，列出已注册架构名。

**Code：**

```python
# 来源：python/sglang/srt/managers/multimodal_processor.py L80-L83
    raise ValueError(
        f"No processor registered for architecture: {hf_config.architectures}.\n"
        f"Registered architectures: {[model_cls.__name__ for model_cls in PROCESSOR_MAPPING.keys()]}"
    )
```

**Comment：** 部署新 VLM 时此错误是首要排查点。

---

## 10. import_processors 完整循环

**Explain：** 启动时扫描 `sglang.srt.multimodal.processors` 包下所有模块，将 `BaseMultimodalProcessor` 子类注册到 `PROCESSOR_MAPPING`。

**Code：**

```python
# 来源：python/sglang/srt/managers/multimodal_processor.py L16-L41
def import_processors(package_name: str, overwrite: bool = False):
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue
            all_members = inspect.getmembers(module, inspect.isclass)
            classes = [
                member
                for name, member in all_members
                if member.__module__ == module.__name__
            ]
            for cls in (
                cls for cls in classes if issubclass(cls, BaseMultimodalProcessor)
            ):
                assert hasattr(cls, "models")
                for arch in getattr(cls, "models"):
                    if overwrite:
                        for model_cls, processor_cls in PROCESSOR_MAPPING.items():
                            if model_cls.__name__ == arch.__name__:
                                del PROCESSOR_MAPPING[model_cls]
                                break
                    PROCESSOR_MAPPING[arch] = cls
```

**Comment：** `overwrite=True` 时可热替换已注册 Processor；import 失败仅 warning 不阻断启动。

---

## 11. Transformers backend 回退

**Code：**

```python
# 来源：python/sglang/srt/managers/multimodal_processor.py L71-L78
    if uses_transformers_backend:
        from sglang.srt.multimodal.processors.transformers_auto import (
            TransformersAutoMultimodalProcessor,
        )

        return TransformersAutoMultimodalProcessor(
            hf_config, server_args, processor, transport_mode, **kwargs
        )
```

**Comment：** `model_impl=auto` 时通过 `get_resolved_model_impl` 判断是否走 Transformers 路径。

