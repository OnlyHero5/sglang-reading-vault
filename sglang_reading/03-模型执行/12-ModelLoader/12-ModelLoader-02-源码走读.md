---
type: batch-doc
module: 12-ModelLoader
batch: "12"
doc_type: walkthrough
title: "ModelLoader · 源码走读"
tags:
 - sglang/batch/12
 - sglang/module/model-loader
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# ModelLoader · 源码走读

## 走读顺序

1. `loader.py` — BaseModelLoader / DefaultModelLoader
2. `weight_utils.py` — 文件下载与 iterator
3. `weight_sync/tensor_bucket.py` — 跨进程 tensor 打包

---

## 1. DefaultModelLoader.load_model

**Explain：** 典型路径：解析架构 → `init_empty_weights` 或 CPU 上建模型 → iterator 灌权重 → 移到 target device。

**Code：**

```python
# 来源：python/sglang/srt/model_loader/loader.py L352-L387
# 提交版本：70df09b
class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    _MTP_PATTERN = re.compile(r"model\.mtp\.layers\.(\d+)\.")

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: Optional[str]
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        model_config: Optional[ModelConfig] = None
        """The model configuration (for checking architecture, etc)."""

        @classmethod
        def init_new(cls, model_config: ModelConfig, model):
            return cls(
                model_config.model_path,
                model_config.revision,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                model_config=model_config,
            )
```

**Comment：** `Source` 封装 HF path/revision；`prefix` 用于 MTP 等附加权重前缀映射。

---

## 2. _prepare_weights

**Explain：** 列出 safetensors/bin 文件、过滤 duplicate、可选 ModelScope 下载。

**Code：**

```python
# 来源：python/sglang/srt/model_loader/loader.py L405-L429
# 提交版本：70df09b
    def _maybe_download_from_modelscope(
        self, model: str, revision: Optional[str]
    ) -> str:
        """Download model from ModelScope hub if SGLANG_USE_MODELSCOPE is True.

        Returns the path to the downloaded model, or the original model path if
        not downloaded from ModelScope."""
        if get_bool_env_var("SGLANG_USE_MODELSCOPE"):
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            # pylint: disable=C.
            from modelscope.hub.snapshot_download import snapshot_download

            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=self.load_config.download_dir,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    revision=revision,
                    ignore_file_pattern=self.load_config.ignore_patterns,
                )
            else:
                model_path = model
            return model_path
        return model
```

---

## 3. 模型侧 load_weights

**Explain：** 每个模型类实现 `load_weights(self, weights: Iterable[Tuple[str, Tensor]])`，内部调 `default_weight_loader` 处理 TP slice。

**Code（Llama 示例，Models 通用 同源）：**

```python
# 来源：python/sglang/srt/models/llama.py L680-L700
# 提交版本：70df09b（节选）
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip loading kv_scale from ckpts towards new design.
                if name.endswith(".kv_scale") and name not in params_dict:
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")
```

---

## 4. download_weights_from_hf

**Code：**

```python
# 来源：python/sglang/srt/model_loader/weight_utils.py L517-L540
# 提交版本：70df09b（节选）
def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: Optional[str],
    allow_patterns: List[str],
    revision: Optional[str] = None,
    ignore_patterns: Optional[Union[str, List[str]]] = None,
    max_retries: int = 3,
) -> str:
    """Download model weights from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (List[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        ignore_patterns (Optional[Union[str, List[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.
        max_retries (int): Maximum number of download retries if corruption
            is detected. Defaults to 3.

```

---

## 5. get_quant_config

**Explain：** 从 HF config 或 quantize_config.json 解析 AWQ/GPTQ/FP8 等量化配置。

**Code：**

```python
# 来源：python/sglang/srt/model_loader/weight_utils.py L237-L260
# 提交版本：70df09b（节选）
def get_quant_config(
    model_config: ModelConfig,
    load_config: LoadConfig,
    packed_modules_mapping: Dict[str, List[str]],
    remap_prefix: Dict[str, str] | None = None,
) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)

    # GGUF doesn't have config file
    if model_config.quantization == "gguf":
        return quant_cls.from_config({})

    # Read the quantization config from the HF model config, if available.
    hf_quant_config = getattr(model_config.hf_config, "quantization_config", None)
    # some vision model may keep quantization_config in their text_config
    hf_text_config = getattr(model_config.hf_config, "text_config", None)
    if hf_quant_config is None and hf_text_config is not None:
        hf_quant_config = getattr(hf_text_config, "quantization_config", None)
    if hf_quant_config is None:
        # compressed-tensors uses a compressions_config
        hf_quant_config = getattr(model_config.hf_config, "compression_config", None)
    if hf_quant_config is not None:
        if not isinstance(hf_quant_config, dict):
            hf_quant_config = hf_quant_config.to_dict()
```

---

## 6. RemoteInstanceModelLoader

**Explain：** PD 分离或集群扩容时，从已运行实例 NCCL/RDMA 拉全量权重。

**Code：**

```python
# 来源：python/sglang/srt/model_loader/loader.py L2194-L2210
# 提交版本：70df09b（节选）
class RemoteInstanceModelLoader(BaseModelLoader):
    """Model loader that can load Tensors from remote sglang instance."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )
        self.remote_instance_transfer_engine_weight_info = None

    def download_model(self, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def load_model(
        self,
```

---

## 7. GGUFModelLoader

**Code：**

```python
# 来源：python/sglang/srt/model_loader/loader.py L2086-L2100
# 提交版本：70df09b（节选）
class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

```

**Comment：** 使用 `gguf_quant_weights_iterator` 逐 tensor 解码。

---

## 8. FlattenedTensorBucket.reconstruct_tensors

**Code：**

```python
# 来源：python/sglang/srt/weight_sync/tensor_bucket.py L90-L107
# 提交版本：70df09b
    def reconstruct_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Reconstruct original tensors from flattened tensor with optimized performance.
        Uses memory-efficient operations to minimize allocations and copies.
        """
        # preallocate the result list
        reconstructed = [None] * len(self.metadata)

        for i, meta in enumerate(self.metadata):
            tensor = (
                self.flattened_tensor[meta.start_idx : meta.end_idx]
                .view(meta.dtype)
                .reshape(meta.shape)
            )

            reconstructed[i] = (meta.name, tensor)

        return reconstructed
```

---

## 9. LayeredModelLoader

**Explain：** 按层加载并立即 offload 未用层，适合超大模型单卡加载。

**Code：**

```python
# 来源：python/sglang/srt/model_loader/loader.py L824-L835
# 提交版本：70df09b（节选）
class LayeredModelLoader(DefaultModelLoader):
    """Model loader that loads weights layer by layer so that one can quantize a
    layer before loading another to make the peak memory envelope smaller."""

    def __init__(self, load_config: LoadConfig):
        # Back to the default load format
        load_config.load_format = LoadFormat.AUTO
        super().__init__(load_config)

    def load_model(
        self,
        *,
```

---

## 10. DummyModelLoader

**Code：**

```python
# 来源：python/sglang/srt/model_loader/loader.py L1371-L1390
# 提交版本：70df09b（节选）
class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
```

**Comment：** 配合 `--load-format dummy` 做调度压测，不读真实 checkpoint。
