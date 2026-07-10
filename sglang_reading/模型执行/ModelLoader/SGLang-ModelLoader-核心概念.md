---
title: "ModelLoader · 核心概念"
type: concept
framework: sglang
topic: "ModelLoader"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-10
---
# ModelLoader · 核心概念

ModelLoader 的核心不是“支持多少文件格式”，而是把权重来源、模型结构、TP rank 和参数写入协议对齐。它要在冷启动时完成一件事：让当前进程的模型参数槽拿到自己该持有的 tensor 分片。

## 读者任务

读本篇是为了建立四个判断：

1. `LoadConfig` 是加载事实表，不是普通参数袋。
2. Loader 负责找文件和产生 weight iterator，但不负责理解每个模型的全部名字规则。
3. 模型类的 `load_weights` 负责把 checkpoint name 映射到参数名。
4. TP 切片由 layer parameter 的 `weight_loader` 完成。

## 三层模型

| 层 | 源码对象 | 问题 |
|----|----------|------|
| 格式层 | `LoadFormat` / `LoadConfig` / `get_model_loader` | 应该用哪个 loader，权重从哪里来 |
| 文件层 | `_prepare_weights` / `safetensors_weights_iterator` | 应该读哪些 shard，如何读，是否 mmap/prefetch |
| 参数层 | `model.load_weights` / `param.weight_loader` | 这个 tensor 写进哪个参数槽，是否需要 TP narrow |

这三层不要混在一起。很多加载 bug 的根因就是把“找到了文件”误认为“本 rank 已经拿到正确分片”。

## LoadFormat 是 loader 选择的入口

`LoadFormat` 把普通 HF、dummy、GGUF、remote、fastsafetensors、bitsandbytes 等路径放到同一个枚举里：

```python
# 来源：python/sglang/srt/configs/load_config.py L17-L37
class LoadFormat(str, enum.Enum):
    AUTO = "auto"
    PT = "pt"
    SAFETENSORS = "safetensors"
    NPCACHE = "npcache"
    DUMMY = "dummy"
    SHARDED_STATE = "sharded_state"
    GGUF = "gguf"
    BITSANDBYTES = "bitsandbytes"
    MISTRAL = "mistral"
    LAYERED = "layered"
    FLASH_RL = "flash_rl"  # For RL training with quantized models
    JAX = "jax"
    REMOTE = "remote"
    REMOTE_INSTANCE = "remote_instance"
    RDMA = "rdma"
    LOCAL_CACHED = "local_cached"
    FASTSAFETENSORS = "fastsafetensors"
    PRIVATE = "private"
    RUNAI_STREAMER = "runai_streamer"
```

`LoadConfig` 不是只存 `load_format`。它还记录 TP rank、remote instance 信息、ModelOpt 配置、RL quant profile 和 draft model index。

```python
# 来源：python/sglang/srt/configs/load_config.py L39-L56
@dataclass
class LoadConfig:
    """
    download_dir: Directory to download and load the weights, default to the
        default cache directory of huggingface.
    load_format: The format of the model weights to load:
        "auto" will try to load the weights in the safetensors format and
            fall back to the pytorch bin format if safetensors format is
            not available.
        "pt" will load the weights in the pytorch bin format.
        "safetensors" will load the weights in the safetensors format.
        "npcache" will load the weights in pytorch format and store
            a numpy cache to speed up the loading.
        "dummy" will initialize the weights with random values, which is
            mainly for profiling.
        "bitsandbytes" will load nf4 type weights.
        "flash_rl" will load weights with support for RL training
            with quantized models, enabling efficient weight reloading.
```

`__post_init__` 还会把 string 转成 enum，并给下载忽略规则一个默认值：

```python
# 来源：python/sglang/srt/configs/load_config.py L107-L135
    def __post_init__(self):
        model_loader_extra_config = self.model_loader_extra_config or {}
        if isinstance(model_loader_extra_config, str):
            self.model_loader_extra_config = orjson.loads(model_loader_extra_config)
        self._verify_load_format()

        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info(
                "Ignoring the following patterns when downloading weights: %s",
                self.ignore_patterns,
            )
        else:
            self.ignore_patterns = ["original/**/*"]

        # Create ModelOptConfig if not provided
        if self.modelopt_config is None:
            self.modelopt_config = ModelOptConfig(
                checkpoint_restore_path=self.modelopt_checkpoint_restore_path,
                checkpoint_save_path=self.modelopt_checkpoint_save_path,
                export_path=self.modelopt_export_path,
            )

    def _verify_load_format(self) -> None:
        if not isinstance(self.load_format, str):
            return

        load_format = self.load_format.lower()
        self.load_format = LoadFormat(load_format)
```

## Loader 的共同接口很窄

所有 loader 只承诺两件事：下载准备、返回已装载的 `nn.Module`。

```python
# 来源：python/sglang/srt/model_loader/loader.py L330-L349
class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        """Load a model with the given configurations."""
        raise NotImplementedError
```

这解释了为什么 `DefaultModelLoader`、`GGUFModelLoader`、`RemoteInstanceModelLoader`、`DummyModelLoader` 可以被同一个 `ModelRunner` 调用：它们隐藏了权重来源差异，但交付物都必须是可执行的 `nn.Module`。

## 默认 loader 的 Source 只是权重来源描述

`DefaultModelLoader.Source` 记录模型路径、revision、prefix、是否允许 `.pt` fallback。它不写参数，也不做 TP 切片。

```python
# 来源：python/sglang/srt/model_loader/loader.py L352-L403
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

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"enable_multithread_load", "num_threads"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )
```

这里的 `prefix` 是多来源权重的命名补偿，比如 secondary weights；不要把它理解成文件系统路径前缀。

## Iterator 是内存边界

默认路径通过 iterator 逐个吐出 `(name, tensor)`。safetensors 路径使用 `safe_open`，在每个文件内按 key 取 tensor。

```python
# 来源：python/sglang/srt/model_loader/weight_utils.py L930-L964
def safetensors_weights_iterator(
    hf_weights_files: List[str],
    disable_mmap: bool = False,
    prefetch: bool = False,
    prefetch_num_threads: int = 4,
    drop_cache_after_load: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    if prefetch and not disable_mmap:
        _prefetch_all_checkpoints(
            sorted(hf_weights_files), num_threads=prefetch_num_threads
        )

    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors checkpoint shards",
        disable=not enable_tqdm,
        bar_format=BAR_FORMAT,
        position=tqdm._get_free_pos(),
    ):
        if disable_mmap:
            with open(st_file, "rb") as f:
                result = safetensors.torch.load(f.read())
                for name in sorted(result.keys()):
                    yield name, result[name]
        else:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)
        if drop_cache_after_load:
            _drop_file_cache_after_load(st_file)
```

这就是为什么 `enable_multithread_load` 和 mmap/prefetch 会影响启动速度与 CPU 内存，但不会改变模型参数映射语义。

## TP 切片在参数写入时发生

以 `RowParallelLinear.weight_loader` 为例，loader 传入的是 checkpoint tensor；参数自己的 `weight_loader` 根据 `input_dim`、`tp_rank`、`shard_size` 做 narrow，再 copy 到本 rank 参数。

```python
# 来源：python/sglang/srt/layers/linear.py L1426-L1487
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if input_dim:
                weight_shape[input_dim] = weight_shape[input_dim] // self.tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

        param_data = param.data
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        if (
            input_dim is not None
            and not use_bitsandbytes_4bit
            and not self.use_presharded_weights
        ):
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size

            if _is_cpu:
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    param_data,
                    loaded_weight,
                    0,  # param_data_start
                    start_idx,
                    input_dim,
                    shard_size,
                )
            else:
                # Padding for special case like qwen2_5_VL's mlp which is not 8-aligned
                end_idx = start_idx + shard_size
                if end_idx > loaded_weight.shape[input_dim]:
                    loaded_weight = pad_or_narrow_weight(
                        loaded_weight, input_dim, start_idx, shard_size
                    )
                else:
                    loaded_weight = loaded_weight.narrow(
                        input_dim, start_idx, shard_size
                    )

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert (
            param_data.shape == loaded_weight.shape
        ), f"{param_data.shape=} {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)
```

这个片段是理解 ModelLoader 的关键：如果你在 loader 文件里寻找 TP slice，方向就错了。文件层只提供 tensor，参数层才决定如何切。

## 复盘

ModelLoader 的正确读法是：

- `LoadConfig` 决定加载路线。
- loader 决定权重来源和 iterator。
- 模型类决定名字怎么映射。
- 参数对象决定本 rank 怎么 narrow。
- 量化后处理在 `model.load_weights` 之后补齐执行态参数。
