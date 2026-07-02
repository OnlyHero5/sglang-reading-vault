---
type: batch-doc
module: 19-Quantization
batch: "19"
doc_type: walkthrough
title: "Quantization · 源码走读"
tags:
 - sglang/batch/19
 - sglang/module/quantization
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# Quantization · 源码走读

> 走读顺序：`base_config.py` → `fp8.py/fp8_utils.py` → `gptq/` → `awq/` → `kv_cache.py` → `unquant.py`

---

## 1. base_config.py — 量化方法基类

### 1.1 QuantizeMethodBase 与 LinearMethodBase

**Explain：** 量化体系的核心抽象；`create_weights` 在 model init 时为 layer 注册 Parameter（quant weight、scale、zero point 等），`apply` 在前向时被 LinearBase/MoE 层调用。LinearMethodBase 扩展 create_weights 签名以支持 TP 分片（input_size_per_partition / output_partition_sizes）。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/base_config.py L20-L84
class QuantizeMethodBase(ABC):
    """Base class for different quantized methods."""

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        """Create weights for a layer.

        The weights will be set as attributes of the layer."""
        raise NotImplementedError()

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError()

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Process the weight after loading.

        This can be used for example, to transpose weights for computation.
        """
        return


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for a linear layer.
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            input_size: Size of the input dim of the weight across all ranks.
            output_size: Size of the output dim of the weight across all ranks.
            params_dtype: Datatype of the parameters.
        """
        raise NotImplementedError()

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError()

```

### 1.2 FusedMoEMethodBase

**Explain：** MoE 量化方法的基类；除 create_weights 外还提供 `create_moe_runner`，将量化信息（weight layout、scale dtype）绑定到 MoeRunnerConfig。EP 场景下每个 rank 只 create 本地 expert 的 quant weight。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/base_config.py L86-L100
class FusedMoEMethodBase(QuantizeMethodBase):

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
```

---

## 2. fp8_utils.py — GEMM backend dispatch

### 2.1 dispatch_w8a8_block_fp8_linear

**Explain：** FP8 linear 的核心路由函数；优先读 `--fp8-gemm-backend` 显式配置，auto 模式按 SM 版本、FlashInfer/DeepGEMM 可用性选择。Blackwell 可选 flashinfer_trtllm；SM90 可选 deepgemm；ROCm gfx95 可选 aiter。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/fp8_utils.py L394-L409
def dispatch_w8a8_block_fp8_linear() -> Callable:
    """
    Dispatch to the appropriate FP8 block linear implementation.

    This function selects the backend based on:
    1. The --fp8-gemm-backend server argument (preferred)
    2. Auto-detection based on hardware capabilities
    """
    backend = get_fp8_gemm_runner_backend()

    # Handle explicit backend selection via --fp8-gemm-backend
    if not backend.is_auto():
        return _dispatch_explicit_backend(backend)

    # Auto mode: Select based purely on hardware/backend availability
    return _dispatch_auto_backend()
```

### 2.2 _dispatch_explicit_backend

**Explain：** 显式 backend 选择时做硬件兼容性检查；不满足条件直接 RuntimeError 而非 silent fallback，避免性能不达预期。flashinfer_trtllm 要求 SM100+ 且 FlashInfer 可用。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/fp8_utils.py L419-L428
def _dispatch_explicit_backend(backend: Fp8GemmRunnerBackend) -> Callable:
    """Dispatch based on explicitly selected backend."""
    if backend.is_flashinfer_trtllm():
        if not (is_sm100_supported() and is_flashinfer_available()):
            raise RuntimeError(
                "FlashInfer FP8 GEMM requested via --fp8-gemm-backend=flashinfer_trtllm, "
                "but FlashInfer is not available or not supported on this hardware. "
                "FlashInfer TRTLLM FP8 GEMM requires SM100/SM103 GPUs and FlashInfer."
            )
        return flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
```

---

## 3. fp8_kernel.py — 激活量化 kernel

### 3.1 scaled_fp8_quant — dynamic / static 分支

**Explain：** FP8 linear 的 activation 侧入口。输入必须是 2D `[num_tokens, hidden]`。`scale=None` 时走 **dynamic**：per-tensor 或 per-token（`use_per_token_if_dynamic`）在 GPU 上算 scale 并 quantize；`scale` 为标量时走 **static**，直接除以 checkpoint 里的固定 scale。优先 vLLM `_C` op，其次 aiter，最后 native PyTorch fallback——避免无 vLLM 时 silent 错结果。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/fp8_kernel.py L1790-L1836
    def scaled_fp8_quant(
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        num_token_padding: Optional[int] = None,
        use_per_token_if_dynamic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert input.ndim == 2, f"Expected 2D input tensor, got {input.ndim}D"
        shape = input.shape
        if num_token_padding:
            shape = (max(num_token_padding, input.shape[0]), shape[1])
        output = torch.empty(shape, device=input.device, dtype=fp8_dtype)

        if scale is None:
            # Dynamic scaling
            if use_per_token_if_dynamic:
                scale = torch.empty(
                    (shape[0], 1), device=input.device, dtype=torch.float32
                )
                if _use_aiter:
                    dynamic_per_token_scaled_quant(output, input, scale)
                elif _has_vllm:
                    torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                        output, input.contiguous(), scale, None
                    )
                else:
                    _native_dynamic_per_token_quant_fp8(output, input, scale)
            else:
                scale = torch.zeros(1, device=input.device, dtype=torch.float32)
                if _use_aiter:
                    dynamic_per_tensor_quant(output, input, scale)
                elif _has_vllm:
                    torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
                else:
                    _native_dynamic_per_tensor_quant_fp8(output, input, scale)
        else:
            # Static scaling
            assert (
                scale.numel() == 1
            ), f"Expected scalar scale, got numel={scale.numel()}"
            if _use_aiter:
                static_per_tensor_quant(output, input, scale)
            elif _has_vllm:
                torch.ops._C.static_scaled_fp8_quant(output, input, scale)
            else:
                _native_static_quant_fp8(output, input, scale)

        return output, scale
```

**Comment：**

- 返回 `(fp8_output, scale)` 供后续 GEMM 消费；dynamic per-token scale shape 为 `[num_tokens, 1]`。
- `_native_*` fallback 用 `torch.clamp(input / scale, fp8_min, fp8_max)`，注释强调避免 `.item()` 造成 CPU-GPU sync。
- block quant 路径另走 `per_token_group_quant_fp8` + `w8a8_block_fp8_matmul_*`（见 §2 dispatch）。

---

## 4. gptq/schemes/gptq_linear.py — GPTQ Linear Scheme

### 4.1 GPTQLinearScheme

**Explain：** 标准 GPTQ 4bit 路径；`_init_kernel` 根据 quant_config 选择 GPTQLinearKernel（CUDA）或 Ascend 变体。create_weights 检查 input_size 与 group_size 对齐；apply 调用 kernel 解压 weight 并做 GEMM。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/gptq/schemes/gptq_linear.py L295-L320
class GPTQLinearScheme(GPTQLinearSchemeBase):
 def __init__(self, quant_config: GPTQConfig):
 self.quant_config = quant_config
 self.use_v2_format = quant_config.checkpoint_format == "gptq_v2"
 self.kernel = self._init_kernel(quant_config)

 def create_weights(self, layer, input_size_per_partition, output_partition_sizes, ...):
 if input_size_per_partition % self.quant_config.group_size != 0:
 raise ValueError("The input size is not aligned with the quantized group size")
```

---

## 5. awq/awq.py — AWQ 入口

### 5.1 AWQConfig 与 scheme 选择

**Explain：** AWQ 4bit weight-only 量化；加载时 check_marlin_supported 检测硬件是否支持 Marlin kernel，支持则选 AWQMarlinLinearScheme 否则 AWQLinearScheme。MoE 层独立 AWQMoEScheme，create_weights 含 num_experts 维度。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/awq/awq.py L346-L370
            return AWQLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return None
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config

            if not check_moe_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by AWQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                return MoeWNA16Config.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            layer.scheme = self.get_moe_scheme(layer)
            return AWQMoEMethod(self)
        return None

    def get_linear_scheme(self, layer: torch.nn.Module):
        return AWQMarlinLinearScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        return AWQMoEScheme(self)

    @classmethod
```

---

## 6. kv_cache.py — KV FP8 量化

### 6.1 BaseKVCacheMethod

**Explain：** KV 量化不通过 apply 做 GEMM，而是在 Attention backend 读写 KV cache 时使用 k_scale/v_scale 做 quantize/dequantize。初始 scale=-1.0 表示未加载；process_weights_after_loading 从 checkpoint 或 kv_cache_dtype 推导有效 scale。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/kv_cache.py L416-L447
class BaseKVCacheMethod(QuantizeMethodBase):
 def create_weights(self, layer: torch.nn.Module):
 layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32), requires_grad=False)
 layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32), requires_grad=False)
 def apply(self, layer: torch.nn.Module) -> torch.Tensor:
 raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")
 def process_weights_after_loading(self, layer) -> None:
 if layer.k_scale > 0.0 and layer.v_scale > 0.0:
 k_scale = layer.k_scale.to("cpu").tolist()
 if is_fp8_fnuz():
 k_scale *= 2
```

---

## 7. unquant.py — 无量化 fallback

### 7.1 UnquantizedLinearMethod

**Explain：** 默认 bf16/fp16 路径；apply 调用 `F.linear` 或 AMX/XPU 专用 backend。MoE 无量化时创建 TritonMoeQuantInfo + Triton MoeRunner，与量化 MoE 共享 FusedMoE.dispatch/combine 框架。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/unquant.py L488-L513
            # otherwise use_fp8=True for FP8 dispatch path
            use_fp8 = not envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
            quant_info = DeepGemmMoeQuantInfo(
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                use_fp8=use_fp8,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_cutlass:
            from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
                FlashInferCutlassMoeQuantInfo,
            )

            quant_info = FlashInferCutlassMoeQuantInfo(
                quant_type="bf16",
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                output_dtype=x.dtype,
                moe_ep_size=layer.moe_ep_size,
                moe_ep_rank=layer.moe_ep_rank,
                moe_tp_size=layer.moe_tp_size,
                moe_tp_rank=layer.moe_tp_rank,
                apply_routed_scaling_factor=not layer.should_fuse_routed_scaling_factor_in_topk,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_trtllm_moe:
```

---

## 8. fp8.py — Fp8LinearMethod

### 8.1 apply 路径 — Marlin / block / 默认 FP8 linear

**Explain：** `Fp8LinearMethod.apply` 按初始化时的 flags 分叉：**Marlin**（无 FP8 硬件时的 weight-only 快路径）、**mxfp8 block**、**w8a8 block**（dynamic group quant + dispatch 的 GEMM）、默认 **`apply_fp8_linear`**（per-channel weight + dynamic activation）。`ignored_layers` regex 匹配的层在 `get_quant_method` 阶段已 fallback 到 `UnquantizedLinearMethod`，不会进入此 apply。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/fp8.py L760-L776
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_marlin:
            return torch.ops.sglang.apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

```

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/fp8.py L801-L840
        if self.block_quant:
            if use_intel_amx_backend(layer):
                return torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
                    x,
                    layer.weight,
                    layer.weight_scale_inv,
                    self.quant_config.weight_block_size,
                    bias,
                    x.dtype,
                    True,  # is_vnni
                )

            if isinstance(x, tuple):
                return self.w8a8_block_fp8_linear(
                    input=x[0],
                    weight=layer.weight,
                    block_size=self.quant_config.weight_block_size,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=x[1],
                    bias=bias,
                )

            return self.w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
                bias=bias,
            )

        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
        )
```

**Comment：**

- block quant 路径：`w8a8_block_fp8_linear` 内部先 `scaled_fp8_quant` / group quant（§3），再调 §2 dispatch 返回的 GEMM。
- `apply_fp8_linear` 是 SM89+ 上最常见的 dynamic activation + per-channel weight 组合。
- MoE 层走 `Fp8MoEMethod.apply`（§8 之后独立类），与 Linear 共享 quant config 但 dispatch 到 MoeRunner。

---

## 9. marlin_utils.py — Marlin layout 检测

**Explain：** Marlin 要求特定 weight reorder layout 才能使用 fused kernel；check_marlin_format 检测 checkpoint 是否已 Marlin 格式，check_marlin_supported 检测当前 GPU 是否支持。GPTQ/AWQ 均可在 load 时做 prepare_*_for_marlin 转换。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/gptq/gptq.py L43-L48
def check_marlin_format(hf_quant_cfg: Dict[str, Any]) -> bool:
    # compat: gptqmodel and autogptq (eol) main use checkpoint_format: str
    # compat: autogptq <=0.7.1 is_marlin_format: bool
    return hf_quant_cfg.get("checkpoint_format") == "marlin" or hf_quant_cfg.get(
        "is_marlin_format", False
    )
```

**Comment：**
- Marlin 不支持 desc_act=True 的 GPTQ
- MoE Marlin 需 check_moe_marlin_supports_layer 额外检查

---

## 10. QuantizationConfig.get_quant_method

**Explain：** 每个 QuantizationConfig 子类实现 get_quant_method(layer, prefix)，按 layer 类型（Linear/MoE/Attention）和 prefix regex 返回对应 Method 实例。dynamic 规则允许同一模型不同 layer 使用不同 bits。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/base_config.py L102-L120
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        raise NotImplementedError

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        """Return a ``TritonMoeQuantInfo`` describing the quantisation state
        stored on *layer*.

        The LoRA MoE runner calls this so that ``invoke_fused_moe_kernel``
        receives the correct flags / scales / block-shape for the base
        weights.  Each quantisation method must override this with the
        same construction it already uses inside ``apply()``.
        """
```

**Comment：**
- prefix 是 layer 在 model 中的完整路径（如 `model.layers.0.self_attn.qkv_proj`）
- 返回 None 表示该 layer 不量化
