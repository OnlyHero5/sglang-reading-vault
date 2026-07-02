---
type: batch-doc
module: 19-Quantization
batch: "19"
doc_type: faq
title: "Quantization：关键问题"
tags:
 - sglang/batch/19
 - sglang/module/quantization
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# Quantization：关键问题

## Q1：FP8 vs GPTQ 怎么选？

**Explain：** FP8（W8A8）需要 Tensor Core 支持，activation 也量化到 8bit，适合 H100/B200 等原生 FP8 硬件；GPTQ 是 4bit weight-only + fp16 activation，适合消费级 GPU 或无 FP8 TC 的场景。FP8 checkpoint 可直接 load FP8 weight；GPTQ 需 group index 解压。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/fp8.py L124-L143
        raise RuntimeError(
            "DeepSeek-V4 FP4 experts require torch.float4_e2m1fn_x2 support."
        )
    return fp4_dtype


if _use_aiter or _use_hip_int4:
    from aiter.ops.shuffle import (
        shuffle_scale,
        shuffle_weight,
    )

if _use_aiter:
    from sglang.srt.layers.quantization.fp8_utils import (
        aiter_w8a8_block_fp8_linear,
        use_aiter_triton_gemm_w8a8_tuned_gfx950,
    )


ACTIVATION_SCHEMES = ["static", "dynamic"]
```

**易错对比：**

```python
# ❌ 在 A100 上强开 --fp8-gemm-backend=flashinfer_trtllm——SM 不够会 RuntimeError
# ✅ A100 用 auto 或 triton/deepgemm；B200 可选 flashinfer_trtllm
```

---

## Q2：Marlin 是什么？何时启用？

**Explain：** Marlin 是一种 GPU weight layout + fused GEMM kernel，专为 4bit GPTQ/AWQ 优化。checkpoint 已含 `checkpoint_format=marlin` 时直接 load；否则 load 后可 runtime 做 prepare_for_marlin reorder（一次性开销）。硬件不支持时 silent fallback 到标准 GPTQ kernel。

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

---

## Q3：dynamic activation vs static activation？

**Explain：** dynamic 每 token/group 计算 activation scale，精度更高但有 quant kernel 开销；static 使用固定 scale（从 calibration 或 checkpoint），apply 路径更简单。server_args 和环境变量 `SGLANG_FP8_IGNORED_LAYERS` 可排除特定 layer。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/fp8.py L144-L156

logger = logging.getLogger(__name__)


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
```

---

## Q4：KV cache 量化如何与 Attention 配合？

**Explain：** BaseKVCacheMethod 不为 layer 做 GEMM apply，而是提供 k_scale/v_scale 给 Attention backend。写入 KV pool 前 quantize K/V 到 FP8/FP4；Attention 计算时 dequantize 回 compute dtype。FNuz 平台 scale 需 ×2 修正。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/kv_cache.py L449-L456
 def process_weights_after_loading(self, layer) -> None:
 if layer.k_scale > 0.0 and layer.v_scale > 0.0:
 k_scale = layer.k_scale.to("cpu").tolist()
 v_scale = layer.v_scale.to("cpu").tolist()
 if is_fp8_fnuz():
 k_scale *= 2
 v_scale *= 2
```

---

## Q5：MoE 层量化与 Linear 层有何不同？

**Explain：** MoE 使用 FusedMoEMethodBase 而非 LinearMethodBase；create_weights 签名含 num_experts，create_moe_runner 绑定 MoeRunner。EP 下每个 rank 只 quant 本地 expert weight。FP8 MoE 可选 block-scaled 或 mxfp8 layout，kernel config JSON 按 E/N/device 自动选择。

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

## Q6：GPTQ dynamic per-module 规则怎么用？

**Explain：** `GPTQConfig.dynamic` 是 regex→override dict；`+:` 前缀 positive match 覆盖 bits/group_size，`-:` 前缀 negative match 跳过量化。layer prefix 与 regex 匹配后应用 override，否则用 base config。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/gptq/gptq.py L188-L210

        assert isinstance(layer, FusedMoE)
        raise NotImplementedError("GPTQConfig does not support MoE.")


class GPTQAscendConfig(GPTQConfig):
    """Config class for GPTQ on Ascend NPU."""

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            'NPU hardware does not support "get_min_capability" feature.'
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[LinearMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
```

---

## Q7：SGLang 量化路径与 vLLM 如何对照选型？

**Explain：** vLLM 以 `quantization` 字符串（`fp8`, `awq`, `gptq`, `compressed-tensors` 等）在 loader 层选 `QuantizationConfig`；SGLang 在 `srt/layers/quantization/` 按 checkpoint `quantization_config` 与 `ServerArgs` 组合注册 `Fp8Config`/`GPTQConfig`/Marlin 等。两者都支持 FP8 W8A8 与 GPTQ 4bit，但 SGLang 额外深度集成 **KV cache quant**（`BaseKVCacheMethod`）与 **MoE FusedMoEMethodBase**，vLLM 侧 MoE FP8 路径随版本快速演进。迁移时先对齐 checkpoint format（是否 Marlin、是否 serialized FP8），再比 hardware backend（FlashInfer TRT-LLM vs Marlin kernel）。

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

**Comment：** vLLM 文档中的 `quantization=fp8` 大致对应 SGLang `--quantization fp8` + 硬件 capability 检查；A100 上勿强开 B200 专用 FP8 GEMM backend。

---

## 设计追问

### Q1：为何 SGLang 把 KV quant 独立成 `BaseKVCacheMethod` 而非复用 Linear quant？

**Explain：** KV cache 不参与 weight GEMM，而是 Attention 读写路径上的 scale 元数据。独立 Method 让各 Attention backend 在 matmul 前 dequant K/V，而不改动 Linear 的 `apply_weights`。FNuz 平台还有单独 scale 修正分支。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/kv_cache.py L449-L456
 def process_weights_after_loading(self, layer) -> None:
 if layer.k_scale > 0.0 and layer.v_scale > 0.0:
 k_scale = layer.k_scale.to("cpu").tolist()
```

**Comment：** 与 vLLM KV cache dtype 配置类似，但 SGLang 与 RadixAttention slot 布局绑定更紧。

---

### Q2：MoE 层为何不能直接用 Linear 的 GPTQ apply？

**Explain：** Expert 权重是三维 `(num_experts, ...)`，路由后仅激活 subset；`FusedMoEMethodBase` 提供 `create_moe_runner` 绑定 MoeRunner kernel config。EP 下每 rank 只 load 本地 expert，quant layout 与 TP 分片规则不同。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/base_config.py L86-L90
class FusedMoEMethodBase(QuantizeMethodBase):

    def create_weights(
        self,
        layer: torch.nn.Module,
```

**Comment：** DeepSeek MoE + FP8 需同时读MoE 与 19。

---

### Q3：dynamic per-module GPTQ regex 在生产中解决什么问题？

**Explain：** 同一模型内 attention 层与 MoE expert 对量化敏感度不同；`GPTQConfig.dynamic` 用 regex 对浅层提 bits、对 MoE skip，在精度与体积间做细粒度 trade-off。negative match `-:.*\.moe\..*` 避免误伤路由层。

**Code：**

```python
# 来源：python/sglang/srt/layers/quantization/gptq/gptq.py L188-L210

        assert isinstance(layer, FusedMoE)
        raise NotImplementedError("GPTQConfig does not support MoE.")


class GPTQAscendConfig(GPTQConfig):
    """Config class for GPTQ on Ascend NPU."""

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            'NPU hardware does not support "get_min_capability" feature.'
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[LinearMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
```

**Comment：** vLLM 侧类似能力多在 `compressed-tensors` schema 或外部 calibration 脚本中表达，迁移时需重写规则而非复制 JSON。

---

## 验证建议（零基础可试）

1. **确认量化配置被识别** 
 - 操作：启动带 `quantization_config` 的 HF 模型，看日志中的 `Fp8Config` / `GPTQConfig` 等字样。 
 - 预期：无「unsupported quantization」；权重 load 完成。 
 - 对应：[[19-Quantization-01-核心概念|01-核心概念 §2]]

2. **切换 FP8 GEMM backend** 
 - 操作：同一模型试 `--fp8-gemm-backend deep_gemm` 与 `triton`（若支持）。 
 - 预期：吞吐或延迟有差异；错误 backend 可能 fallback 或报错。 
 - 对应：用户故事「FP8 上线」

3. **检查 kernel 是否走 sgl_kernel** 
 - 操作：设 `SGLANG_KERNEL_API_LOGLEVEL=1` 再发一条 generate。 
 - 预期：日志打印实际 dispatch 的 kernel 名；便于确认未 silent fallback。 
 - 对应：[[26-sgl-kernel-00-MOC|26-sgl-kernel]]
