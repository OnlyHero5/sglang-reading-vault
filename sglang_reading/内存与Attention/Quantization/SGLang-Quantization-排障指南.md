---
title: "Quantization · 排障指南"
type: troubleshooting
framework: sglang
topic: "Quantization"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# Quantization · 排障指南

## 读者任务

这篇按症状排障，不按量化格式背概念。遇到量化问题时，先判断它发生在六本账的哪一账：

| 症状 | 多半发生阶段 | 优先入口 |
|------|--------------|----------|
| `Invalid quantization method` | 配置账 | `get_quantization_config` |
| GPU capability 或 dtype 不支持 | 配置账到模型初始化 | `_get_quantization_config` |
| FP8 backend 与预期不同 | 执行账 | `dispatch_w8a8_block_fp8_linear` |
| static/dynamic scale 形态错 | 执行账 | `scaled_fp8_quant` |
| GPTQ TP size 报对齐错误 | 权重账 | `GPTQLinearScheme.create_weights` |
| 默认 GPU GPTQ 遇到 MoE 报错 | 绑定账 | `GPTQConfig.get_quant_method`；NPU、CPU AMX 与 GPTQ Marlin 另有 MoE 路径 |
| AWQ MoE 没走 Marlin | 绑定账 | `AWQMarlinConfig.get_quant_method`；同时确认当前平台是否走普通 AWQ、NPU 或 CPU 配置 |
| KV scale 相关错误 | 整形账到执行账 | `BaseKVCacheMethod.process_weights_after_loading` |
| unquant MoE 仍出现 quant info | 执行账 | `UnquantizedFusedMoEMethod.apply` |
| 全局量化开启但个别 Linear 是 unquant | 绑定账 | ignored layer、ROCm QKV/RowParallel config 清空、consumer 最终 method |

## 1. 启动时报 invalid quantization：先查注册表

`get_quantization_config` 会先检查字符串是否在当前进程的 `QUANTIZATION_METHODS` 中。这个表不是跨平台常量：`mxfp4`、NPU 的 GPTQ 覆盖、MPS 的 MLX 方法和 out-of-tree hook 都受运行平台影响。该错误说明还没有进入 HF config 解析，更没有进入 layer 或 kernel。

```python
# 来源：python/sglang/srt/layers/quantization/__init__.py L141-L165
def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    from sglang.srt.utils import is_cpu

    if is_cpu() and cpu_has_amx_support():
        if quantization not in CPU_QUANTIZATION_METHODS:
            raise ValueError(
                f"Invalid quantization method on CPU: {quantization}. "
                f"Available methods on CPU: {list(QUANTIZATION_METHODS.keys())}"
            )
        else:
            return CPU_QUANTIZATION_METHODS[quantization]

    if current_platform.is_out_of_tree():
        config = current_platform.get_quantization_config(quantization)

        # If the platform has a quantization config, use it else use the default
        if config is not None:
            return config

    return QUANTIZATION_METHODS[quantization]
```

排查动作：

- 确认 `--quantization` 字符串拼写是否在注册表里。
- CPU AMX 路径只允许 CPU 子集，不能把 GPU-only method 直接搬过去。
- CPU AMX 分支的错误消息当前打印的是全量 `QUANTIZATION_METHODS`，不是实际 `CPU_QUANTIZATION_METHODS`；诊断时以源码子集为准，不能照抄报错列表。
- out-of-tree platform 可能覆盖默认配置类，要确认 platform hook 是否返回了自定义 config。

## 2. GPU 或 dtype 不支持：这是启动期 fail fast

如果报 minimum capability 或 supported dtypes，说明 `quant_config` 已经创建，但在模型初始化前被硬件和 dtype 检查拦住。

```python
# 来源：python/sglang/srt/model_loader/loader.py L270-L289
        if not _is_npu:
            major, minor = get_device_capability()

            if major is not None and minor is not None:
                assert 0 <= minor < 10
                capability = major * 10 + minor
                if capability < quant_config.get_min_capability():
                    raise ValueError(
                        f"The quantization method {model_config.quantization} "
                        "is not supported for the current GPU. "
                        f"Minimum capability: {quant_config.get_min_capability()}. "
                        f"Current capability: {capability}."
                    )
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}"
            )
```

排查动作：

- 不要先怀疑 checkpoint 权重损坏；此时还没到权重加载主线。
- 对照当前 GPU capability 与 method 的 `get_min_capability`。
- 对照 `--dtype` 或模型 dtype 是否在 `get_supported_act_dtypes()` 返回列表中。

## 3. 显式 FP8 backend：不走 auto，不等于永不 fallback

显式 `--fp8-gemm-backend=flashinfer_trtllm` 会检查 SM100/SM103 和 FlashInfer 可用性，不满足直接报错；这证明它不会静默改走 auto 选择。但返回值是 `flashinfer_gemm_w8a8_block_fp8_linear_with_fallback`：进入具体调用后，TRT-LLM 遇到 `K < 256` 或非 BF16 等不支持格式仍会转 Triton。CUTLASS、DeepGEMM 也有 shape/dtype 级 fallback。

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

排查动作：

- 如果你想让系统自动选择 backend，用 auto。
- 把“选择期”与“调用期”分开：依赖/硬件门禁失败会报错，单层 shape/dtype 不满足则可能走 `*_with_fallback`。
- benchmark 前开启 kernel 日志，并按 layer 记录最终 kernel；仅记录 server arg 不能证明所有 GEMM 都执行首选 backend。

## 4. static/dynamic FP8 scale 搞混：看 activation 形态

`scaled_fp8_quant` 用 `scale is None` 区分 dynamic 和 static。dynamic 又分 per-token 和 per-tensor；static 要求传入标量 scale。但它只是一个 helper，不是所有 FP8 method 的统一入口：默认 CUDA 非 block、block/MXFP8、compressed-tensors 与平台专用路径可能调用别的量化函数。

```python
# 来源：python/sglang/srt/layers/quantization/fp8_kernel.py L1802-L1836
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

排查动作：

- static 报 scale numel 错时，先查 checkpoint 或 config 提供的 input scale 是否是标量。
- dynamic per-token 输出 scale shape 应为 `[tokens, 1]`，有 padding 时用 padded token 数。
- 输出精度异常但没有 load 错时，先确认实际 quant helper，再核对 activation scale 是 static 还是 dynamic；不要从配置字段直接假定调用了 `scaled_fp8_quant`。

## 5. GPTQ TP 对齐错误：这是权重槽位创建失败

GPTQ packed 权重要求 TP 分片后的输入和输出维度满足 group size 与 pack factor。错误消息里提到 tensor parallel size 时，通常是 `create_weights` 阶段拦截。

```python
# 来源：python/sglang/srt/layers/quantization/gptq/schemes/gptq_linear.py L48-L60
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor.numerator != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )
```

排查动作：

- 降低 TP size 或换 group size/模型分片组合。
- 不要把它当成 checkpoint tensor 缺失；此时是在创建 layer 参数槽位。
- 检查报错 layer 的 input/output partition，而不是全局 hidden size。

## 6. 默认 GPTQ 遇到 MoE 报错：先确认平台配置类

默认 `GPTQConfig` 对 `FusedMoE` 直接报错，提示使用 `gptq_marlin`。这不是跨平台定律：NPU 的 `gptq` key 被替换为 `GPTQAscendConfig`，CPU AMX 返回 `CPUGPTQConfig`，它们都提供 MoE scheme；`GPTQMarlinConfig` 也支持 MoE。

```python
# 来源：python/sglang/srt/layers/quantization/gptq/gptq.py L172-L181
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[LinearMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            raise TypeError("GPTQ Method does not support MoE, please use gptq_marlin")
        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQLinearMethod
        )
```

排查动作：

- 如果模型含 MoE，不要只看 checkpoint 写了 GPTQ；还要确认 SGLang 使用的 method 是否支持 MoE。
- 同时记录 `type(quant_config)` 和平台；只有默认 `GPTQConfig` 才由这张卡直接证明“不支持 MoE”。
- 错误发生在绑定账，说明 layer 类型已经识别出来，但配置类拒绝给 MoE 发 method。

## 7. AWQ MoE 没走 Marlin：先确认你用的是哪一种 AWQ 配置

这里的 fallback 属于 `AWQMarlinConfig`，不能外推到所有 `AWQConfig`。CUDA/HIP 的普通 `AWQConfig` 只给 Linear 发 method；NPU 的普通 AWQ 和 CPU AWQ 有各自的 MoE scheme。只有当前配置类确实是 `AWQMarlinConfig` 时，MoE 才会做 Marlin 支持检查，并在不满足时回退到 Moe WNA16、打印 warning。

```python
# 来源：python/sglang/srt/layers/quantization/awq/awq.py L347-L362
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
```

排查动作：

- 搜索 warning 文本，确认是否主动 fallback。
- 先记录 `type(quant_config)`；若是普通 CUDA/HIP `AWQConfig`，没有这条 MoE fallback 路径。
- 检查 `modules_to_not_convert` 是否跳过了该 layer。
- 检查 layer 结构和 group size 是否满足 `check_moe_marlin_supports_layer`。

## 8. KV cache scale 异常：确认加载后规范化结果

KV cache scale 不是 Linear apply 参数。它在 postprocess 中被规范化，并最终写入 `k_scale_float/v_scale_float`。

```python
# 来源：python/sglang/srt/layers/quantization/kv_cache.py L76-L85
        if not isinstance(k_scale, float) or not isinstance(v_scale, float):
            raise ValueError(
                "Only support per-tensor scaling factor " "for fp8 KV cache"
            )

        # These are used in the final Attention.forward()
        layer.k_scale.copy_(k_scale)
        layer.v_scale.copy_(v_scale)
        layer.k_scale_float = k_scale
        layer.v_scale_float = v_scale
```

排查动作：

- 加载后断点看 `k_scale_float/v_scale_float`，不要只看原始 checkpoint 字段。
- 如果 scale 不是 float，说明当前路径不支持 per-channel 或更细粒度 KV scale。
- checkpoint 没有 scale 是合法情况，postprocess 会把两路默认成 `1.0`。若仍保持初始 `-1.0`，说明 postprocess 没有完成或观察时机过早，不能把“checkpoint 未提供 scale”本身判成故障。

## 9. unquant MoE 为什么还有 quant info

无量化 MoE 仍走统一 runner，但 quant info 是 backend-specific ABI。下面的 dtype、EP/TP rank 与 routed scaling 字段只证明 FlashInfer CUTLASS 分支；其他 runner 的字段不同。DeepGEMM 中即使出现 `use_fp8=True`，也可能只是在描述 FP8 dispatch，而不是 bf16 expert 权重。

```python
# 来源：python/sglang/srt/layers/quantization/unquant.py L501-L512
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
```

排查动作：

- 不要把 `quant_info` 名称误读成“这里一定量化了权重”。
- 对 MoE 来说，quant info 是 runner ABI 的一部分；bf16 也需要 backend-specific 描述对象，但不同 runner 并不共享完全相同的字段结构。

## 10. 全局 quant config 已生效，为什么个别层仍是 unquant？

先区分三种来源：配置类的 ignored/exclude 规则主动返回 unquant method；ROCm 的 `SGLANG_ROCM_DISABLE_LINEARQUANT` 让 QKV/RowParallel 在绑定前清掉 config；MoE 的 raw method 为 `None` 时由 consumer 补成 `UnquantizedFusedMoEMethod`，KTEP 还可能再包 wrapper。

排查动作：

- 同时记录 model-level quantization、layer class/prefix、raw `get_quant_method` 返回值和最终 `layer.quant_method`。
- 在 ROCm 上记录 `SGLANG_ROCM_DISABLE_LINEARQUANT`，并分别检查 QKV、RowParallel 与其他 Linear，不能从一层外推全模型。
- MoE 还要记录 KTEP 配置和 wrapper 内部 `gpu_method`；只看最外层类名可能漏掉真实 GPU method。

## 复盘迁移

遇到量化问题时按这个顺序缩小范围：

1. 配置字符串是否存在，平台是否支持。
2. HF config 是否提供了 method 需要的字段。
3. layer 类型拿到的 raw method，是否被 consumer 清空、补成 unquant 或包成 wrapper。
4. `create_weights` 是否通过 shape 与 TP 对齐检查。
5. 实际 loader 是 Default、Layered、ModelOpt 还是特殊路线，是否在该路线承诺的时机完成 postprocess。
6. forward 时 consumer 是 Linear、MoE 还是 KV scale。
