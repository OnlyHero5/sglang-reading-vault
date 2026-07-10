---
title: "Quantization · 学习检查"
type: exercise
framework: sglang
topic: "Quantization"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# Quantization · 学习检查

## 读者能做什么

- [ ] 能画出六本账：配置账、绑定账、权重账、装载账、整形账、执行账。
- [ ] 能沿 `HF config → QuantizationConfig → quant_method → create_weights → load_weights → postprocess → forward` 复述一次生命周期。
- [ ] 能说清 Linear、MoE、KV cache 三类消费者的输入和输出分别是什么。
- [ ] 能解释为什么普通 GPTQ 不支持 MoE，而 AWQ MoE 可以在不支持 Marlin 时 fallback。
- [ ] 能指出 KV cache 量化为什么不应该调用 `apply`。
- [ ] 能用一个日志、断点或配置实验验证实际 backend 或 method。

## 改代码前自查

新增或修改一个 quant method 前，逐项确认：

- [ ] `QuantizationConfig.from_config` 读取字段完整，默认值和 HF config 兼容。
- [ ] `get_quant_method(layer, prefix)` 对 Linear、MoE、RadixAttention 的返回值明确；不支持时 fail fast 或返回安全 fallback。
- [ ] `create_weights` 注册了所有 checkpoint 会加载的参数，并标注必要的 weight loader 属性。
- [ ] `process_weights_after_loading` 覆盖 layout reorder、scale 修正、在线量化或 workspace 准备。
- [ ] `apply` 的输入 ABI 与 consumer 一致：Linear 是 `(layer, x, bias)`，MoE 是 `(layer, dispatch_output)`，KV method 不应执行 GEMM。
- [ ] 对 TP/EP、group size、pack factor、block size 的不变量有错误信息。
- [ ] 对显式 backend 的不可用情况有清晰报错或 warning，不让 benchmark 误读。

## 排障自测

| 现象 | 你应该先查 | 能说出的判断 |
|------|------------|--------------|
| invalid quantization | 注册表 | 配置字符串未映射到 config 类 |
| dtype 不支持 | `_get_quantization_config` | config 已创建，但 activation dtype 不合法 |
| FP8 backend 不一致 | `fp8_utils.py` | 显式 backend 与 auto 分支语义不同 |
| GPTQ TP 报 shape | `GPTQLinearScheme.create_weights` | 分片后维度不满足 group/pack 对齐 |
| AWQ MoE fallback | `AWQConfig.get_quant_method` | Marlin support check 失败或 layer 被跳过 |
| KV scale 错 | `BaseKVCacheMethod.process_weights_after_loading` | scale 没规范化成 per-tensor float |
| unquant MoE 有 quant info | `unquant.py` | runner ABI 需要 bf16 quant info |

## 最小验证实验

- [ ] 启动一个量化模型，故意给错 `--quantization`，确认错误发生在配置账。
- [ ] 对支持 FP8 的模型切换 `--fp8-gemm-backend auto` 和一个显式 backend，观察日志或错误是否符合预期。
- [ ] 在 `LinearBase.forward` 或具体 method `apply` 加断点，确认当前 layer 的 method 类型。
- [ ] 在 `DefaultModelLoader.load_weights_and_postprocess` 遍历 module 时加断点，确认带 `quant_method` 的 layer 都执行 postprocess。
- [ ] 对 KV cache 量化模型，在加载后检查 `k_scale_float/v_scale_float`。

## 核心结论

量化在 SGLang 中是一条生命周期，不是一条 kernel 调用。配置决定 method，method 创建参数，loader 灌权重并整形，forward 才按 Linear、MoE、KV cache 三种 ABI 消费。能按六本账定位问题，就不会把配置错误、shape 对齐、backend 选择和 Attention scale 混成一个“量化不对”的黑箱。
