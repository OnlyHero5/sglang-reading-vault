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
updated: 2026-07-12
---
# Quantization · 学习检查

## 读者能做什么

- [ ] 能画出六本账：配置账、绑定账、权重账、装载账、整形账、执行账。
- [ ] 能沿 `HF config → QuantizationConfig → quant_method → create_weights → load_weights → postprocess → forward` 复述一次生命周期。
- [ ] 能说清 Linear、MoE、KV cache 三类消费者的输入和输出分别是什么。
- [ ] 能解释默认 `GPTQConfig` 为何拒绝 MoE，并区分 GPTQ Marlin、NPU GPTQ、CPU AMX GPTQ；同时区分普通 AWQ、AWQ Marlin、NPU AWQ、CPU AWQ，只把 Moe WNA16 fallback 归给 `AWQMarlinConfig`。
- [ ] 能解释 GPTQ `group_size=-1` 的 scale-size 与 `g_idx` 初始化为何形成可疑契约，并设计 checkpoint 覆盖/内核消费验证。
- [ ] 能指出 KV cache 量化为什么不应该调用 `apply`。
- [ ] 能区分 FP8 backend 的选择期门禁和调用期 shape/dtype fallback，并用日志确认逐层最终 kernel。
- [ ] 能画出原始 server arg → 初始化规范化 → selected callable → 最终 kernel 四层，并解释 SM120/MXFP8 的 auto 改写。
- [ ] 能指出当前 HIP unrolled-x4 helper 缺少 `return` 的可疑可达性问题，并设计不预设性能方向的验证。
- [ ] 能区分 raw `get_quant_method` 与 layer 最终 method，并解释 MoE unquant/KTEP wrapper、ROCm QKV/RowParallel 主动清空 config。

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
| FP8 backend 不一致 | `fp8_utils.py` | 显式 backend 不走 auto，但所选 `*_with_fallback` 仍可能因逐层 shape/dtype 转 Triton |
| GPTQ TP 报 shape | `GPTQLinearScheme.create_weights` | 分片后维度不满足 group/pack 对齐 |
| AWQ MoE fallback | `AWQMarlinConfig.get_quant_method` | 先确认配置类与平台，再判断 Marlin support check、skip 或 Moe WNA16 fallback |
| KV scale 错 | `BaseKVCacheMethod.process_weights_after_loading` | scale 没规范化成 per-tensor float |
| unquant MoE 有 quant info | `unquant.py` | runner ABI 需要 bf16 quant info |

## 最小验证实验

- [ ] 启动一个量化模型，故意给错 `--quantization`，确认错误发生在配置账。
- [ ] 对支持 FP8 的模型切换 `--fp8-gemm-backend auto` 和一个显式 backend，同时选择至少两个不同 shape 的 Linear，确认选择期报错与调用期 fallback 的区别。
- [ ] 在 `LinearBase.forward` 或具体 method `apply` 加断点，确认当前 layer 的 method 类型。
- [ ] 先记录实际 loader 类；Default 路线在全量加载后观察 postprocess，Layered/ModelOpt 路线在各自处理循环观察，并确认 forward 前参数已是 kernel-ready。
- [ ] 对 KV cache 量化模型，在加载后检查 `k_scale_float/v_scale_float`。

## 核心结论

量化在 SGLang 中是一条生命周期，不是一条 kernel 调用。配置决定 method，method 创建参数，loader 灌权重并整形，forward 才按 Linear、MoE、KV cache 三种 ABI 消费。能按六本账定位问题，就不会把配置错误、shape 对齐、backend 选择和 Attention scale 混成一个“量化不对”的黑箱。
