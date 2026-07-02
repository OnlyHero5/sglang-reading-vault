---
type: batch-doc
module: 19-Quantization
batch: "19"
doc_type: checkpoint
title: "Quantization 验收清单"
tags:
 - sglang/batch/19
 - sglang/module/quantization
 - sglang/doc/checkpoint
aliases:
 - "checkpoint"
updated: 2026-07-02
---
# Quantization 验收清单

## 读者自测（不打开 sglang/）

- [x] 能解释 QuantizationConfig + Method 双轨扩展模型
- [x] 能说明 FP8 dispatch 按 SM 版本选 backend 的逻辑
- [x] 能对比 GPTQ/AWQ/Marlin 路径差异
- [x] 能解释 KV cache quant 与 Attention 的协作方式
- [x] 五篇正文 ≥ 15 段内嵌源码

## 维护者检查

- [x] 覆盖 `base_config.py`、`fp8.py/fp8_utils.py`、`gptq/`、`awq/`、`kv_cache.py`
- [x] 行号对齐 git `70df09b`
- [ ] [[progress]] 由 P8 更新

## 核心结论（3 句话）

1. **QuantizationConfig 解析 HF config，get_quant_method 为每层绑定 Linear/MoE/KV Method**，create_weights + apply 双方法契约统一扩展点。
2. **FP8 按 SM 版本与 --fp8-gemm-backend dispatch 到 DeepGEMM/Triton/FlashInfer**，dynamic activation 需 per-token-group quant kernel。
3. **GPTQ/AWQ 4bit weight-only 可选 Marlin layout 加速；KV quant 通过 k_scale/v_scale 与 Attention backend 协作**，不做 GEMM apply。

## 内嵌源码统计（维护者）

| 文档 | ETC 段数（约） |
|------|----------------|
| README.md | 2 |
| 01-核心概念.md | 9 |
| 02-源码走读.md | 12 |
| 03-数据流与交互.md | 6 |
| 04-关键问题.md | 8 |
| **合计** | **37 段** |

合计内嵌源码行数：**约 250+ 行**
