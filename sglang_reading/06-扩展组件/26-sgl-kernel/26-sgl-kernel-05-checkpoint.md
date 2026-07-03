---
type: batch-doc
module: 26-sgl-kernel
batch: "26"
doc_type: checkpoint
title: "sgl-kernel 验收清单"
tags:
 - sglang/batch/26
 - sglang/module/sgl-kernel
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# sgl-kernel 验收清单

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明 sgl-kernel 是 srt 底层 CUDA 算子库，Python 薄封装 + architecture-specific `.so`
- [ ] 能画出 srt Layer → sgl_kernel Python → torch.ops → common_ops.so 的调用链
- [ ] 能说出 `_load_architecture_specific_ops`、`merge_state_v2`、`moe_align_block_size` 的职责（文档中均有内嵌代码）
- [ ] 能追踪 MoE forward 或 import 加载链经过本模块的路径（26-sgl-kernel-03-数据流与交互.md 有逐步讲解）

## 验证统计（2026-07-02 人工复核）

| 文件 | ETC 段数（Explain+Code+Comment） | 内嵌代码行数 |
|------|----------------------------------|-------------|
| 26-sgl-kernel-00-MOC.md | 1 | 22 |
| 26-sgl-kernel-01-核心概念.md | 5 | 48 |
| 26-sgl-kernel-02-源码走读.md | 13 | 198 |
| 26-sgl-kernel-03-数据流与交互.md | 6 | 72 |
| 26-sgl-kernel-04-关键问题.md | 6 | 42 |
| **合计** | **31** | **~382** |

- ETC 段数 ≥ 15：✅（31）
- 代码行数 ≥ 200：✅（~382）
- 26-sgl-kernel-03-数据流与交互.md 完整：✅

## 核心结论（3 句话）

1. sgl-kernel 通过 `load_utils` 按 GPU SM 版本加载 `common_ops.so`，注册全部 `torch.ops.sgl_kernel.*` 算子。
2. Python 模块按 attention/MoE/gemm/KV/speculative/sampling 分域，只做校验与 dispatch，热点在 csrc。
3. srt 推理栈在 layer forward、PD disagg、投机解码等路径 import 并调用本包，与 FlashInfer/Triton 形成互补 fallback。

## 遗留问题

- csrc 层 CUDA kernel tile 配置未在本模块展开（读者若需 GPU 性能调优可另开专题）。
- Metal/MUSA 专算子仅概念覆盖，未逐 kernel 走读。
