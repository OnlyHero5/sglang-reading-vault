---
title: "sgl-kernel · 学习检查"
type: exercise
framework: sglang
topic: "sgl-kernel"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---

# sgl-kernel · 学习检查

## 你为什么要做这组检查

这组检查判断你是否能从 SRT 层调用追到 Python wrapper、`torch.ops` 注册和 architecture-specific 动态库，而不是只知道“这里有 CUDA 算子”。

## 能力检查

- [ ] 能把 SRT 算法门禁、Python wrapper、dispatcher schema、C++ launcher、device kernel 五层分开。
- [ ] 能解释 `sm90/` 与 `sm100/` 首先是 fast-math/precise-math 变体，实际硬件覆盖还取决于构建时 gencode。
- [ ] 能指出加载器的三次 fallback 都是扩展搜索 fallback，不是 Triton/PyTorch 算法 fallback。
- [ ] 能沿 `merge_state()` 证明 FP8、非 CUDA 或不满足 pack-size 时由 SRT 改走 Triton。
- [ ] 能说明 sampling renorm 在 FlashInfer 可用且非 MUSA 时不进入内部 `torch.ops.sgl_kernel`。
- [ ] 能比较 CUDA 与 ROCm `init_custom_ar` 的参数和后续方法集合，拒绝“custom allreduce 仅 ROCm”这一结论。
- [ ] 能识别当前基线的两个静态风险：CUDA runtime preload 顺序与 `torch.torch.ops` 的 GPTQ shuffle typo。

## 最小验证

操作：

```powershell
rg -n 'if compute_capability == 90|ops_subdir = "sm100"|use_fast_math|TORCH_LIBRARY_FRAGMENT|m\.impl\("merge_state_v2"|def merge_state\(|return merge_state_triton|torch\.torch\.ops|_preload_cuda_library' sglang/sgl-kernel/python/sgl_kernel/load_utils.py sglang/sgl-kernel/python/sgl_kernel/__init__.py sglang/sgl-kernel/python/sgl_kernel/gemm.py sglang/sgl-kernel/CMakeLists.txt sglang/sgl-kernel/csrc/common_extension.cc sglang/python/sglang/srt/layers/attention/merge_state.py
```

预期：同时看到 CC90/其他设备的目录二分、只有 fast-math target 多出的编译选项、CUDA schema 绑定、SRT 的 Triton fallback、`common_ops` 先于 preload 的调用顺序，以及 GPTQ shuffle 的双重 `torch` 属性。任何一项缺失都说明基线已变化，应回源码更新专题，而不是机械保留旧结论。

## 目标环境实验

在装有匹配 wheel 的 Linux GPU 环境，记录以下事实而不是只报耗时：

1. 操作：import 前设置 `SGLANG_KERNEL_API_LOGLEVEL=1`，调用一次 `merge_state()` 的支持 shape 和一个不支持 dtype/shape。预期：支持样本进入 `sgl_kernel.merge_state_v2`，不支持样本走 Triton；两组输出都与高精度参考在给定 tolerance 内一致。
2. 操作：有/无 FlashInfer 两个环境分别调用 top-k renorm。预期：有 FlashInfer且非 MUSA时走 FlashInfer；缺失时进入内部 op，结果分布一致。
3. 操作：记录 `common_ops.__file__`、GPU compute capability 与 wheel 内 cubin/PTX 架构。预期：CC90 指向 `sm90/`，其余指向 `sm100/`，且二进制确实覆盖目标设备。

若本机没有匹配 GPU/wheel，静态检查只能确认控制流与注册关系，不能宣称 kernel 已运行、数值正确或更快。

## 复盘

深读调用链见 [[SGLang-sgl-kernel-源码走读]]，张量和跨层边界见 [[SGLang-sgl-kernel-数据流]]。
