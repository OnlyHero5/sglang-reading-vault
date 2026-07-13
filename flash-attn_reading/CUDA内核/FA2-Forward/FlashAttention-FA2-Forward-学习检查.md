---
title: "FA2-Forward · 学习检查"
type: exercise
framework: flash-attn
topic: "FA2-Forward"
learning_role: practice
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# FA2-Forward · 学习检查

> 这个 checkpoint 以基线 `002cce0` 的 FA2 CUDA 为准，不按“看过几段代码”验收，而按你能不能独立定位、解释并验证 forward 主线验收。

## 必达能力

- [ ] 能从 `flash_attn_func(q,k,v)` 说到 `mha_fwd`，并说明为什么本专题从 C++ forward 开始。
- [ ] 能列出 `mha_fwd` 的关键输入约束：Ampere+、fp16/bf16、CUDA device、last-dim contiguous、head dim 上限/对齐、GQA 整除。
- [ ] 能解释 `Flash_fwd_params` 里至少 8 类字段：Q/K/V/O 指针、stride、heads、sequence length、scale、dropout、window/causal、LSE/P 指针。
- [ ] 能说明 fixed-length 与 varlen 的参数分界：`cu_seqlens_q/k == nullptr` vs 累计长度数组。
- [ ] 能解释 `kBlockM`、`kBlockN`、`kHeadDim`、`kNWarps` 分别控制什么。
- [ ] 能从 `D=64` 或 `D=128` 找到对应 head-dim launch helper，并说出 dropout/causal/GPU 架构为什么会影响 traits。
- [ ] 能按顺序复述 kernel 主循环：load tile、QK GEMM、softcap、mask/ALiBi、online softmax、指数分子转换、权重乘 V、累积 `acc_o`。
- [ ] 能解释 `row_max/row_sum` 如何让分块扫描得到完整行 softmax 的等价结果。
- [ ] 能解释 `acc_s/rP` 为什么不是最终概率，以及最终除 `row_sum` 在哪里发生。
- [ ] 能解释为什么主路径只写回 `O` 和 `LSE`，不写完整 `P`。
- [ ] 能解释为什么 fixed-length `flash_attn_func` 仍可能进入 SplitKV，并区分 standard、aligned single-split、multi-split+combine。
- [ ] 能解释全 mask 时 non-split `+inf`、split partial `-inf` LSE 哨兵，以及 empty-K 入口旁路。

## 源码定位任务

| 任务 | 入口 | 通过标准 |
|------|------|----------|
| 找到 dtype/head dim 检查 | `csrc/flash_attn/flash_api.cpp` | 能定位到 `mha_fwd` 的检查段，并解释每个检查保护的 kernel 假设。 |
| 找到 dense 参数包边界 | `set_params_fprop` | 能指出 fixed-length 路径为什么传 `cu_seqlens_* = nullptr`。 |
| 找到普通 forward 与 SplitKV 分叉 | `run_mha_fwd` | 能指出 `num_splits <= 1 && !force_split_kernel` 的 standard 条件，以及强制 single-split 与 multi-split 的不同回程。 |
| 找到 grid 形状 | `run_flash_fwd` | 能解释 grid 三个维度分别是 query block、batch、head。 |
| 找到 head_dim=64 traits | `run_mha_fwd_hdim64` | 能比较 dropout/non-dropout 的 tile 差异。 |
| 找到 online softmax 更新 | `softmax_rescale_o` | 能解释旧 `acc_o` 为什么需要重缩放。 |
| 找到输出写回 | `flash_fwd_kernel.h` epilogue | 能指出 `LSE` 和 `O` 分别在哪里写回。 |

## 运行或静态验证

有 CUDA 环境时，跑一个小形状对比：

```python
import math
import torch
from flash_attn import flash_attn_func

torch.manual_seed(0)
q = torch.randn(2, 5, 8, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 7, 2, 64, device="cuda", dtype=torch.float16)  # GQA: Hq / Hkv = 4
v = torch.randn(2, 7, 2, 64, device="cuda", dtype=torch.float16)
scale = 1 / math.sqrt(q.shape[-1])
k_qheads = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
v_qheads = v.repeat_interleave(q.shape[2] // v.shape[2], dim=2)

def reference(q, k, v, causal, dtype):
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if causal:
        row = torch.arange(q.shape[1], device=q.device)[:, None]
        col = torch.arange(k.shape[1], device=q.device)[None, :]
        keep = col <= row + k.shape[1] - q.shape[1]  # bottom-right 对齐
        scores = scores.masked_fill(~keep, -torch.inf)
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), v)

for causal in (False, True):
    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
    ref_fp32 = reference(q, k_qheads, v_qheads, causal, torch.float32)
    ref_low = reference(q, k_qheads, v_qheads, causal, q.dtype)
    fa_diff = (out.float() - ref_fp32).abs()
    low_diff = (ref_low.float() - ref_fp32).abs()
    report = {
        "causal": causal,
        "finite": torch.isfinite(out).all().item(),
        "flash_max_abs": fa_diff.max().item(),
        "flash_mean_abs": fa_diff.mean().item(),
        "low_precision_baseline_max_abs": low_diff.max().item(),
    }
    print(report)
    assert report["finite"]
    assert report["flash_max_abs"] <= 2 * report["low_precision_baseline_max_abs"] + 1e-5
```

预期：non-causal 与 bottom-right causal 两组都打印 `finite=True` 并通过断言，同时覆盖不等长 Q/K 和 GQA。这里复用上游“相对低精度基线误差倍数”的验收思路；`1e-5` 只是避免基线误差极小时的数值毛刺，不是跨 GPU/workload 的性能或精度承诺。

完整 upstream 测试还需要匹配的 CUDA GPU、已编译的 `flash_attn_2_cuda`、`einops` 与兼容的 PyTorch/NumPy 环境。若 `pytest --collect-only` 已因缺依赖、ABI 或扩展失败，就记录为环境门禁；零收集不能证明 kernel 正确。当前 Windows 维护环境在 collection 阶段缺 `einops`，同时存在 Torch/NumPy ABI 警告，因此这里只能完成语法、源码证据和静态契约验收。

没有 CUDA 环境时，完成静态验证：

- [ ] 在 `flash_api.cpp` 找到 `softcap > 0` 与 dropout 不兼容的检查。
- [ ] 在 `flash_fwd_launch_template.h` 找到 head_dim=64 的 dropout/non-dropout traits。
- [ ] 在 `flash_fwd_kernel.h` 找到 `gemm -> softcap -> mask -> softmax_rescale_o -> gemm_rs` 的顺序。
- [ ] 在 `softmax.h` 找到 `normalize_softmax_lse` 对 `acc_o` 的归一化。
- [ ] 在 fixed-length `mha_fwd` 找到 `set_params_splitkv(..., num_splits=0)`，并说明 standard、强制 aligned single-split 与 multi-split 的回程差别。
- [ ] 在 `softmax.h` 与 empty-K 分支找到全 mask/空 K 的 LSE 哨兵，并说明为何不能混为普通 softmax 数学值。

静态替代还应确认 Python padding 边界：在 `FlashAttnFunc.forward` 找到非 8 倍数 D 的 pad，以及返回前 `[..., :head_size_og]` 的裁剪。预期是公开 API 能处理这类 D；直接调用底层 C++ binding 不享受这个适配。

## 口述验收

用五分钟讲清楚：

> FA2 fixed-length API 如何经过 Python padding、C++ 参数装配与 SplitKV heuristic；standard kernel 如何把一个 query block 变成 grid 中的 CTA 并扫描 K/V blocks；SplitKV 又为什么要暂存 partial O/LSE 后 combine，而两条路径都不需要完整 P。

如果讲不清 `row_max/row_sum`，回到 [[FlashAttention-Online-Softmax]]；如果讲不清 dispatch，回到 [[FlashAttention-FA2-Forward-源码走读]]；如果讲不清内存位置，回到 [[FlashAttention-FA2-Forward-数据流]]。
