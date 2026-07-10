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
updated: 2026-07-10
---
# FA2-Forward · 学习检查

> 这个 checkpoint 不按“看过几段代码”验收，而按你能不能独立定位和解释 forward 主线验收。

## 必达能力

- [ ] 能从 `flash_attn_func(q,k,v)` 说到 `mha_fwd`，并说明为什么本专题从 C++ forward 开始。
- [ ] 能列出 `mha_fwd` 的关键输入约束：Ampere+、fp16/bf16、CUDA device、last-dim contiguous、head dim 上限/对齐、GQA 整除。
- [ ] 能解释 `Flash_fwd_params` 里至少 8 类字段：Q/K/V/O 指针、stride、heads、sequence length、scale、dropout、window/causal、LSE/P 指针。
- [ ] 能说明 fixed-length 与 varlen 的参数分界：`cu_seqlens_q/k == nullptr` vs 累计长度数组。
- [ ] 能解释 `kBlockM`、`kBlockN`、`kHeadDim`、`kNWarps` 分别控制什么。
- [ ] 能从 `D=64` 或 `D=128` 找到对应 head-dim launch helper，并说出 dropout/causal/GPU 架构为什么会影响 traits。
- [ ] 能按顺序复述 kernel 主循环：load tile、QK GEMM、mask/softcap、online softmax、P 转换、PV GEMM、累积 `acc_o`。
- [ ] 能解释 `row_max/row_sum` 如何让分块扫描得到完整行 softmax 的等价结果。
- [ ] 能解释为什么主路径只写回 `O` 和 `LSE`，不写完整 `P`。

## 源码定位任务

| 任务 | 入口 | 通过标准 |
|------|------|----------|
| 找到 dtype/head dim 检查 | `csrc/flash_attn/flash_api.cpp` | 能定位到 `mha_fwd` 的检查段，并解释每个检查保护的 kernel 假设。 |
| 找到 dense 参数包边界 | `set_params_fprop` | 能指出 fixed-length 路径为什么传 `cu_seqlens_* = nullptr`。 |
| 找到普通 forward 与 SplitKV 分叉 | `run_mha_fwd` | 能指出 `num_splits <= 1` 与 split dispatch 的分界。 |
| 找到 grid 形状 | `run_flash_fwd` | 能解释 grid 三个维度分别是 query block、batch、head。 |
| 找到 head_dim=64 traits | `run_mha_fwd_hdim64` | 能比较 dropout/non-dropout 的 tile 差异。 |
| 找到 online softmax 更新 | `softmax_rescale_o` | 能解释旧 `acc_o` 为什么需要重缩放。 |
| 找到输出写回 | `flash_fwd_kernel.h` epilogue | 能指出 `LSE` 和 `O` 分别在哪里写回。 |

## 运行或静态验证

有 CUDA 环境时，跑一个小形状对比：

```python
import torch
from flash_attn import flash_attn_func

torch.manual_seed(0)
q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)

out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
ref = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
).transpose(1, 2)
print((out - ref).abs().max())
```

没有 CUDA 环境时，完成静态验证：

- [ ] 在 `flash_api.cpp` 找到 `softcap > 0` 与 dropout 不兼容的检查。
- [ ] 在 `flash_fwd_launch_template.h` 找到 head_dim=64 的 dropout/non-dropout traits。
- [ ] 在 `flash_fwd_kernel.h` 找到 `gemm -> mask -> softmax_rescale_o -> gemm_rs` 的顺序。
- [ ] 在 `softmax.h` 找到 `normalize_softmax_lse` 对 `acc_o` 的归一化。

## 口述验收

用五分钟讲清楚：

> FA2 fixed-length forward 如何把一个 query block 从 `mha_fwd` 的 tensor 输入，变成 CUDA grid 中的一个 CTA；这个 CTA 如何扫描 K/V blocks，在寄存器中维护 online softmax 和输出累积；最后为什么只写回 `O` 与 `LSE`。

如果讲不清 `row_max/row_sum`，回到 [[FlashAttention-Online-Softmax]]；如果讲不清 dispatch，回到 [[FlashAttention-FA2-Forward-源码走读]]；如果讲不清内存位置，回到 [[FlashAttention-FA2-Forward-数据流]]。
