---
title: "Backward · 学习检查"
type: exercise
framework: flash-attn
topic: "Backward"
learning_role: practice
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# Backward · 学习检查

## 读者能做什么

- [ ] 能画出 `dO + Q/K/V/O/LSE/RNG -> D -> P -> dS -> dQ/dK/dV` 主线。
- [ ] 能解释 forward 为什么保存 `out/softmax_lse/rng_state`，而不是保存完整 `P`。
- [ ] 能写出 `D = sum(dO * O)` 和 `dS = P * (dO V^T - D)`。
- [ ] 能从 `FlashAttnFunc.backward` 追到 `_flash_attn_backward`、`flash_attn_gpu.bwd` 和 C++ `mha_bwd`。
- [ ] 能说明 `Flash_bwd_params` 相比 `Flash_fwd_params` 新增了哪些指针。
- [ ] 能解释 preprocess、main backward、convert_dQ 三段 launch 顺序。
- [ ] 能说明 deterministic、varlen、GQA/MQA 各自改变了什么 layout 或归约边界。
- [ ] 能说明 `flash_attn_with_kvcache` 为什么不是训练 backward API。

## 最小运行验收

在已安装 FlashAttention 扩展且有 CUDA 的环境中运行：

```powershell
python - <<'PY'
import torch
from flash_attn import flash_attn_func

torch.manual_seed(0)
q = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.float16, requires_grad=True)

out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True, deterministic=True)
loss = out.float().square().mean()
loss.backward()
for name, tensor in [("q", q), ("k", k), ("v", v)]:
    grad = tensor.grad
    print(name, grad.shape, torch.isfinite(grad).all().item(), grad.float().norm().item())
PY
```

预期现象：

- `q/k/v.grad` 都存在。
- shape 与输入一致。
- `isfinite` 为 `True`。
- norm 是有限正数。

如果失败，按这个顺序定位：

| 失败现象 | 回看 |
|----------|------|
| import 或 ABI 错误 | [[FlashAttention-Python-API-排障指南]] |
| dtype/head dim/contiguous 报错 | [[FlashAttention-Backward-源码走读]] 的 `mha_bwd` 检查 |
| dropout 梯度不稳定 | [[FlashAttention-Backward-数据流]] 的 RNG 边界 |
| deterministic 显存或耗时异常 | [[FlashAttention-Backward-排障指南]] 的 split 归约 |
| varlen shape 错 | [[FlashAttention-Backward-数据流]] 的 `cu_seqlens` 与 `unpadded_lse` |

## 源码定位练习

| 问题 | 应定位到 |
|------|----------|
| Python backward 保存了什么 | `flash_attn/flash_attn_interface.py` 的 `FlashAttnFunc.forward/backward` |
| Python 到 C++ 的 custom op 桥在哪里 | `flash_attn/flash_attn_interface.py` 的 `_flash_attn_backward` |
| C++ dense backward 入口在哪里 | `csrc/flash_attn/flash_api.cpp` 的 `mha_bwd` |
| varlen backward 怎么表达 batch 边界 | `csrc/flash_attn/flash_api.cpp` 的 `mha_varlen_bwd` |
| `Flash_bwd_params` 有哪些字段 | `csrc/flash_attn/src/flash.h` |
| `D=sum(dO*O)` 在哪里计算 | `csrc/flash_attn/src/flash_bwd_preprocess_kernel.h` |
| 主 backward kernel 如何形成 `dS` | `csrc/flash_attn/src/flash_bwd_kernel.h` |
| backward launch 顺序在哪里 | `csrc/flash_attn/src/flash_bwd_launch_template.h` |

## 口述验收

用五分钟讲清楚：

> FlashAttention backward 如何只依赖 `Q/K/V/O/LSE/RNG` 和 `dO` 重算 softmax 概率，并生成 `dQ/dK/dV`。

合格答案必须包含：

- `out` 用来算 `D=sum(dO*O)`。
- `softmax_lse` 用来在 tile 内恢复 `P=exp(S-LSE)`。
- dropout backward 需要 forward 的 `rng_state`。
- `dS=P*(dO V^T-D)` 是连接 softmax backward 与 Q/K 梯度的中间量。
- `dV=P^T dO`、`dQ=dS K`、`dK=dS^T Q`。
- deterministic 和 varlen 是 layout/归约变化，不是公式变化。

## 下一步

对比推理路径：进入 [[FlashAttention-KV-Cache]]，看 decode 为什么只有 forward cache update，没有训练 backward。
