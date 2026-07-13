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
updated: 2026-07-12
---
# Backward · 学习检查

## 读者能做什么

- [ ] 能画出 `dO + Q/K/V/O/LSE/RNG -> D -> P -> dS -> dQ/dK/dV` 主线。
- [ ] 能解释 forward 为什么保存 `out/softmax_lse/rng_state`，而不是保存完整 `P`。
- [ ] 能区分数学 `D = sum(dO * O)` 与 dropout 实现中乘过 `p_keep` 的 `dsoftmax_sum`，并解释最终缩放为何延后。
- [ ] 能从 `FlashAttnFunc.backward` 追到 `_flash_attn_backward`、`flash_attn_gpu.bwd` 和 C++ `mha_bwd`。
- [ ] 能说明 `Flash_bwd_params` 相比 `Flash_fwd_params` 新增了哪些指针。
- [ ] 能解释 preprocess、main backward、convert_dQ 三段 launch 顺序。
- [ ] 能说明 deterministic、varlen、GQA/MQA 各自改变了什么 layout 或归约边界。
- [ ] 能说明 dQ/dK 最终乘 `softmax_scale / p_keep`，而 dV 只乘 `1 / p_keep`。
- [ ] 能说明 `flash_attn_with_kvcache` 为什么不是训练 backward API。

## 最小运行验收

在已安装 FlashAttention 扩展且有 CUDA 的环境中运行：

```powershell
@'
import torch
from flash_attn import flash_attn_func

torch.manual_seed(0)
B, S, H, D = 2, 64, 4, 64
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
qr = q.detach().clone().requires_grad_(True)
kr = k.detach().clone().requires_grad_(True)
vr = v.detach().clone().requires_grad_(True)

out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True, deterministic=True)
scores = torch.einsum("bshd,bthd->bhst", qr.float(), kr.float()) * (D ** -0.5)
mask = torch.triu(torch.ones(S, S, device="cuda", dtype=torch.bool), diagonal=1)
probs = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
out_ref = torch.einsum("bhst,bthd->bshd", probs, vr.float())
dout = torch.randn_like(out)
out.backward(dout)
out_ref.backward(dout.float())

for name, actual, reference in [
    ("out", out, out_ref), ("dq", q.grad, qr.grad),
    ("dk", k.grad, kr.grad), ("dv", v.grad, vr.grad),
]:
    diff = (actual.float() - reference.float()).abs()
    print(name, "finite=", bool(torch.isfinite(actual).all()),
          "max_abs=", diff.max().item(), "mean_abs=", diff.mean().item())
'@ | python -
```

预期现象：

- 四行均为 `finite=True`。
- `out/dq/dk/dv` 都与显式 causal reference 对齐。
- 同时观察 `max_abs` 与 `mean_abs`；误差判断必须绑定这里的 FP16、shape 和输入分布，不写脱离 workload 的全局阈值。

当前维护环境没有可加载的 CUDA FlashAttention 扩展。静态替代不是验证梯度正确性，只验证实验代码可执行：

```powershell
@'
import ast
from pathlib import Path

p = Path("flash-attn_reading/CUDA内核/Backward/FlashAttention-Backward-学习检查.md")
text = p.read_text(encoding="utf-8")
block = text.split("```powershell", 1)[1].split("```", 1)[0]
code = block.split("@'", 1)[1].rsplit("'@ | python -", 1)[0]
ast.parse(code)
print("backward exercise syntax: OK")
'@ | python -
```

预期输出为 `backward exercise syntax: OK`。

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
| `D` 的点积、`p_keep` 对齐和共享 `dQaccum` 清零在哪里 | `csrc/flash_attn/src/flash_bwd_preprocess_kernel.h` |
| 主 backward kernel 如何形成 `dS` | `csrc/flash_attn/src/flash_bwd_kernel.h` |
| backward launch 顺序在哪里 | `csrc/flash_attn/src/flash_bwd_launch_template.h` |

## 口述验收

用五分钟讲清楚：

> FlashAttention backward 如何只依赖 `Q/K/V/O/LSE/RNG` 和 `dO` 重算 softmax 概率，并生成 `dQ/dK/dV`。

合格答案必须包含：

- `out` 用来形成数学 `D=sum(dO*O)`；dropout 实现 buffer 会乘 `p_keep` 与内部 `dP` 对齐。
- `softmax_lse` 用来在 tile 内恢复 `P=exp(S-LSE)`。
- dropout backward 需要 forward 的 `rng_state`。
- `dS=P*(dO V^T-D)` 是连接 softmax backward 与 Q/K 梯度的中间量。
- `dV=P^T dO`、`dQ=dS K`、`dK=dS^T Q`。
- deterministic 隔离 `dQaccum` split 并固定归并；varlen 改变 packed 地址与 LSE layout，它们都不改主公式。
- dQ/dK 与 dV 的最终缩放所有权不同。

## 下一步

对比推理路径：进入 [[FlashAttention-KV-Cache]]，看 decode 为什么只有 forward cache update，没有训练 backward。
