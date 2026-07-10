---
title: "Python-API · 学习检查"
type: exercise
framework: flash-attn
topic: "Python-API"
learning_role: practice
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# Python-API · 学习检查

## 读者任务

这篇用于验收你是否已经能把 Python API 层作为入口契约来读：能画出 dense 调用主线，能检查 varlen 边界，能区分 import/ABI、layout、custom op、KV cache 等问题的第一源码入口。

## 读者能做什么

- [ ] 能画出 `flash_attn_func → FlashAttnFunc.apply → _wrapped_flash_attn_forward → flash_attn_gpu.fwd → mha_fwd`。
- [ ] 能区分 dense、packed、varlen、KV cache 四类 API 的输入形态和使用场景。
- [ ] 能解释 `maybe_contiguous`、head dim padding、`softmax_scale` 默认值分别在哪一层发生。
- [ ] 能说明 forward 为什么保存 `q/k/v/out/softmax_lse/rng_state` 给 backward。
- [ ] 能说出 `return_attn_probs` 为什么是 testing-only，不是生产 attention map。
- [ ] 能用 `cu_seqlens[-1] == total_nnz` 检查 varlen 边界。
- [ ] 能判断 import/ABI、stride/dtype、custom op、varlen、KV cache 五类问题的第一源码入口。

## 可执行检查

### 1. API 表面

```powershell
@'
from flash_attn import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
print("api ok")
'@ | python -
```

预期：Python API import 成功。如果失败，先看安装和 extension ABI，不要进入 kernel。

### 2. Extension 边界

```powershell
@'
import flash_attn_2_cuda
print("extension ok")
'@ | python -
```

预期：`flash_attn_2_cuda` 可 import。若失败，检查 wheel、PyTorch、CUDA、ROCm fallback。

### 3. Varlen 边界

```powershell
@'
import torch
from flash_attn.bert_padding import unpad_input

x = torch.randn(2, 4, 3)
mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.int32)
packed, indices, cu_seqlens, max_s, used = unpad_input(x, mask)
print(packed.shape, indices.numel(), cu_seqlens.tolist(), max_s)
assert cu_seqlens.tolist() == [0, 2, 5]
assert packed.shape[0] == indices.numel() == 5
'@ | python -
```

预期：`cu_seqlens` 表示每条样本的边界，最后一个值等于 packed token 数。

### 4. Custom op 可见性

```powershell
@'
import torch
import flash_attn.flash_attn_interface as fai
print(torch.__version__)
print(hasattr(torch.ops, "flash_attn"))
print(fai._wrapped_flash_attn_forward)
'@ | python -
```

预期：PyTorch 2.4+ 环境下应能看到 `torch.ops.flash_attn` 注册路径；旧版本会退回 Python wrapper。

## 口述练习

用三分钟讲清楚：

> 一个 padded training batch 如何经过 `unpad_input → flash_attn_varlen_func → varlen_fwd → pad_input`，在减少 padding token 计算的同时保持样本边界不互相污染。

再用三分钟讲清楚：

> 一个 decode step 为什么应该走 `flash_attn_with_kvcache`，而不是把它当作普通 `flash_attn_func` 的小 batch。

## 下一步

进入 [[FlashAttention-FA2-Forward]]，沿着 `mha_fwd → Flash_fwd_params → run_mha_fwd → flash_fwd_kernel` 看 CUDA forward 主路径；decode 相关问题进入 [[FlashAttention-KV-Cache]]。
