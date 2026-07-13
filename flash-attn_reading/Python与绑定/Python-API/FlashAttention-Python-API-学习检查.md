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

- [ ] 能先画 backend 路由，再画 `flash_attn_func → FlashAttnFunc.apply → wrapped op → flash_attn_gpu.fwd`；只在 compiled-extension 分支继续到 pybind/`mha_fwd`。
- [ ] 能区分 dense、packed、varlen、KV cache 四类 API 的输入形态和使用场景。
- [ ] 能解释 `maybe_contiguous`、head dim padding、`softmax_scale` 默认值分别在哪一层发生。
- [ ] 能说明 forward 为什么保存 `q/k/v/out/softmax_lse/rng_state` 给 backward。
- [ ] 能说出 `return_attn_probs` 为什么是 testing-only，不是生产 attention map。
- [ ] 能解释 dropout 为 0 时公开三元组的第三项为什么不能当真实概率矩阵。
- [ ] 能用 `cu_seqlens[-1] == total_nnz` 检查 varlen 边界，并区分 packed length 与第五返回值。
- [ ] 能说明 KV-cache 为什么不经过 dense/varlen 的 custom-op/fake 路径、为什么没有 backward。
- [ ] 能判断 import/ABI、stride/dtype、custom op、varlen、KV cache 五类问题的第一源码入口及适用 backend。

## 可执行检查

### 1. 无 extension 也必须通过的静态契约

```powershell
@'
import ast
from pathlib import Path

root = Path("flash-attn/flash-attention/flash_attn")
init_tree = ast.parse((root / "__init__.py").read_text(encoding="utf-8"))
interface_tree = ast.parse((root / "flash_attn_interface.py").read_text(encoding="utf-8"))
padding_tree = ast.parse((root / "bert_padding.py").read_text(encoding="utf-8"))

exports = {
    alias.name
    for node in init_tree.body
    if isinstance(node, ast.ImportFrom)
    for alias in node.names
}
required = {"flash_attn_func", "flash_attn_qkvpacked_func", "flash_attn_varlen_func", "flash_attn_with_kvcache"}
assert required <= exports

unpad = next(node for node in padding_tree.body if isinstance(node, ast.FunctionDef) and node.name == "unpad_input")
ret = next(node for node in ast.walk(unpad) if isinstance(node, ast.Return))
assert isinstance(ret.value, ast.Tuple) and len(ret.value.elts) == 5

names = {node.name for node in interface_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
assert {"FlashAttnFunc", "flash_attn_func", "flash_attn_varlen_func", "flash_attn_with_kvcache"} <= names
print("static API contracts: PASS")
'@ | python -
```

预期：公开 API、五项 unpad 返回与三类入口定义全部存在。这个检查不导入包，因此不会把缺 extension/Aiter 与 Python 契约混在一起。

### 2. Backend 分支定位

```powershell
rg -n 'USE_TRITON_ROCM|import flash_attn_2_cuda|flash_attn_gpu|noop_custom_op_wrapper|register_fake|fwd_kvcache' flash-attn/flash-attention/flash_attn/flash_attn_interface.py
```

预期：同时看到 HIP fallback、compiled/Triton backend 选择、PyTorch 2.4 版本分叉与 KV-cache 直调。普通 CUDA 的 extension import 失败不应被解释成自动 Triton fallback。

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
assert used.tolist() == [2, 3]
'@ | python -
```

预期：依赖与包 import 可用时，`cu_seqlens` 表示每条样本边界，最后一个值等于 packed token 数，第五返回值只统计 `attention_mask`。若包级 import 被 extension/Aiter 阻断，记录环境限制并以第 1 项 AST 契约为静态替代，不能宣称行为实验通过。

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

预期：backend 可加载时，PyTorch 2.4+ 应能看到 `torch.ops.flash_attn` 注册路径；旧版本退回 Python wrapper。该检查只覆盖 dense/varlen wrapped op，不替 KV-cache 证明 compile/fake 支持。

### 5. 条件化动态矩阵

| 环境 | 必须验收 | 通过信号 |
|------|----------|----------|
| CUDA compiled extension | import、dense forward、可选 backward | `flash_attn_2_cuda` 可加载，固定 shape 下 out 与 reference 在声明容差内 |
| HIP compiled extension | import 与对应 HIP kernel | 实际 `flash_attn_gpu` 指向 extension；不要套用 CUDA SM 门禁 |
| ROCm Triton | Aiter import 与 dense/varlen 路径 | `USE_TRITON_ROCM=True`，实际 backend 指向 Aiter Triton |
| 任意 GPU backend | `return_attn_probs` 两种 dropout | dropout=0 第三项为空槽位；dropout>0 只作 testing/debug 对照 |
| KV-cache 合格环境 | output、cache 写回、下一 step 可见性 | 三项同时正确；确认接口无 backward，非法 paged 组合被拒绝 |

## 口述练习

用三分钟讲清楚：

> 一个 padded training batch 如何经过 `unpad_input → flash_attn_varlen_func → varlen_fwd → pad_input`，在减少 padding token 计算的同时保持样本边界不互相污染。

再用三分钟讲清楚：

> 一个 decode step 为什么应该走 `flash_attn_with_kvcache`，而不是把它当作普通 `flash_attn_func` 的小 batch。

最后回答两个反例题：

1. 为什么“public varlen API 有 `block_table` 参数”不能直接推出 paged-varlen backward 受支持？预期答案应指出 backward 状态不保存 block table，upstream 测试仅在 `block_table is None` 时执行梯度。
2. 为什么“pybind 表存在 `fwd_kvcache`”不能推出所有平台都经过这张 C++ 表？预期答案应先判断当前 `flash_attn_gpu` 是 compiled extension 还是 ROCm Triton。

## 通过标准

- 静态契约脚本必须通过；否则不能进入下一专题。
- 四条主线必须按条件作答，不能画成单一 CUDA 直线。
- 动态环境不可用时必须明确记录未验收项，不能用 AST/`rg` 代替数值、ABI 或性能结论。
- 能独立解释 `S_dmask`、五项 unpad 返回、paged-varlen backward 证据边界和 KV-cache 无 backward，才算通过。

## 下一步

进入 [[FlashAttention-FA2-Forward]] 时，先限定为 compiled CUDA forward，再沿 `mha_fwd → Flash_fwd_params → run_mha_fwd → flash_fwd_kernel` 阅读；ROCm backend 不套用这条链。decode 相关问题进入 [[FlashAttention-KV-Cache]]。
