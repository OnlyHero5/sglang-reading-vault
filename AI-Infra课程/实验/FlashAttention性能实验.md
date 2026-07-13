---
title: "FlashAttention 性能实验"
type: exercise
framework: flash-attn
topic: "Attention Kernel"
learning_role: practice
source_baseline: "002cce0"
difficulty: intermediate
estimated_time: "90 到 180 分钟"
prerequisites:
  - "[[Attention算子主线]]"
tags:
  - framework/flash-attn
  - content/exercise
  - source-reading
updated: 2026-07-13
---

# FlashAttention 性能实验

## 读者任务

这不是“跑一个越快越好的数字”，而是建立一条可复现证据链：先确认当前 Python 环境确实加载了 FA2 extension，再用同一组 Q/K/V 对照 PyTorch reference，随后改变 shape 与 causal 条件，最后用 profiler 解释时间变化来自 dispatch、数据搬运还是片上资源压力。

## 环境边界

| 项目 | 最低要求 | 不满足时怎么做 |
|------|----------|----------------|
| 正确性与计时 | CUDA GPU、可导入 `torch` 与 `flash_attn_2_cuda` | 完成静态定位，不报告性能数字 |
| FA2 CUDA 主线 | Ampere 或更新架构、fp16/bf16 | ROCm、FA3、FA4 另建实验，不混入本表 |
| Nsight Systems/Compute | `nsys`、`ncu` 在 PATH | 保存 smoke 脚本，转到具备工具的 Linux/GPU 节点 |
| 可比结果 | 固定 GPU、dtype、shape、warmup、重复次数 | 缺任一项时只记录现象，不下性能结论 |

先从知识库根目录确认入口仍存在：

```powershell
rg -n 'flash_attn_func|mha_fwd|run_mha_fwd|flash_fwd_kernel|softmax_rescale_o' flash-attn/flash-attention
```

预期：依次命中 Python API、C++ `mha_fwd`、launch/dispatch、forward kernel 与 online softmax。只命中 Python wrapper，不能证明 CUDA extension 可用。

## 第一步：生成可重复使用的 smoke 脚本

下面的脚本同时完成 extension 检查、reference 对照、warmup 和小型 shape sweep。它不会调用仓库中默认的大规模 benchmark，因此更适合第一次验证。

```powershell
@'
import itertools
import time
import torch
import torch.nn.functional as F

from flash_attn import flash_attn_func
import flash_attn_2_cuda  # noqa: F401

assert torch.cuda.is_available(), "需要 CUDA GPU"
torch.manual_seed(0)
device = "cuda"
dtype = torch.float16

def reference(q, k, v, causal):
    # 转成 B,H,S,D；用 fp32 math reference，最后转回输入 dtype。
    return F.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
        dropout_p=0.0,
        is_causal=causal,
    ).transpose(1, 2).to(dtype)

def run_case(seqlen, headdim, causal, repeats=30):
    shape = (2, seqlen, 8, headdim)
    q, k, v = [torch.randn(shape, device=device, dtype=dtype) for _ in range(3)]
    ref = reference(q, k, v, causal)
    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
    max_abs = (out - ref).abs().max().item()
    close = torch.allclose(out, ref, atol=5e-3, rtol=5e-3)

    for _ in range(10):
        flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / repeats
    print({"S": seqlen, "D": headdim, "causal": causal,
           "ms": round(ms, 4), "max_abs": max_abs, "allclose": close})
    assert close, "先解决正确性，再解释性能"

print("gpu=", torch.cuda.get_device_name(0), "torch=", torch.__version__)
for seqlen, headdim, causal in itertools.product([128, 512], [64, 128], [False, True]):
    run_case(seqlen, headdim, causal)
'@ | Set-Content -Encoding UTF8 fa_smoke.py

$env:PYTHONPATH = (Resolve-Path 'flash-attn/flash-attention').Path
python fa_smoke.py
```

预期：打印 8 组结果，全部 `allclose=True`。时间数字只在同一机器内部横向比较；首次 import、编译或 cache 建立不计入稳定迭代。

如果 import 失败，先分别执行：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import flash_attn_2_cuda; print('extension ok')"
```

前者失败是 PyTorch/CUDA 环境问题；后者失败通常是 wheel、编译产物或 ABI 问题，不能据此评价 kernel 性能。

## 第二步：解释 shape，而不是只抄时间

每组至少记录：GPU、CUDA/PyTorch/FlashAttention 版本、`B/S/H/D`、dtype、causal、warmup、重复次数和 CUDA-event 区间时间。该区间测量的是一次 operator 调用的 device work；若 dispatch 包含多个 kernel，不能自动称为“单 kernel 时间”。

预期判断：

- `D=64` 与 `D=128` 可能进入不同预实例化或 tile 组合。
- causal 改变有效 score 区域和 mask 分支，不能把它与 non-causal 合并成一条曲线。
- `S` 增大时总时间上升是必然现象；真正要比较的是同一 workload 下的实现和 profiler 证据。
- `allclose=False` 时，任何“更快”结果都无效。

仓库还提供 `benchmarks/benchmark_flash_attention.py`，但它默认扫描到 `seqlen=16384`，可能占用大量显存。先读配置并按硬件缩小范围，再运行；不要把它当 smoke test。

## 第三步：Profiler

```powershell
nsys profile -o flash_attn_trace python fa_smoke.py
ncu --set full --target-processes all -o flash_attn_ncu python fa_smoke.py
```

`ncu --set full` 成本很高。首次使用时先把 sweep 临时缩成一个 shape、`repeats=1`，确认目标 kernel 后再扩大；不要在共享 GPU 上直接对全部 8 组 × 30 次做 full collection。

预期：

- Systems 时间线能看到 extension/kernel launch；Python import 与 warmup 应和稳定迭代分开。
- Compute 报告同时查看 DRAM bytes、带宽、Tensor Core/SM 利用率、occupancy、register 和 shared memory。
- occupancy 高不等于整体更快；register/shared-memory 压力、访存与 tile 复用必须一起解释。

## KV cache 对照的边界

`flash_attn_with_kvcache` 是带 cache 长度、可选新 K/V、paged block table 和 SplitKV 状态的 decode 契约，不是把 `flash_attn_func` 的 `S` 改小。进入该实验前先读 [[FlashAttention-KV-Cache-数据流]]，并单独记录 `seqlen_q`、cache 长度、page block size 与 `num_splits`。

预期：decode 热点更偏向历史 KV 读取和 split combine；如果 cache shape、长度或 block table 非法，应在 wrapper/C++ 检查或 launch 边界失败，而不是产生一份可用于比较的性能结果。

## 通过标准

- [ ] `flash_attn_2_cuda` 可导入，GPU、版本和 shape 已记录。
- [ ] 8 组 reference 对照全部通过，再开始解释时间。
- [ ] 能指出至少两个 shape 为什么可能进入不同 specialization。
- [ ] profiler 结论同时引用时间、HBM/DRAM 与片上资源证据。
- [ ] 能区分 dense forward 与 KV cache decode 的输入契约。

实验结束删除临时脚本：

```powershell
Remove-Item -LiteralPath 'fa_smoke.py'
```
