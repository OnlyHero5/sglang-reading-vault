---
title: "Online-Softmax · 学习检查"
type: exercise
framework: flash-attn
topic: "Online-Softmax"
learning_role: practice
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/exercise
  - source-reading
updated: 2026-07-11
---
# Online-Softmax · 学习检查

## 读者能做什么

- [ ] 能解释普通 block-wise softmax 为什么会错。
- [ ] 能画出 `row_max/row_sum/acc_o -> LSE` 的状态机。
- [ ] 能说明 `acc_s` 在 forward 主循环中从 score 变成 probability。
- [ ] 能解释新最大值出现时为什么 `row_sum` 和 `acc_o` 必须同时 rescale。
- [ ] 能在 `softmax.h` 找到 `Softmax<kNRows>`、`softmax_rescale_o`、`normalize_softmax_lse`。
- [ ] 能在 `flash_fwd_kernel.h` 找到 `QK -> mask -> softmax_rescale_o -> gemm_rs` 的顺序。
- [ ] 能说明 `Return_softmax` 分支和常规 backward 保存路径的区别。
- [ ] 能说明 `softmax_lse` 为什么足以支持 backward 重算 probability。

## 最小运行验收

这个脚本不依赖 CUDA，用纯 Python 验证 streaming online softmax 和全量 softmax 等价：

```powershell
@'
import math

scores = [1.0, -2.0, 3.0, 0.5, 4.0, -1.0]
values = [2.0, 5.0, -1.0, 3.0, 0.25, 7.0]
blocks = [(scores[:2], values[:2]), (scores[2:4], values[2:4]), (scores[4:], values[4:])]

m = -math.inf
l = 0.0
o = 0.0
for s_block, v_block in blocks:
    new_m = max(m, max(s_block))
    old_scale = 0.0 if m == -math.inf else math.exp(m - new_m)
    l *= old_scale
    o *= old_scale
    for s, v in zip(s_block, v_block):
        p = math.exp(s - new_m)
        l += p
        o += p * v
    m = new_m

online = o / l
full_m = max(scores)
den = sum(math.exp(s - full_m) for s in scores)
full = sum(math.exp(s - full_m) * v for s, v in zip(scores, values)) / den
print("online", online)
print("full", full)
print("match", abs(online - full) < 1e-12)
'@ | python -
```

预期现象：`match True`。

改错实验：删除脚本中的 `o *= old_scale` 再运行，`match` 应变为 `False`。这验证了源码里 `acc_o_rowcol *= scores_scale` 的必要性。

## 源码定位练习

| 问题 | 应定位到 |
|------|----------|
| 行级 max/sum 如何从 fragment 归约 | `csrc/flash_attn/src/softmax.h` 的 `thread_reduce_` / `quad_allreduce_` |
| `row_max/row_sum` 在哪里保存 | `csrc/flash_attn/src/softmax.h` 的 `Softmax<kNRows>` |
| 后续 block 如何重标尺历史状态 | `softmax_rescale_o` 的 `Is_first=false` 分支 |
| `P` 在 forward 主循环中何时被消费 | `csrc/flash_attn/src/flash_fwd_kernel.h` 的 `gemm_rs` 调用 |
| LSE 在哪里生成 | `normalize_softmax_lse` |
| Python 在哪里保存 LSE | `flash_attn/flash_attn_interface.py` 的 `ctx.save_for_backward` |

## 口述验收

用三分钟讲清楚：

> 一个 query 行分三块扫描 K/V 时，FlashAttention 如何保持全局 softmax 等价？

合格答案必须包含：

- 每块 score 先经过 mask/softcap，再进入 online softmax。
- 第一块初始化 `row_max/row_sum`。
- 后续块用新旧最大值比例缩放 `row_sum` 和 `acc_o`。
- 当前概率 tile 立即参与 `P @ V`，不保存完整 `P`。
- epilogue 归一化 `acc_o`，并生成 LSE 给 backward。

## 下一步

进入 [[FlashAttention-FA2-Forward]]，把这里的状态机放回完整 FA2 forward kernel。
