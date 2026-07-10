---
title: "多模态 · 学习检查"
type: exercise
framework: sglang
topic: "多模态"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# 多模态 · 学习检查

> 图片不会因为出现在 JSON 里就自动变成模型能吃的 tensor。它要经过选 processor、加载媒体、展开占位 token、生成视觉特征和对齐文本位置等一串交接。本页检查你是否真的能追踪这些交接。

## 你为什么要做

多模态故障常常跨越 API、processor、进程通信和模型 forward。完成这组检查后，你应能先判断交接在哪一层断掉，再进入对应源码，而不是从一张“看不懂”的图片一路猜到 CUDA。

## 读完应能完成

- [ ] 从模型架构名解释 `PROCESSOR_MAPPING` 如何选中具体 processor。
- [ ] 说清 `BaseMultimodalProcessor`、`MultimodalSpecialTokens` 与 `MultimodalProcessorOutput` 各自负责什么。
- [ ] 沿一张图片追踪媒体加载、预处理、placeholder 展开、视觉编码和 prefill 对齐。
- [ ] 解释 `grid_thw` 或等价元数据为什么必须和视觉 token 数量一致。
- [ ] 区分语义主线与性能通道：CUDA IPC、特征驻留 GPU、ViT CUDA Graph 可以改变搬运方式，但不能改变 token 与特征的对应关系。

## 先画一张交接图

不看笔记，补全下面的对象流：

```text
请求中的图片与文本
  -> 模型架构选择 __________
  -> 媒体加载与预处理
  -> 生成 __________ 和视觉输入
  -> ViT / vision encoder 生成视觉特征
  -> 按 placeholder 位置并入 prefill
  -> language model forward
```

**预期：** 两个空分别能落到具体 processor 与展开后的多模态 token/元数据；不能把“图片路径”直接连到 attention 层。

## 静态验证

**操作：** 在仓库根目录执行以下检索，并逐个打开命中位置：

```powershell
rg -n "PROCESSOR_MAPPING|def import_processors|def get_mm_processor" sglang/python/sglang/srt/managers/multimodal_processor.py
rg -n "class MultimodalSpecialTokens|class BaseMultimodalProcessor|grid_thw|placeholder" sglang/python/sglang/srt/multimodal/processors/base_processor.py
rg -n "class MultimodalProcessorOutput" sglang/python/sglang/srt/managers/schedule_batch.py
rg -n "keep_mm_feature_on_device" sglang/python/sglang/srt/server_args.py sglang/python/sglang/srt/multimodal
```

**预期：**

1. processor 注册表与选择函数位于 manager 层，具体模型行为位于 `processors/`。
2. placeholder 数量不匹配时存在显式校验，而不是悄悄截断或补齐。
3. `keep_mm_feature_on_device` 只改变特征所在设备和搬运路径，不替代 processor 生成的结构化输出。

## 故障推演

| 现象 | 先检查 | 为什么 |
|------|--------|--------|
| 架构报“不支持多模态” | `get_mm_processor` 与注册表 | 可能根本没有选到具体 processor |
| 图片能加载但 token 数量不对 | placeholder 展开与 `grid_thw` | 文本位置和视觉网格没有对齐 |
| 跨进程后特征为空或设备错误 | CUDA IPC / device 搬运分支 | 语义对象存在，但交接通道出了问题 |
| 只在第二次请求或 graph replay 出错 | ViT CUDA Graph 的 shape 与缓存键 | 被复用的执行形状可能不再匹配当前输入 |

## 口述验收

用三分钟解释“用户上传一张图问这是什么”如何进入模型。必须说出一个注册入口、一个 processor 基类、一个结构化输出、一个对齐不变量和一个性能优化边界。

讲不清注册与选择，回到 [[SGLang-多模态-核心概念]]；讲不清对象生命周期，回到 [[SGLang-多模态-数据流]]；要进入实现细节，再读 [[SGLang-多模态-源码走读]]。
