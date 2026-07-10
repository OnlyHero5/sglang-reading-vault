---
title: "SGLang 综合学习检查"
type: exercise
framework: sglang
topic: "总结复盘"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# SGLang 综合学习检查

## 你为什么要做

读完目录不等于能处理真实请求。本页检查你能否从现象回到对象、进程、资源和源码证据，并能设计最小实验验证判断。

## 请求主线

- [ ] 不看文档，画出 `GenerateReqInput -> Req -> ScheduleBatch -> ForwardBatch -> token id -> text chunk`。
- [ ] 为每次交接标注函数调用、ZMQ、GPU 执行或事件唤醒。
- [ ] 解释 `rid` 为什么不是 batch 下标，也不是 KV 地址。
- [ ] 说明 token id 已产生但文本未返回时，为什么不应先怀疑 attention kernel。

## 调度与 KV

- [ ] 解释 waiting、running、prefill、decode、chunked prefill 和 retract 的关系。
- [ ] 区分逻辑 prefix 命中与物理 KV slot 分配。
- [ ] 给出 KV 压力上升时的观测顺序：queue、KV usage、retract、TTFT/TPOT、失败率。
- [ ] 说明 LoRA、prefix cache 与 `extra_key` 为什么会影响共享边界。

## 模型与 GPU

- [ ] 从 architecture 字符串找到模型类，再说明 checkpoint 参数名怎样写入 fused 参数。
- [ ] 区分 ModelRunner、attention backend 和具体 kernel 的职责。
- [ ] 用 profiler 或日志证明实际 backend，而不是根据配置名猜测。
- [ ] 解释 prefill 与 decode 为什么可能选择不同优化路径。

## 分布式与高级特性

- [ ] 画出至少一个 TP/PP group，并说明 global rank 为什么不足以判断职责。
- [ ] 解释 PD 分离新增的请求状态、KV transfer 与 Decode prealloc 边界。
- [ ] 解释 speculative decoding 的 draft、verify、accept/reject 对 KV 和输出顺序的要求。
- [ ] 说明权重更新为什么需要版本、cache 与 active request 的一致性处理。

## 排障演练

任选一个症状：请求长期 waiting、prefix 不命中、decode retract、无文本输出、backend 未生效、多卡 hang。

**操作：**

1. 写下当前对象与最后一个可信观测。
2. 在主线上圈出可能出错的唯一交接。
3. 找到该交接两端的源码入口。
4. 设计一次只改变一个变量的静态或运行验证。

**预期：** 最终记录包含症状、假设、证据、操作、实际结果和结论；若假设被否定，应保留结果并移动到下一条边界，而不是继续堆配置。

## 已知边界

- 本库源码说明基于 commit `70df09b`，行号漂移时以函数名和对象名重新定位。
- CUDA serving 是主线；TPU、Ascend 等平台差异需要结合对应 upstream 文档和环境验证。
- 未单独展开的 HiCache、remote KV connector 等主题可从 [[SGLang-补充主题]] 进入，但结论仍需按当前版本核对。

## 通过标准

你能独立完成 [[SGLang服务实验]] 中一个对照实验，并用对象生命周期解释结果；遇到不知道的分支时，能从 [[SGLang-源码地图]] 找入口，而不是依赖旧编号或固定阅读套件。
