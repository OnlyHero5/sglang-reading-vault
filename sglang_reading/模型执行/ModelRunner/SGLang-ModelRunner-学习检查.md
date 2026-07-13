---
title: "ModelRunner · 学习检查"
type: exercise
framework: sglang
topic: "ModelRunner"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-11
---
# ModelRunner · 学习检查

## 读者能做什么

- [ ] 能画出 `Scheduler.run_batch → TpModelWorker.forward_batch_generation → ForwardBatch.init_new → ModelRunner.forward → _forward_raw → runner.execute → sample → GenerationBatchResult`。
- [ ] 能解释 `ScheduleBatch` 和 `ForwardBatch` 的边界，并说明后者为什么是借用字段、可受控变形的工作对象，而非深拷贝快照。
- [ ] 能区分 live batch、DP/MLP padding 后 batch、eager registry view、Graph replay view。
- [ ] 能解释 `out_cache_loc` 是 generic KV write location，不能无条件当物理 slot。
- [ ] 能用 `ForwardMode` 判断一次 batch 可能走 decode graph、prefill graph、split prefill 还是 eager。
- [ ] 能解释 `forward_metadata_ready`、计划 shape、`replan_equivalent` 与 `needs_forward_metadata_init()`。
- [ ] 能说明 PP 末 rank 与非末 rank 的结果字段差异。
- [ ] 能解释 overlap + grammar 为什么会产生 `delay_sample_func`，以及 Scheduler 何时执行它。
- [ ] 能说出在线权重更新后 graph 是否重建由哪个参数控制。

## 源码定位验收

不用打开全文，只凭函数名应能定位：

| 问题 | 入口 |
|------|------|
| Scheduler 把 batch 交给执行层 | `Scheduler.run_batch` |
| Worker 把调度态换成执行态 | `TpModelWorker.forward_batch_generation` |
| 执行工作对象构造 | `ForwardBatch.init_new` |
| graph/eager 选路 | `ModelRunner._forward_raw` |
| eager view 与 metadata | `EagerRunner.load_batch` / `_execute_decode` / `_execute_extend` |
| decode graph replay | `DecodeCudaGraphRunner.execute` |
| Graph replay view | `build_replay_fb_view` / `DecodeCudaGraphRunner.load_batch` |
| 采样 token | `ModelRunner.sample` |
| delayed sampling | `Scheduler.launch_batch_sample_if_needed` |

## 失败模式验收

- [ ] decode 吞吐低：能从 `can_run_cuda_graph` 回到 `_forward_raw` 和 capture bs。
- [ ] prefill graph 没生效：能从启动日志回到 `init_prefill_cuda_graph` 的禁用条件。
- [ ] spec/DP padding 后索引错位：能追到 metadata 的 owner、计划 shape 和是否允许 replan。
- [ ] PP rank 没有 token：能先判断是否末 rank。
- [ ] structured output 显存涨：能检查 delayed sampling 闭包是否释放 logits 和 vocab mask。
- [ ] embedding 请求没 token：能说明它返回 embedding，不走 generation sampling。

## 运行或观测验收

任选一种完成：

- [ ] 固定模型、硬件、backend、batch/context 与输入，对照 Graph on/off；记录路径、输出与实测吞吐，不预设方向。
- [ ] 对 structured output 请求断点 `delay_sample_func`，确认采样后 `next_token_ids` 被填入。
- [ ] 在 PP 环境断点末 rank 与非末 rank 分支，确认返回字段不同。
- [ ] 在线更新权重时打开重建 graph 参数，确认更新后重新 capture decode graph。

## 维护检查

改写本组文档后执行：

```powershell
node maintenance\audit_source_evidence.mjs --note 'sglang_reading\模型执行\ModelRunner\SGLang-ModelRunner-源码走读.md'
node maintenance\audit_source_evidence.mjs
node maintenance\audit_wikilinks.mjs
git diff --check
```

通过标准：

- 源码引用文件存在，行号在当前 upstream 范围内。
- 双链无断链。
- 本组不再以源码摘录数量作为验收项。
- checkpoint 能让读者复述主线、定位失败模式并做至少一个验证实验。
