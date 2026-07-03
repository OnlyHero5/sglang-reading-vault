---
type: batch-doc
module: 21-Speculative
batch: "21"
doc_type: checkpoint
title: "投机解码 验收清单"
tags:
 - sglang/batch/21
 - sglang/module/speculative
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# 投机解码 验收清单

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明投机解码 Draft→Verify 两阶段职责
- [ ] 能画出 Spec Worker 在 Scheduler 与 KV Cache 之间的位置
- [ ] 能说出 `SpeculativeAlgorithm`、`EAGLEWorkerV2`、`NGRAMWorker` 三者职责（文档中均有内嵌代码）
- [ ] 能追踪一条 EAGLE decode step：draft → verify → reject sampling → KV 写回
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 核心结论（3 句话）

1. `SpeculativeAlgorithm` 统一枚举与插件注册，通过 `create_worker` 分发到 V2 Worker（EAGLE/NGRAM/DFLASH/MTP 等）。
2. 每步 decode 经 Draft 提议候选、Target Verify 并行验证、Triton reject sampling 决定接受长度并写回 KV。
3. NGRAM 无 draft 模型与 draft KV，依赖语料库匹配；EAGLE 家族支持 PD 分离下的 draft hidden transfer。

## 遗留问题

- Multi-Layer EAGLE 与 Frozen KV MTP 的详细 forward 路径可在后续专项补充。
- Adaptive spec 参数调优策略需结合 benchmark 数据单独成文。

## Wave-3 升级（2026-07-02）

- [x] `21-Speculative-03-数据流与交互.md`：`EagleVerifyInput` / `move_accept_tokens_to_target_kvcache` / `AdaptiveController` 替换为实码（eagle_info.py、spec_utils.py、adaptive_runtime_state.py）
- [x] `21-Speculative-03-数据流与交互.md`：新增 §8「用户故事：一步投机的内心独白」
- [x] `21-Speculative-01-核心概念.md`：新增 §6 设计追问（accept rate 低 → 调步数/batch/关 spec）
- [x] `21-Speculative-04-关键问题.md`：新增 Q7 SGLang EAGLE vs vLLM spec decode 架构对比
- [x] 使用实码片段，Explain 段落完整
