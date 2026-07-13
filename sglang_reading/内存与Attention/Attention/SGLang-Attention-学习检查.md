---
title: "Attention · 学习检查"
type: exercise
framework: sglang
topic: "Attention"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-11
---
# Attention · 学习检查

Checkpoint 不检查你看过多少源码段，而检查你能不能拿这套模型解释、排障、修改 attention 后端。

## 读者能做什么

- [ ] 能画出 `raw flags → post-init resolver → registry factory → linear/per-mode/TBO wrapper → metadata planner → RadixAttention → kernel` 主线。
- [ ] 能区分原始 flag、归一后的三个 backend 字段、最终两个 per-mode 名称与实际对象树。
- [ ] 能解释全部 `ForwardMode`，并指出 `DRAFT_EXTEND_V2`、`PREBUILT`、`SPLIT_PREFILL`、`DLLM_EXTEND` 不能怎样被机械归类。
- [ ] 能说明 `ForwardBatch` 为什么是可变执行视图，以及 DP padding、inner view、piecewise 裁剪如何影响 metadata。
- [ ] 能说明 generic `out_cache_loc`、`KVWriteLoc.swa_loc/full_loc`、`kv_indptr`、`kv_indices` 分别表示什么。
- [ ] 能解释为什么 CUDA Graph metadata 要拆成 out_graph 和 in_graph。
- [ ] 能解释 capture-stable buffer 为什么要原地刷新，而不是在 graph 外随意换 tensor。
- [ ] 能指出 FlashInfer eager/wrapper/ragged-paged merge 与 Triton `ForwardMetadata` 的差异。
- [ ] 能解释 `forward_metadata_ready`、计划 shape 与 `replan_equivalent` 的所有权契约。
- [ ] 能给出一个后端选错、Graph capture 失败、KV 写错 slot、DP merge fallback 的源码入口。

## 场景题

| 场景 | 合格回答要点 |
|------|--------------|
| 设置不同 per-mode backend 后启动 | 先经过兼容性归一；registry 分别建子对象并各自 wrapper；两个 resolved 名不同才组成 per-mode Hybrid；外层仍可能有 TBO/PDMux |
| speculative target verify 变慢 | 查 `HybridAttnBackend._select_backend` 和 `speculative_attention_mode`，不能只按 `TARGET_VERIFY` 属于 extend 来判断 |
| decode CUDA Graph capture 报 host sync | 查 `init_forward_metadata_in_graph`；host/dynamic 操作放图外，同时检查 replay 是否原地刷新固定 buffer |
| Unified 下 decode 第二步读不到第一步 token | 不能只查 `out_cache_loc`；要核对 generic→physical 翻译、`KVWriteLoc`、pool 类型与读取索引流 |
| DP attention 下 merge state 从 FlashInfer 变 Triton | 查 `_safe_merge_state` 的 head 数上限；说明只替换 merge helper，不是整个 backend |
| multi-step draft 在 DP padding 后输出异常 | 查 metadata 由谁预计划、计划 shape 是否 stale、是否声明 `replan_equivalent`，不能无条件 replan |
| 修改第一个 `HybridAttnBackend.forward` 不生效 | Python 后定义覆盖前定义；检查类末尾和 wrapper 后的运行时对象 |

## 可执行验证

| 验证 | 命令或动作 | 预期现象 |
|------|------------|----------|
| 文档源码引用 | `node maintenance/audit_source_evidence.mjs --note "sglang_reading/内存与Attention/Attention/SGLang-Attention-源码走读.md"` | 列出本专题引用的 upstream 文件，行号有效 |
| 双链 | `node maintenance/audit_wikilinks.mjs` | 本专题链接无断链 |
| 语法 | 在 `sglang/` 下运行 `python -m py_compile python/sglang/srt/layers/attention/base_attn_backend.py python/sglang/srt/layers/attention/hybrid_attn_backend.py python/sglang/srt/layers/radix_attention.py` | 三个文件通过语法编译；这不等于运行依赖与 GPU kernel 可用 |
| 运行观测 | 启动时设置不同 prefill/decode backend | 日志出现 hybrid attention backend 和两个 resolved backend 名 |
| Graph 定位实验 | 固定模型、硬件、backend、batch/context 与输入，对比 Graph on/off | 记录错误是否转为 eager、输出是否变化；吞吐方向由实测决定，变化本身不是 metadata bug 的充分证据 |

## 修改代码前的自检

- [ ] 新 backend 是否实现或明确继承 eager、out_graph、in_graph metadata 契约，是否支持预计划场景。
- [ ] 每个声明支持的 mode 是否把 generic KV 地址翻译到正确 pool，并读取正确的本轮 KV index stream。
- [ ] 是否处理或尽早拒绝 Unified、SWA、cross-attention、target verify、draft、DP padding、piecewise Graph。
- [ ] 是否有最小测试或 profiling 能证明选路进入了预期 backend。
- [ ] 是否更新了日志或错误信息，让用户能看到 resolved backend。

## 复盘

Attention 后端的核心不是“FlashInfer vs Triton 谁更好”，而是 SGLang 如何把每个 batch 的运行事实编译成 kernel 可执行的参数。读懂这层之后，再读 [[SGLang-RadixAttention]] 和 [[SGLang-KV-Cache]]，你会更容易把 prefix cache、paged KV、kernel metadata 三件事分开。
