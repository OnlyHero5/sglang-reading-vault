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
updated: 2026-07-10
---
# Attention · 学习检查

Checkpoint 不检查你看过多少源码段，而检查你能不能拿这套模型解释、排障、修改 attention 后端。

## 读者能做什么

- [ ] 能画出 `ServerArgs → ModelRunner → HybridAttnBackend → RadixAttention.forward → AttentionBackend.forward → forward_extend/decode` 主线。
- [ ] 能说清 `attention_backend`、`prefill_attention_backend`、`decode_attention_backend` 的覆盖关系。
- [ ] 能解释 `ForwardMode.EXTEND`、`DECODE`、`MIXED`、`IDLE`、`TARGET_VERIFY` 的后端选路差异。
- [ ] 能说明 `out_cache_loc`、`kv_indptr`、`kv_indices` 分别表示什么。
- [ ] 能解释为什么 CUDA Graph metadata 要拆成 out_graph 和 in_graph。
- [ ] 能指出 FlashInfer wrapper metadata 与 Triton `ForwardMetadata` 的差异。
- [ ] 能给出一个后端选错、Graph capture 失败、KV 写错 slot、DP merge fallback 的源码入口。

## 场景题

| 场景 | 合格回答要点 |
|------|--------------|
| 设置 `--prefill-attention-backend triton --decode-attention-backend flashinfer` 后启动 | `ServerArgs.get_attention_backends` 产出两个名字；`ModelRunner._get_attention_backend` 创建 `HybridAttnBackend`；prefill 走 Triton，decode 走 FlashInfer |
| speculative target verify 变慢 | 查 `HybridAttnBackend._select_backend` 和 `speculative_attention_mode`，不能只按 `TARGET_VERIFY` 属于 extend 来判断 |
| decode CUDA Graph capture 报 host sync | 查 backend 的 `init_forward_metadata_in_graph`，host/dynamic 操作应在 `init_forward_metadata_out_graph` |
| decode 第二步读不到第一步 token | 查 `forward_decode` 是否把本步 K/V 写入 `forward_batch.out_cache_loc`，再查 metadata 的 `kv_indices` |
| DP attention 下 merge state 从 FlashInfer 变 Triton | 查 `_safe_merge_state` 的 head 数上限，不一定是配置失效 |

## 可执行验证

| 验证 | 命令或动作 | 预期现象 |
|------|------------|----------|
| 文档源码引用 | `node maintenance/audit_source_evidence.mjs --note "sglang_reading/内存与Attention/Attention/SGLang-Attention-源码走读.md"` | 列出本专题引用的 upstream 文件，行号有效 |
| 双链 | `node maintenance/audit_wikilinks.mjs` | 本专题链接无断链 |
| 语法 | `python -m py_compile python/sglang/srt/layers/attention/base_attn_backend.py python/sglang/srt/layers/attention/hybrid_attn_backend.py python/sglang/srt/layers/radix_attention.py` | Python 语法可编译；依赖缺失另记 |
| 运行观测 | 启动时设置不同 prefill/decode backend | 日志出现 hybrid attention backend 和两个 resolved backend 名 |
| 性能对照 | 对比默认 decode CUDA Graph 与禁用 decode Graph | 如果 Graph metadata 有问题，禁用后错误形态或吞吐会明显变化 |

## 修改代码前的自检

- [ ] 新 backend 是否实现 eager metadata、out_graph metadata、in_graph metadata。
- [ ] extend 和 decode 是否都能写入新 KV，并读取正确历史 KV。
- [ ] 是否处理 SWA、cross-attention、target verify、piecewise CUDA Graph 的差异。
- [ ] 是否有最小测试或 profiling 能证明选路进入了预期 backend。
- [ ] 是否更新了日志或错误信息，让用户能看到 resolved backend。

## 复盘

Attention 后端的核心不是“FlashInfer vs Triton 谁更好”，而是 SGLang 如何把每个 batch 的运行事实编译成 kernel 可执行的参数。读懂这层之后，再读 [[SGLang-RadixAttention]] 和 [[SGLang-KV-Cache]]，你会更容易把 prefix cache、paged KV、kernel metadata 三件事分开。
