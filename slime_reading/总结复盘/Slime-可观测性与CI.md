---
title: "Slime 可观测性与 CI"
type: reference
framework: slime
topic: "总结复盘"
learning_role: reference
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/reference
  - source-reading
updated: 2026-07-10
---
# Slime 可观测性与 CI

这篇把 trace、profile、CI 和 fault tolerance 放在一起读。它们不是主训练路径的一部分，但决定了 Slime 这种长链路 RL 系统能否定位慢、错、挂、漂移。

## 1. Trace：Sample 级时间线

`trace_utils` 的基础字段包括 trace version、children key、SGLang token/latency/throughput 元数据，以及 PD prefill/decode 分段键。来源：slime/utils/trace_utils.py L16-L40

读法：

- trace 以 `Sample` 为载体，适合定位单条样本经历了哪些生成、RM、filter、agent 步骤。
- SGLang 返回的 PD 时序可以被拆成 prefill/decode 虚拟 lane。
- 自定义 rollout 或 reward 代码应复用 `trace_span`、`trace_event`、`trace_function`。

官方 trace 文档说明：用 `--save-debug-rollout-data` 保存含 trace 的 rollout dump，再用 `tools/trace_timeline_viewer.py` 打开 HTML 时间线。来源：docs/en/developer_guide/trace.md

## 2. Profile：训练侧性能与显存快照

`TrainProfiler` 根据 `profile_target` 决定是否创建 PyTorch profiler 和 memory profiler；`step(rollout_id)` 会推进 torch profiler，并在指定 rollout 停止 memory profiler。来源：slime/utils/profile_utils.py L13-L40

优先用法：

| 问题 | profile 方向 |
|------|--------------|
| actor train 慢 | `train_overall` 或 train actor 子目标 |
| logprob/value 慢 | train logprob/value 子路径 |
| 显存泄漏 | memory history 与 snapshot |
| rollout 慢 | 先用 trace，再看 SGLang profiling |

Profile 是训练侧工具；不要用它替代 sample-level trace。

## 3. Megatron Server：辅助入口，不是 RL 主循环

`megatron_server.py` 的 `main` 共享 `parse_args`，但追加 `add_megatron_server_arguments` 后调用 `launch(args)`。来源：slime/backends/megatron_utils/server/megatron_server.py L762-L770

这个文件适合补课：

- Megatron forward-only 或服务化调试。
- 非 SGLang rollout 场景。
- 训练 actor 与独立 server 入口的边界。

它不参与默认 `generate → train → update_weights` 主闭环，读者不应把它当训练入口。

## 4. CI：CPU 常驻 + GPU label-gated

CI 文档明确两层：每个 PR/push 都跑 CPU correctness；GPU e2e 通过 label 在 self-hosted GPU runner 上验证真实 Megatron + SGLang 路径。来源：docs/en/developer_guide/ci.md L3-L8

GPU job 的关键步骤包括安装 editable package、通过 `tests/ci/gpu_lock_exec.py --count <num_gpus>` 获取 GPU，再执行测试文件。来源：docs/en/developer_guide/ci.md L28-L31

| 层级 | 覆盖重点 |
|------|----------|
| CPU | arguments、plugin contracts、agent adapter、纯 Python shape contract |
| GPU | PPO/GRPO e2e、async、PD、checkpoint、SGLang + Megatron 联动 |

因此静态文档审计和 plugin contract 测试只能证明引用与接口边界，不能替代真实 GPU e2e。

## 5. Fault tolerance：恢复发生在权重更新前

RolloutManager 在 CI fault tolerance 场景下会尝试注入故障。来源：slime/ray/rollout.py L550-L551

训练 actor 的 `update_weights` 在 `use_fault_tolerance` 时，rank 0 先调用 `recover_updatable_engines`，随后用 gloo barrier 对齐。来源：slime/backends/megatron_utils/actor.py L587-L590

读法：

- fault tolerance 不是“任何地方自动恢复”，而是在特定边界做 engine recover。
- 权重更新前恢复 engine，是为了避免把新权重推给已失效或半恢复的 rollout engine。
- 排障时要同时看 RolloutManager 的 engine 管理和 Actor 的 update_weights。

## 6. Debug 模式速查

| 参数 | 作用 | 适合场景 |
|------|------|----------|
| `--debug-rollout-only` | 只跑 generate，不转 train data 或训练 | 看 SGLang 吞吐、RM、Sample 形状 |
| `--debug-train-only` | 跳过 generate，用已有 rollout data | 复现 loss、actor train、microbatch 问题 |
| `--save-debug-rollout-data` | 保存 rollout dump 和 trace | 事后打开时间线或重放 |
| `--load-debug-rollout-data` | 从 dump 重放训练 | 切断 rollout 随机性 |
| `--check-weight-update-equal` | 比对训练与 rollout 权重 | 排查同步错误 |

## 7. 验证建议

| 场景 | 入口 |
|------|------|
| Placement group | `tests/test_placement_group.py` |
| PPO e2e | `tests/test_qwen3_4B_ppo.py` |
| async rollout | `tests/test_qwen2.5_0.5B_async_short.py` |
| plugin 契约 | `tests/plugin_contracts/` |
| HF checkpoint saver | `tests/utils/test_hf_checkpoint_saver.py` |

实际能否运行取决于当前环境依赖和 GPU 可用性。文档迁移验收以 source evidence、wikilink 和专题残留为门禁；运行时正确性仍要靠对应测试层证明。

## 导航

- [[Slime-复杂度热点]]
- [[Slime-综合学习检查]]
