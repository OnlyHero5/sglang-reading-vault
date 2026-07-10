---
title: "Slime 源码地图"
type: reference
framework: slime
topic: "导读与总览"
learning_role: reference
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/reference
  - source-reading
updated: 2026-07-10
---
# Slime 源码地图

本页是 Slime 源码入口地图，用来在读专题前把 `train.py`、Ray actor、rollout、SGLang 后端、Megatron 训练和权重同步文件先定位清楚。读完后，你应该能从一个 RL 训练步骤反查到对应源码目录。

> slime/ 顶层与核心子目录 · 按运行层定位源码入口

---

## 根目录入口

| 文件 | 职责 | 代码锚点 |
|------|------|----------|
| `train.py` | 同步 RL 主循环 | `train(args)` |
| `train_async.py` | 异步 prefetch generate | `train_async(args)` |
| `setup.py` | 包定义 | `name="slime"` |
| `requirements.txt` | 依赖清单 | — |

**源码锚点：**

```python
## 来源：train.py L101-L103
if __name__ == "__main__":
    args = parse_args()
    train(args)
```

---

## 入口与编排

| 文件 | 职责 |
|------|------|
| `slime/utils/arguments.py` | Megatron+Slime+Ray 参数；`parse_args()` |
| `slime/utils/misc.py` | `load_function`, `should_run_periodic_action` |
| `slime/ray/placement_group.py` | `create_placement_groups`, `create_rollout_manager` |
| `slime/ray/actor_group.py` | `RayTrainGroup`：async_train, update_weights |
| `slime/ray/train_actor.py` | `TrainRayActor` 基类 |
| `slime/ray/rollout.py` | `RolloutManager`（1486 行热点） |

```python
## 来源：slime/ray/actor_group.py L1-L15（节选）
class RayTrainGroup:
    """A group of Ray actors that run training."""
    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        return self._run_async("train", rollout_id, rollout_data_ref, external_data=external_data)
```

---

## Rollout 生成

| 文件 | 职责 |
|------|------|
| `slime/rollout/sglang_rollout.py` | 默认 `generate_rollout` |
| `slime/rollout/data_source.py` | `RolloutDataSource` |
| `slime/rollout/base_types.py` | `RolloutFnTrainOutput`, `call_rollout_fn` |
| `slime/rollout/rm_hub/` | Reward model / verifier |
| `slime/rollout/filter_hub/` | Dynamic sampling filters |
| `slime/rollout/fully_async_rollout.py` | 全异步 rollout |
| `slime/utils/types.py` | `Sample`, `RolloutBatch` |

```python
## 来源：slime/rollout/data_source.py L1-L20（节选）
class RolloutDataSource:
    """Provides prompts for rollout and stores completed samples."""
```

---

## SGLang 后端

| 文件 | 职责 |
|------|------|
| `slime/backends/sglang_utils/sglang_engine.py` | `SGLangEngine`, weight reload |
| `slime/backends/sglang_utils/sglang_config.py` | `SglangConfig`, PD 拓扑 |
| `slime/backends/sglang_utils/arguments.py` | `--sglang-*` 透传 |
| `slime/backends/sglang_utils/external.py` | 外部 engine 发现 |
| `slime/backends/sglang_utils/server_control.py` | 进程控制 |

---

## Megatron 训练

| 文件 | 职责 |
|------|------|
| `slime/backends/megatron_utils/actor.py` | `MegatronTrainRayActor` |
| `slime/backends/megatron_utils/model.py` | `train_one_step`, forward |
| `slime/backends/megatron_utils/model_provider.py` | 模型构建 |
| `slime/backends/megatron_utils/loss.py` | advantage + policy loss（1322 行） |
| `slime/backends/megatron_utils/data.py` | `get_batch`, iterator |
| `slime/backends/megatron_utils/initialize.py` | Megatron init |
| `slime/backends/megatron_utils/cp_utils.py` | Context Parallel |
| `slime/backends/megatron_utils/routing_replay.py` | MoE routing replay |

---

## 权重同步

| 文件 | 职责 |
|------|------|
| `update_weight/update_weight_from_distributed.py` | NCCL broadcast |
| `update_weight/update_weight_from_disk.py` | 磁盘全量 |
| `update_weight/update_weight_from_disk_delta.py` | Delta patch |
| `update_weight/update_weight_from_tensor.py` | Colocate tensor |
| `megatron_to_hf/__init__.py` | `convert_to_hf` 路由 |
| `checkpoint.py` | 训练 checkpoint |
| `hf_checkpoint_saver.py` | HF 格式保存 |

---

## Agent 与定制

| 文件 | 职责 |
|------|------|
| `slime/agent/trajectory.py` | `TrajectoryManager` |
| `slime/agent/adapters/` | OpenAI/Anthropic 适配 |
| `slime/agent/harness/` | Claude Code / Codex harness |
| `docs/en/get_started/customization.md` | 17 类接口文档 |

---

## 扩展与运维

| 文件 | 职责 |
|------|------|
| `slime_plugins/rollout_buffer/buffer.py` | 插件 buffer |
| `examples/search-r1/` | 搜索增强 RL 示例 |
| `tools/convert_hf_to_torch_dist.py` | HF→Megatron 权重 |
| `slime/utils/trace_utils.py` | Sample 级 trace |
| `slime/utils/profile_utils.py` | TrainProfiler |
| `docs/en/developer_guide/ci.md` | CI 说明 |

---

## tests/ 快速索引

| 测试 | 覆盖 |
|------|------|
| `tests/test_placement_group.py` | PG 分配 |
| `tests/test_qwen3_4B_ppo.py` | PPO 端到端 |
| `tests/test_qwen2.5_0.5B_async_short.py` | 异步训练 |
| `tests/plugin_contracts/` | customization 契约 |
| `tests/test_agent/` | Agent trajectory |

---

## 导航

- [[Slime-架构分层]]
- [[Slime-模块依赖图]]
