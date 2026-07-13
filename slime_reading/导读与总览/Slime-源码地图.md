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
updated: 2026-07-12
---

# Slime 源码地图

## 你为什么要读

这张地图回答“为了证明某个判断，第一批应该打开哪些文件”，而不是罗列所有目录。入口按读者任务组织：启动、资源、样本、训练、权重、异步、扩展、排障与测试。文件行数和固定 hook 数量会迅速漂移，不作为复杂度或完成度指标。

## 先用对象选择入口

| 手里的对象/现象 | 第一入口 | 下一跳 |
|-----------------|----------|--------|
| CLI 参数与最终 args 不一致 | `slime/utils/arguments.py` | SGLang/Megatron argument adapters、role YAML |
| GPU 排不到或 rank/GPU 错位 | `slime/ray/placement_group.py` | `actor_group.py`、`ray_actor.py`、`rollout_validation.py` |
| rollout server/router 起不来 | `slime/ray/rollout.py` | `sglang_config.py`、`sglang_engine.py`、`server_control.py` |
| prompt/buffer/epoch 漂移 | `slime/rollout/data_source.py` | `slime/utils/data.py`、prompt dataset |
| Sample 字段或 metadata 错 | `slime/utils/types.py` | rollout function、`ray/rollout.py` converter |
| DP rank 数据不均/OOM | `slime/utils/dp_schedule.py` | `ray/rollout.py::_split_train_data_by_dp`、Megatron data iterator |
| logprob/advantage/loss 异常 | `megatron_utils/actor.py` | `loss.py`、`data.py`、`model.py` |
| 下一轮仍像旧模型 | `actor.py::update_weights` | `update_weight/*`、`sglang_engine.py` |
| async 样本陈旧 | `train_async.py` | sample weight versions、fully-async example/rollout |
| custom hook 未生效 | `utils/arguments.py` | `utils/misc.py::load_function`、具体调用点 |

## 根入口：只负责节拍

| 文件 | 当前职责 | 不负责什么 |
|------|----------|------------|
| `train.py` | 同步 bootstrap、generate/train/update、save/eval/offload | rollout 算法、Megatron forward、SGLang scheduler |
| `train_async.py` | generation future 预取、更新间隔与在途生成屏障 | fully async 的全部生产消费协议 |
| `setup.py` | 包元数据与 packages | 训练 console entry point |

```python
# 来源：train.py L101-L103
if __name__ == "__main__":
    args = parse_args()
    train(args)
```

入口没有 `console_scripts` 抽象；日常运行从仓库根脚本进入。读主循环先看 [[Slime-业务流程]]，不要从 import 列表猜运行拓扑。

## 启动与参数

| 文件 | 关键所有权 |
|------|------------|
| `slime/utils/arguments.py` | pre-parse、namespace 合并、Slime normalize/validate、role args |
| `slime/backends/sglang_utils/arguments.py` | SGLang 参数解析/校验适配 |
| `slime/backends/megatron_utils/arguments.py` | Megatron parser/validator 适配 |
| `slime/ray/placement_group.py` | PG 总量、actor/rollout bundle views、模型组创建顺序 |
| `slime/ray/ray_actor.py` | 通用 Ray actor 基础能力 |
| `slime/ray/train_actor.py` | torch distributed 初始化与 rollout-manager 回连 |

阅读判断：CLI 只是输入，`slime_validate_args` 后的 args 才是大多数运行时消费者看到的事实；custom config 又可能在校验后期覆盖字段。

## Ray training group

`RayTrainGroup` 负责按 world rank 创建远程 workers，并把同一 rollout ref 列表扇出给所有 worker。它不根据 DP rank 重新切数据；DP rank-local 选择发生在 worker 内。

```python
# 来源：slime/ray/actor_group.py L131-L149
    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        """Do one rollout training. Returns a list of Ray refs (one per worker).

        For critics, each ref resolves to ``{"values": [cpu tensors...]}`` (or ``{}``
        for non-last-PP-stage workers). Actor refs resolve to ``None``.

        ``external_data`` may be a list (one item per worker) or a single dict
        broadcast to all workers.
        """
        if isinstance(external_data, list):
            assert len(external_data) == len(self._actor_handlers)
            return [
                actor.train.remote(rollout_id, rollout_data_ref, external_data=ed)
                for actor, ed in zip(self._actor_handlers, external_data, strict=False)
            ]
        return [
            actor.train.remote(rollout_id, rollout_data_ref, external_data=external_data)
            for actor in self._actor_handlers
        ]
```

## Rollout 编排与数据源

### RolloutManager 主文件

`slime/ray/rollout.py` 包含几类不同责任：

- `ServerGroup` / `RolloutServer`：多模型、PD/EPD、router、engine/offload/recovery；
- `RolloutManager`：动态加载 DataSource/rollout/eval/converter，generate/eval/debug；
- Sample→train_data、rollout 分组、DP schedule 与 Ray/NIXL 封装；
- rollout metrics、trace-derived SGLang performance 与 fault tolerance。

准备修改它时先确定自己碰的是 server 生命周期、数据变形还是指标；不要把整个文件当作一个“rollout function”。

### DataSource 契约

```python
# 来源：slime/rollout/data_source.py L17-L40
class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """
```

默认 `RolloutDataSource` 是只读 dataset source，`add_samples` 会报错；默认参数实际指向 `RolloutDataSourceWithBuffer`，后者优先消费 buffer，再回退全局 dataset。epoch、offset、group index 与 sample index 都属于可保存状态。

### Rollout function 族

| 文件 | 用途 |
|------|------|
| `slime/rollout/sglang_rollout.py` | 默认 SGLang rollout、reward/filter 主线 |
| `sglang_streaming_rollout.py` | streaming/partial 相关生成路径 |
| `fully_async_rollout.py` | fully async rollout 组件 |
| `sft_rollout.py` | SFT 数据生产 |
| `forge_load.py` | 保持 server 活跃时加载 Forge/debug 数据 |
| `on_policy_distillation.py` | SGLang teacher/OPD 相关路径 |
| `rm_hub/`、`filter_hub/` | reward/verifier 与动态过滤 |

## SGLang backend

| 文件 | 关键问题 |
|------|----------|
| `sglang_config.py` | 一个或多个 model/server group 如何描述；PD/EPD/placeholder |
| `sglang_engine.py` | engine actor 怎样启动 server、发 HTTP 控制、更新/检查权重 |
| `external.py` | 外部 engine 信息如何发现并写回 args |
| `server_control.py` | server 进程控制辅助 |
| `ray/rollout_validation.py` | group GPU indices 与配置一致性 |

跨到 SGLang 内部时再读 [[Slime与SGLang-阅读对照]]；Slime 只拥有 engine 编排和交接协议，不拥有 Scheduler/KV Cache 的实现。

## Megatron 训练

| 文件 | 关键所有权 |
|------|------------|
| `actor.py` | role 分流、参数快照切换、logprob/value/advantage、offload、发布入口 |
| `model.py` | model/optimizer 初始化、forward-only、train step 与多 step rollout |
| `loss.py` | response 对齐、CP、advantage/return、policy/value/SFT/custom loss |
| `data.py` | rank-local rollout data、DataIterator、batch 构造与日志 |
| `initialize.py` | Megatron 初始化与 parallel state |
| `cp_utils.py` | context-parallel 切片、gather 与 reducer |
| `routing_replay.py` / `utils/routing_replay.py` | MoE routing replay 接入 |
| `checkpoint.py`、`hf_checkpoint_saver.py` | Megatron 与 HF 保存/加载边界 |

模型构建还会进入 `model_provider.py`、Megatron bridge 与具体模型扩展；遇到 shape/权重键问题不要只停在 actor.py。

## 权重发布

目录：`slime/backends/megatron_utils/update_weight/`

| 文件 | 模式 |
|------|------|
| `update_weight_from_tensor.py` | colocate / CUDA IPC tensor |
| `update_weight_from_distributed.py` | full + NCCL collective |
| `update_weight_from_disk.py` | full + disk checkpoint |
| `update_weight_from_disk_delta.py` | delta + disk apply |
| `common.py` | 公共参数/engine group 辅助 |
| `hf_weight_iterator_base.py` | iterator 抽象 |
| `hf_weight_iterator_direct.py` | raw/direct Megatron→HF mapping |
| `hf_weight_iterator_bridge.py` | bridge mapping |

选择 updater 的代码在 `actor.py::init`，不是在 `train.py`。发布卡住时同时检查 RolloutManager 返回的 updatable engine 集合、engine lock、process-group reconnect 与目标 engine API。

## Agent、扩展与可观测

| 目录/文件 | 入口 |
|-----------|------|
| `slime/agent/trajectory.py` | 对话树、turn、trajectory→samples |
| `slime/agent/adapters/` | provider API 适配 |
| `slime/agent/harness/` | Claude Code/Codex 等 harness |
| `slime_plugins/rollout_buffer/` | generator discovery 与 group buffer |
| `slime/utils/trace_utils.py` | Sample trace carrier、span/event、SGLang timing 子 span |
| `slime/utils/profile_utils.py` | training profiler/memory snapshot |
| `examples/` | full async、search、multi-agent 与模型配置实例 |

动态 hook 的完整入口从 `arguments.py` 搜索 `-path`，再追 `load_function` 调用点；不要维护固定数量。

## 测试地图：按契约找证据

| 要验证什么 | 当前代表测试 |
|------------|--------------|
| PG 与 server group GPU 映射 | `test_placement_group.py`、`test_rollout_validation.py`、`test_sglang_config_mixed_offload*.py` |
| Sample/DP/loss reducer | `test_sample.py`、`test_dp_schedule.py`、`test_loss_cp_invariance.py` |
| async/fully async | `test_qwen2.5_0.5B_async_short.py`、`test_qwen2.5_0.5B_fully_async_short.py` |
| external/PD/multi-config | `test_external_sglang_engines.py`、`test_qwen3_4B_external_pd.py`、`test_qwen2.5_0.5B_sglang_config*.py` |
| weight update | `test_empty_colocated_weight_bucket.py`、`test_full_disk_weight_update.py` |
| plugin contracts | `tests/plugin_contracts/` |
| Agent trajectory | `tests/test_agent/` |
| trace/metrics | `tests/utils/test_trace_utils.py`、`test_rollout_metrics.py`、`test_metric_report*.py` |

端到端测试名通常包含具体模型和硬件假设，不能把“文件存在”当成本机可运行证明。先读测试参数与 skip/fixture，再选择目标环境。

## 导航

- [[Slime-架构分层]]：按责任与生命线理解系统。
- [[Slime-模块依赖图]]：区分 import、创建、远程调用、数据和状态依赖。
- [[Slime-业务流程]]：按六道门理解时序。
- [[Slime-RL训练全链路]]：深入同步 baseline 的对象生命周期。
