---
title: "Slime 学习路径"
type: guide
framework: slime
topic: "导读与总览"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/guide
  - source-reading
updated: 2026-07-13
---
# Slime 学习路径

> 按读者任务与运行时责任组织的学习路线。源码锚点用于定位，不替代专题全文。

建议顺序：先补 Ray/Megatron 直觉，再用同步 baseline 建立版本清晰的闭环；随后分别深挖资源、样本、训练与权重发布，最后才进入流水异步、fully async、Agent 和插件。

## 长文读法

这篇不要求一次读完全部主题，而是让不同读者按目标选路线：先修读者先补 Ray/Megatron 直觉；排障读者从参数、PG、RolloutManager 或训练一步切入；想改扩展点的读者最后再看定制接口、Agent 和插件生态。

| 读者任务 | 先读 | 要抓住的判断 |
|----------|------|--------------|
| 第一次读 Slime | Ray/Megatron 先修 → 项目愿景 → 训练入口 | 先建立 PG → RolloutManager → Actor → 首次权重推送的启动顺序 |
| 排查资源和参数 | 参数中枢 → Ray GPU 编排 | 参数 parse/validate 决定最终 args，PG 只消费最终资源事实 |
| 排查 rollout 数据 | RolloutManager → 默认 Rollout 路径 | RolloutManager 是远程数据生产边界，默认 rollout 负责生成、RM、过滤和输出样本 |
| 排查训练闭环 | SGLang 引擎 → Megatron 训练 → RL Loss → 权重更新 | 生成、训练、loss/advantage、update_weights 串成 RL 闭环 |
| 做自定义或 Agent | 定制接口与 Agent → 示例与插件生态 | `load_function`、trajectory、rollout buffer 和插件是扩展入口 |
| 回到专题深读 | 文末导航和主题双链 | 这页只给路线选择，源码细节要进入对应专题 |

读的时候先选目标再进入对应主题。入口页负责定位，不替代专题源码走读。

## 先选运行模式

同一个函数名在不同主循环里可能承担不同的一致性责任。进入专题前先确定自己读的是哪种模式：

| 模式 | 入口 | 适合先研究什么 | 暂时不要假设什么 |
|------|------|----------------|------------------|
| 同步 baseline | `train.py` | 对象生命周期、critic 分支、offload、每轮权重屏障 | 不要把所有输出都简化成 PPO/GRPO 固定字段 |
| 流水异步 | `train_async.py` | future 预取、生成/训练重叠、更新间隔、策略陈旧度 | 不要假设 rollout 永远使用刚训练出的最新权重，也不要启用 colocate |
| fully async | `examples/full_async` | buffer、生产消费速率、版本窗口、旧样本策略 | 不要把它当成 `train_async.py` 的同义词 |

首次学习固定在同步 baseline，直到你能证明“某条 sample 由哪版权重生成、由哪轮训练消费、何时发布下一版权重”。

---

## Ray / Megatron 零基础先修

**目标：** 理解 Ray 如何预订 GPU 与远程调用 Actor，Megatron 如何把模型切到多卡训练。

**阅读：** [[Slime-零基础先修]]

**源码锚点：**

```python
# 定位骨架（非逐行摘录）：来源 train.py L11-L20
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
```

**读法：** 先修文档只建立直觉：Ray 管资源与远程进程，Megatron 管训练并行。源码深读从 Ray GPU 编排与 Megatron 训练一步开始。

---

## 项目边界与四类责任

**目标：** 理解 Slime 解决什么问题，并区分资源编排、样本生产、训练消费和权重发布。

**阅读：** [[Slime-阅读方法-核心概念]]

**源码锚点：**

```python
# 定位骨架（非逐行摘录）：来源 train.py L62-L81
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
        actor_model.update_weights()
```

---

## 训练入口 train.py

**目标：** bootstrap 顺序：PG → RolloutManager → Actor → 首次 update_weights。

**阅读：** [[Slime-训练主循环-源码走读]]

**源码锚点：**

```python
# 定位骨架（非逐行摘录）：来源 train.py L9-L32
def train(args):
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())
    actor_model.update_weights()
    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())
```

---

## 参数中枢 arguments

**目标：** 三阶段 parse；colocate/offload；`*-path` 定制入口。

**阅读：** [[Slime-Ray参数-源码走读]] · [[Slime-训练与Rollout参数-源码走读]]

**源码锚点：**

```python
# 定位骨架（非逐行摘录）：来源 slime/utils/arguments.py L1546-L1559
def parse_args(add_custom_arguments=None):
    configure_logger()
    add_slime_arguments = get_slime_extra_args_provider(add_custom_arguments)
    pre = _pre_parse_mode()
    skip_sglang = pre.debug_train_only or pre.load_debug_rollout_data is not None
    sglang_ns = None
    if not skip_sglang:
        sglang_ns = sglang_parse_args()
```

---

## Ray GPU 编排

**目标：** PlacementGroup 拆分 train/rollout/critic；RayTrainGroup `.remote()` API。

**阅读：** [[Slime-PlacementGroup-源码走读]] · [[Slime-RayTrainGroup-源码走读]]

**源码锚点：**

```python
# 来源：slime/ray/placement_group.py L15-L18
@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]
```

**读法：** InfoActor 探测各节点 GPU 拓扑，用于 `sort_key` 重排 PG bundle 顺序。

---

## RolloutManager.generate

**目标：** `_get_rollout_data` → convert → split by DP。

**阅读：** [[Slime-RolloutManager-源码走读]]

**源码锚点：**

```python
# 定位骨架（非逐行摘录）：来源 slime/ray/rollout.py L546-L559
    def generate(self, rollout_id):
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        if self.args.debug_rollout_only:
            return
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data)
```

---

## 默认 Rollout 路径

**目标：** DataSource 取 prompt → sglang_rollout HTTP 生成 → RM 打分。

**阅读：** [[Slime-数据源-源码走读]] · [[Slime-SGLang-Rollout-源码走读]]

**源码锚点：**

```python
# 定位骨架（非逐行摘录）：来源 slime/rollout/sglang_rollout.py L618-L632
def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    assert args.rollout_global_dataset
```

---

## SGLang 引擎与权重推送

**目标：** engine 启动、NCCL group、update_weights reload。

**阅读：** [[Slime-SGLang-Engine-源码走读]] · [[Slime-引擎拓扑-源码走读]]

**SGLang 交叉：** [[SGLang-HTTP-Server]] · [[SGLang-PD分离]]

**源码锚点：**

```python
# 定位骨架（非逐行摘录）：来源 slime/backends/sglang_utils/sglang_engine.py L52-L77
def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    if getattr(server_args, "encoder_only", False):
        from sglang.srt.disaggregation.encode_server import launch_server_process as sglang_launch_server_process
        return sglang_launch_server_process(
            server_args,
            start_method="spawn",
            wait_for_server=True,
        )
    from sglang.srt.entrypoints.http_server import launch_server
    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()
    if getattr(server_args, "node_rank", 0) != 0:
        return p
    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )
    return p
```

```python
# 定位骨架（非逐行摘录）：来源 slime/backends/sglang_utils/sglang_engine.py L464-L488
    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=False,
        weight_version: str | None = None,
        load_format: str | None = None,
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        if load_format is not None:
            payload["load_format"] = load_format
        return self._make_request("update_weights_from_distributed", payload)
```

**读法：** `launch_server_process` spawn 子进程启动 SGLang HTTP server；训练侧 NCCL broadcast 完成后由 `update_weights_from_distributed` 触发 engine reload。

---

## Megatron 训练一步

**目标：** `train_actor` → `train_one_step` → loss backward。

**阅读：** [[Slime-Megatron-Actor初始化-源码走读]] · [[Slime-训练步骤-源码走读]]

**源码锚点：**

```python
# 定位骨架（非逐行摘录）：来源 slime/backends/megatron_utils/actor.py L380-L394
    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
        if self.args.offload_train:
            self.wake_up()
        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)
        if self.role == "critic":
            result = self.train_critic(rollout_id, rollout_data)
        else:
            self.train_actor(rollout_id, rollout_data, external_data=external_data)
```

---

## RL Loss 与 Advantage

**目标：** log_probs → advantages → policy_loss。

**阅读：** [[Slime-Advantage计算-源码走读]] · [[Slime-训练数据-源码走读]]

**源码锚点：**

```python
# 定位骨架（非逐行摘录）：来源 slime/backends/megatron_utils/loss.py L661-L669
def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.
    Supported methods: "grpo", "gspo", "cispo", "ppo",
    "reinforce_plus_plus", and "reinforce_plus_plus_baseline".
    """
```

---

## update_weights 闭环

**目标：** megatron_to_hf → broadcast → SGLang reload。

**阅读：** [[Slime-分布式权重同步-源码走读]] · [[Slime-Megatron到HF转换-源码走读]]

**源码锚点：**

```python
# 来源：slime/backends/megatron_utils/actor.py L583-L586
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return
```

---

## 定制接口与 Agent

**目标：** 理解 `--*-path` 动态加载边界，以及 TrajectoryManager 如何保存多轮轨迹；具体 hook 数量以当前参数定义为准，不把易变计数当接口契约。

**阅读：** [[Slime-自定义扩展-源码走读]] · [[Slime-Agent轨迹-源码走读]]

**源码锚点：**

```python
# 来源：slime/utils/misc.py L37-L45
def load_function(path):
    """
    Load a function from a module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
```

```python
# 定位骨架（非逐行摘录）：来源 slime/utils/arguments.py L327-L339
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="slime.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "You should use this model to create your own custom rollout function, "
                    "and then set this to the path of your custom rollout function. "
                    "The signature of the function should be "
                    "`def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput`"
                ),
            )
```

```python
# 定位骨架（非逐行摘录）：来源 slime/agent/trajectory.py L283-L305
    def record_turn(
        self,
        sid: str,
        *,
        turn: TurnRecord,
        prompt_messages: list[dict[str, Any]],
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not prompt_messages:
            logger.warning("record_turn(sid=%s): empty prompt_messages; skipping", sid)
            return
        root = self._trees.setdefault(sid, MessageNode())
        node, depth = self._find_mount_point(root, prompt_messages)
        node, depth = self._try_merge_assistant_rewrite(sid, node, prompt_messages, depth)
        node = self._mount_prompt_messages(node, prompt_messages[depth:])
        self._attach_assistant_leaf(sid, node, turn=turn, response_message=response_message, metadata=metadata)
```

**读法：** `--*-path` 参数经 `load_function` 动态 import；Agent 多轮场景由 `TrajectoryManager.record_turn` 累积对话树，最终 `get_trajectory` 转为 `list[Sample]`。

---

## 示例与插件生态

**目标：** rollout_buffer、search-r1、multi_agent 接入模式。

**阅读：** [[Slime-插件与示例-源码走读]]

**源码锚点：**

```python
# 定位骨架（非逐行摘录）：来源 slime_plugins/rollout_buffer/buffer.py L54-L103
def discover_generators():
    """
    Automatically discover generator modules in the generator directory.
    Returns a dictionary mapping task_type to module with run_rollout function.
    """
    generator_map = {}
    generator_dir = pathlib.Path(__file__).parent / "generator"
    for file_path in glob.glob(str(generator_dir / "*.py")):
        if file_path.endswith("__init__.py"):
            continue
        try:
            spec = importlib.util.spec_from_file_location("generator_module", file_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "TASK_TYPE") or not hasattr(module, "run_rollout"):
                continue
            task_type = module.TASK_TYPE
            generator_info = {
                "module": module,
                "file_path": file_path,
                "run_rollout": module.run_rollout,
            }
            generator_map[task_type] = generator_info
        except Exception as e:
            print(f"Error loading generator from {file_path}: {str(e)}")
            continue
    return generator_map
```

```python
# 来源：slime_plugins/rollout_buffer/buffer.py L125-L138
class BufferQueue:
    def __init__(
        self,
        group_size,
        task_type="math",
        transform_group_func=None,
        is_valid_group_func=None,
        get_group_data_meta_info_func=None,
    ):
        self.data = {}
        self.temp_data = {}
        self.group_timestamps = {}
        self.group_size = group_size
        self.task_type = task_type
```

**读法：** `rollout_buffer` 插件通过 `TASK_TYPE` 自动发现 generator；`BufferQueue` 按 group 聚合样本后供训练消费，是 examples/search-r1 等场景的参考接入模式。

---

## 主题索引表

| 主题 | 文档 |
|------|------|
| Ray/Megatron 先修 | [[Slime-零基础先修]] |
| 三角架构 | [[Slime-阅读方法]] |
| train.py 主循环 | [[Slime-训练主循环]] |
| 参数系统 | [[Slime-Ray参数]] · [[Slime-训练与Rollout参数]] |
| Ray 编排 | [[Slime-PlacementGroup]] · [[Slime-RayTrainGroup]] |
| RolloutManager | [[Slime-RolloutManager]] |
| 默认 Rollout | [[Slime-SGLang-Rollout]] |
| SGLang 引擎 | [[Slime-SGLang-Engine]] |
| Megatron train | [[Slime-训练步骤]] |
| RL Loss | [[Slime-Advantage计算]] · [[Slime-Policy-Loss]] |
| update_weights | [[Slime-分布式权重同步]] · [[Slime-磁盘权重同步]] |
| Agent/定制 | [[Slime-Agent轨迹]] · [[Slime-自定义扩展]] |
| 插件生态 | [[Slime-插件与示例]] |

---

## 每一阶段怎样验收

| 阶段 | 通过标准 | 可执行或静态验证 |
|------|----------|------------------|
| 启动与资源 | 能从参数推导 actor/rollout/critic 的 PG slice 与 colocate 关系 | 给 `_get_placement_group_layout` 构造 debug、external、colocate 参数表并核对返回值 |
| Rollout | 能解释 `Sample` 的 tokens、mask、reward、rollout id、weight version 在何处形成 | 保存一轮 debug rollout data，检查字段长度和版本 metadata |
| DP 调度 | 能解释为何先按 rollout 保组，再 pack micro-batch | 对 `build_dp_schedule` 输入不同长度与 sibling rollout id，观察 partitions 与 indices |
| 训练 | 能区分 rollout/current/ref logprob 与 critic value | 沿 `train_actor` 的配置分支列出每个字段的生产者 |
| 权重发布 | 能说明所选 updater 如何限制半更新可见性，以及失败后可能留下什么状态 | 先选 distributed、disk、tensor colocate 或 external 路径，再静态追踪其 writer/lock、payload、cache/commit 与版本点；可运行时核对所有目标 engine 的 version |
| 异步 | 能画出 sample version 与 actor version 的时间线 | 对 `update_weights_interval > 1` 手工推演每轮 generation 使用的最近已发布版本 |

如果环境无法启动 Ray、SGLang 或多卡 Megatron，静态验证必须写出输入、调用链、预期结果和无法运行的环境限制；只写“阅读源码可知”不算完成。

---

## 导航

- [[Slime-RL训练全链路]] — 学习路线的时序扩展版
- [[Slime-综合学习检查]] — 自测清单
