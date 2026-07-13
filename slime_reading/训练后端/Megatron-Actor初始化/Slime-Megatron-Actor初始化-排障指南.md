---
title: "Megatron-Actor初始化 · 排障指南"
type: troubleshooting
framework: slime
topic: "Megatron-Actor初始化"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-13
---
# Megatron-Actor初始化 · 排障指南

## 读者任务

这篇按症状排障。每个问题都给出源码入口和验证抓手，避免把 Ray 创建失败、distributed 卡住、模型加载错误、权重同步错误混成一类“init 坏了”。

## 速查表

| 症状 | 优先看 | 可能原因 | 验证抓手 |
|------|--------|----------|----------|
| init 立刻返回且无 checkpoint 日志 | `actor.py` early return | 开了 `debug_rollout_only` | 返回值应为 `0` |
| actor 创建时报 `torch_memory_saver` 动态库 | `actor_group.py` runtime env | offload 动态库缺失 | 报错发生在 actor 创建而非 checkpoint load |
| 所有 rank 卡在 init | `train_actor.py` / `initialize.py` | distributed env 或 Megatron 并行配置不一致 | 看 `MASTER_*`、`RANK`、backend、timeout |
| `start_rollout_id` assert 失败 | `placement_group.py` | rank 加载了不同 checkpoint iteration | 打印各 rank 返回值 |
| colocate + delta 直接 assert | `actor.py` updater 选型 | colocate 只能 full tensor 推权 | 改 full 或关闭 colocate |
| init 成功但 update_weights 卡住 | `actor.py` `update_weights` | 连接 rollout engines 或 updater 通道问题 | 转到 [[Slime-分布式权重同步]] |
| NumPy 2.x 失败后重试又报 group 已初始化 | `initialize.py` 断言时序 | 子组先创建、后断言且无清理 | 修环境后重建 Ray actor |
| 辅助 checkpoint 失败后主 load 参数变了 | `load_other_checkpoint` | 临时 args 无 finally 恢复 | 销毁 actor，不要继续原地加载 |
| train/save/update 异常后显存未让出 | wake/sleep 非事务 | 末尾 sleep 被跳过 | 检查 groups/TMS/updater 后重建 actor |

## Q1：为什么 `debug_rollout_only` 下没有 Megatron 模型？

因为这是 `MegatronTrainRayActor.init` 的第一条分支。它只保存 `args` 并返回 `0`，不会调用父类 distributed init，也不会调用 Megatron init。

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/actor.py L46-L62
class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(...):
        if args.debug_rollout_only:
            self.args = args
            return 0

        monkey_patch_torch_dist()
        super().init(args, role, with_ref, with_opd_teacher)

        init(args)
```

参数解析还会关闭训练 offload，并禁止同时打开 `debug_train_only`：

```python
# 定位骨架（非逐行摘录）：slime/utils/arguments.py L1866-L1883
if args.debug_rollout_only:
    ...
    args.colocate = False
    args.offload_train = args.offload_rollout = False
    if args.train_memory_margin_bytes > 0:
        logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
        args.train_memory_margin_bytes = 0

assert not (args.debug_rollout_only and args.debug_train_only), (
    "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
)
```

验证：如果要调 Megatron checkpoint、模型 provider、offload 或 weight updater，不要用 `debug_rollout_only`。

## Q2：为什么 init 里看起来有两次 distributed 初始化？

它们不是同一层：

| 调用 | 作用 |
|------|------|
| `dist.init_process_group` | 建立所有 train actors 的 PyTorch world |
| `mpu.initialize_model_parallel` | 在 world 上切 Megatron TP/PP/DP/CP/EP 子组 |

```python
# 来源：slime/ray/train_actor.py L61-L70
backend = args.distributed_backend

dist.init_process_group(
    backend=backend,
    timeout=timedelta(minutes=args.distributed_timeout_minutes),
)
init_gloo_group()

args.rank = dist.get_rank()
args.world_size = dist.get_world_size()
```

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/initialize.py L37-L53
mpu.initialize_model_parallel(
    args.tensor_model_parallel_size,
    args.pipeline_model_parallel_size,
    args.virtual_pipeline_model_parallel_size,
    ...
    create_gloo_process_groups=args.enable_gloo_process_groups,
)
```

验证：如果卡在 world group，看 Ray rank/env/backend；如果 world group 成功但 Megatron shape 或通信错，看 TP/PP/DP/CP/EP 配置。

## Q3：为什么 critic 没有 `weight_updater`？

critic 训练 value，不向 SGLang rollout engines 推 actor 权重。源码在模型初始化后、actor 权重备份前就对 critic 提前返回。

```python
# 来源：slime/backends/megatron_utils/actor.py L101-L106
start_rollout_id = loaded_rollout_id + 1

if role == "critic":
    if self.args.offload_train:
        self.sleep()
    return start_rollout_id
```

验证：critic 仍会走 `initialize_model_and_optimizer`，所以 checkpoint/model 错误仍可能发生；但找不到 `self.weight_updater` 对 critic 是预期行为。

## Q4：`start_rollout_id` 从哪里来，为什么所选 rank 组要一致？

`initialize_model_and_optimizer` 加载 checkpoint 后返回 `loaded_rollout_id`；actor init 返回下一轮起点 `loaded_rollout_id + 1`。driver 聚合当前选中 role 的 rank 返回值并要求组内一致。

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/actor.py L83-L106
self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
    args, role
)
...
start_rollout_id = loaded_rollout_id + 1
...
return start_rollout_id
```

```python
# 定位骨架（非逐行摘录）：slime/ray/placement_group.py L199-L208
if args.use_critic:
    start_rollout_ids = critic_start_rollout_ids
else:
    start_rollout_ids = actor_start_rollout_ids

assert len(set(start_rollout_ids)) == 1

if args.start_rollout_id is None:
    args.start_rollout_id = start_rollout_ids[0]
```

验证：assert 失败时先打印被选择的 role 每个 rank 返回值，再看 checkpoint 路径与 `ckpt_step`。使用 critic 时源码只检查 critic ranks，不比较 actor/critic；若二者语义可能不同，应显式设置并外部校验 `start_rollout_id`。

## Q5：为什么 colocate 不能配 delta 推权？

colocate 表示训练与推理共享本地 GPU 资源，源码直接选择 `UpdateWeightFromTensor`，且要求 full 模式。delta 模式要求 disk transport。

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/actor.py L139-L161
if self.args.colocate:
    assert (
        self.args.update_weight_mode == "full"
    ), "--update-weight-mode=delta is not supported with --colocate"
    update_weight_cls = UpdateWeightFromTensor
elif self.args.update_weight_mode == "delta":
    assert (
        self.args.update_weight_transport == "disk"
    ), "--update-weight-mode=delta requires --update-weight-transport=disk"
    from .update_weight.update_weight_from_disk_delta import UpdateWeightFromDiskDelta

    update_weight_cls = UpdateWeightFromDiskDelta
else:
    assert self.args.update_weight_mode == "full"
    ...
```

验证：如果希望 colocate，使用 full；如果希望 delta，使用 disk transport 并走非 colocate 路径。

## Q6：为什么 offload 失败可能在 actor 创建阶段就报错？

`torch_memory_saver` 依赖动态库预加载，Slime 在 Ray actor runtime env 中设置 `LD_PRELOAD`。这一步发生在远程 actor 类创建前。

```python
# 定位骨架（非逐行摘录）：slime/ray/actor_group.py L64-L84
if self.args.offload_train and self.args.train_backend == "megatron":
    import torch_memory_saver

    for path in [
        "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
        "torch_memory_saver_hook_mode_preload.abi3.so",
    ]:
        dynlib_path = os.path.join(...)
        if os.path.exists(dynlib_path):
            break
    else:
        raise FileNotFoundError(
            "Cannot find torch_memory_saver dynamic library. Please make sure torch_memory_saver is properly installed."
        )

    env_vars["LD_PRELOAD"] = dynlib_path
    env_vars["TMS_INIT_ENABLE"] = "1"
    env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
```

验证：如果错误出现在 actor 创建阶段，先查安装和 `.so` 路径；如果错误发生在 `sleep/wake_up`，再查 process group reload、未释放的 CUDA tensor 或 updater 连接。

## Q7：为什么 HF config/tokenizer 读取会卡在 barrier？

源码在每个节点内部按 local slot 串行读取 HF config/tokenizer，但同一 local slot 会在多节点同时读取。任何 rank 读 cache 卡住或失败，其他 rank 都会停在全局 gloo barrier。

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/actor.py L69-L76
for i in range(args.num_gpus_per_node):
    if i == dist.get_rank() % args.num_gpus_per_node:
        self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
    dist.barrier(group=get_gloo_group())

dist.barrier(group=get_gloo_group())
```

验证：看 `args.hf_checkpoint` 是否每个节点可访问；必要时先在每台机器本地预热 HF cache。

还要验证 `rank % num_gpus_per_node` 是否真对应节点内 local slot；若 placement/rank 排列不是每节点固定连续块，这个串行协议和 NUMA affinity 都会选错对象。

## Q8：init 成功为什么第一次推权仍然可能失败？

因为 init 只完成 updater 选型，`rollout_manager` 是 init 后才设置的，rollout engines 是 `update_weights()` 里动态获取的。

```python
# 来源：slime/ray/placement_group.py L207-L212
if args.start_rollout_id is None:
    args.start_rollout_id = start_rollout_ids[0]

actor_model.set_rollout_manager(rollout_manager)
if args.use_critic:
    critic_model.set_rollout_manager(rollout_manager)
```

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/actor.py L583-L606
def update_weights(self) -> None:
    if self.args.debug_train_only or self.args.debug_rollout_only:
        return
    ...
    (...) = ray.get(self.rollout_manager.get_updatable_engines_and_lock.remote())

    reconnect_rollout_engines = self.args.offload_train and self.args.use_critic and not self.args.colocate

    if not rollout_engines and not reconnect_rollout_engines:
        if dist.get_rank() == 0:
            logger.info("No updatable SGLang engines are running; skip weight update.")
        return
```

验证：init 日志正常但推权失败时，转查 rollout manager 是否提供 engines、engine lock 是否释放、updater 类型是否符合部署形态。

## Q9：`custom_megatron_init_path` 能做什么，不能做什么？

它在标准 Megatron init 之后执行，适合补注册 hook、metric buffer 或一次性 patch；不适合替代 `dist.init_process_group` 或 `mpu.initialize_model_parallel`。

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/initialize.py L88-L104
if args.deterministic_mode:
    ...
if args.tp_comm_overlap:
    from megatron.training.initialize import _initialize_tp_communicators

    _initialize_tp_communicators()

if getattr(args, "custom_megatron_init_path", None):
    from slime.utils.misc import load_function

    custom_init = load_function(args.custom_megatron_init_path)
    custom_init(args)
```

验证：自定义 hook 里可以读取 Megatron parallel state，但不应重新初始化 world 或改全局 rank。

## Q10：`vocab_size` 为什么优先 HF config？

一些模型的 tokenizer vocab 与模型原生 padded vocab 不一致。源码优先用 HF config 的 `vocab_size`，没有才回退 tokenizer。

```python
# 定位骨架（非逐行摘录）：slime/backends/megatron_utils/actor.py L133-L137
if self.args.vocab_size is None:
    hf_vocab = getattr(self.hf_config, "vocab_size", None)
    self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size
```

验证：embedding 或 logits shape 不对时，不要只看 tokenizer；同时看 HF config 和 Megatron checkpoint 的 vocab 语义。

## Q11：为什么 NumPy 2.x 修好后，原 actor 仍不能重试 init？

`initialize.init` 先调用 `mpu.initialize_model_parallel`，随后才断言 NumPy 主版本为 1。NumPy 2.x 抛错时，PyTorch world 和部分 Megatron groups 已存在；当前没有 except/finally 清理。修正环境后应重建 Ray actor/进程，而不是再次调用同一个 actor 的 init。

## Q12：辅助 ref/teacher load 失败后为什么 args 被污染？

`load_other_checkpoint` 先临时改写 `args.load`、`no_load_optim`、`no_load_rng`、`finetune` 与可选 `ckpt_step`，成功 load 后才手工恢复。中间没有 finally；异常会保留临时值，模型也可能部分覆盖。操作上应记录首个异常并销毁 actor，不能继续下一个 tag 或训练。
