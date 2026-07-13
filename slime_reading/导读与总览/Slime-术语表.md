---
title: "Slime 术语表"
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

# Slime 术语表

## 你为什么要读

这不是缩写翻译表，而是语义消歧表。同一个词在 Ray、RL、Megatron 和 SGLang 中可能指不同对象；定义后都给出“不要混淆什么”和继续阅读入口。源码基线为 `22cdc6e1`。

## 最容易混淆的十组词

| A | B | 关键区别 |
|---|---|----------|
| Ray Actor | RL actor/policy | 前者是远程进程抽象，后者是被训练的策略模型 |
| outer rollout id | `Sample.rollout_id` | 前者是主循环时间轴，后者是一次 rollout 执行的 loss 分组键 |
| `group_index` | `Sample.rollout_id` | 前者常表示同 prompt 的采样组，后者允许 compact siblings 共享一次执行身份 |
| rollout batch | training step | 一次 outer rollout 可拆成多个 training steps |
| training sample | rollout execution | compact/subagent 下，一次执行可产生多条训练 sample |
| micro-batch | global batch size | 前者是单次执行切片，后者是当前 step 的全局 rollout 数 |
| rollout logprob | current/ref logprob | 分别由生成时 policy、训练模型、reference policy 产生 |
| optimizer step | `update_weights` | 前者改训练侧参数，后者把参数发布到 rollout engine |
| colocate | same process | colocate 是 GPU bundle 重叠，通常仍是不同进程 |
| weight version | checksum | 前者是发布序号，不能证明参数数值一致 |

## 身份与索引

### outer `rollout_id`

同步/异步主循环的外层迭代索引，用于 generate/train/save/eval 和日志时间线。它不要求等于每条 `Sample.rollout_id`。

→ [[Slime-训练主循环-核心概念]] · [[Slime-业务流程]]

### `Sample.rollout_id`

一次 rollout 执行的训练聚合身份。compact/subagent 将一次执行拆成多个 sibling samples 时，siblings 必须共享该值，使 loss reducer 只计一次 rollout。

→ [[Slime-Sample数据契约-核心概念]] · [[Slime-RL训练全链路]]

### `group_index`

DataSource 为同一个 prompt 复制出的 sample group 分配的编号，常用于 reward group、zero-std/pass-rate 等语义。它不自动等同于 `rollout_id`。

### `index`

DataSource 分配的单条 sample 序号。默认一执行一 sample 的兼容路径可用它形成 rollout identity，但 compact 路径应显式填写 `rollout_id`。

### DP rank

Megatron 数据并行消费者身份。RolloutManager 返回长度等于 DP size 的 refs，每个 worker 根据自己的 DP rank 取一份 partition。

## 运行主体

### Ray Actor

Ray 管理的远程 Python 对象/进程。Slime 的 training worker、RolloutManager、SGLangEngine wrapper、lock 等都可以是 Ray actors。

### actor / policy

被优化并用于 rollout 的策略模型。训练侧可同时维护 actor、old_actor、rollout_actor、ref、teacher 等参数快照。

### critic

可选 value model。当前 args 中 `advantage_estimator=ppo` 会派生 `use_critic=True`；critic 先计算/训练 values，再把 pipeline-last-stage 的 values 交给 actor。

### RolloutManager

Ray actor 边界，持有 RolloutServer(s)、DataSource、rollout/eval functions、converter、DP schedule、debug/metrics/health/recovery。它不是实际 token generation 算法本身。

→ [[Slime-RolloutManager]]

### RayTrainGroup

按 world rank 管理一组 training Ray actors，提供 async init/train、save、update/offload 等扇出接口。它把同一 rollout refs 发给所有 worker，不在 group 层按 DP 重切数据。

→ [[Slime-RayTrainGroup]]

### SGLangEngine

Slime 对 SGLang server 的 Ray actor wrapper，负责启动/连接 server、HTTP 控制、memory occupation、weight update/check 等。SGLang Scheduler/ModelRunner 仍属于 SGLang upstream。

→ [[Slime-SGLang-Engine]] · [[Slime与SGLang-阅读对照]]

### RolloutServer / ServerGroup

`RolloutServer` 表示一个 model 及其 router；可包含多个同配置 `ServerGroup`。group 可为 regular、prefill、decode、encoder、placeholder，并有独立 GPU 数、offset、offload 与 model path。

## 数据对象

### DataSource

抽象契约：`get_samples`、`add_samples`、`save`、`load`、`len`。默认参数使用 `RolloutDataSourceWithBuffer`：优先取 buffer，再从全局 dataset 取 prompt groups。

→ [[Slime-数据源]]

### prompt group

一个 prompt 复制出 `n_samples_per_prompt` 条 sample seeds 后形成的组。group 用于相对 reward/统计，但一次 seed 后续可能再拆成 compact training samples。

### `Sample`

生成与训练之间的语义护照。当前核心字段为 `tokens`、`response`、`response_length`、`reward`、`loss_mask`、`rollout_log_probs`、`weight_versions`、status 与 metadata；注意 `reward/loss_mask` 是单数。

→ [[Slime-Sample数据契约]] · [[Slime-关键概念]]

### `RolloutFnTrainOutput`

rollout function 的训练返回契约，包含 nested `samples` 与 `metrics`。它不是训练侧 `RolloutBatch`。

### train data / `RolloutBatch`

Sample 转换后的字段 dict。RolloutManager 补 `rollout_ids`、`loss_masks`、`rollout_mask_sums` 等，DP split 后又加入 partition、micro-batch schedule 与全局信息。

### `Box(ObjectRef)`

对 Ray object ref 的轻量包装。RolloutManager 返回每个 DP rank 一份；object-store 与 NIXL 都保持这一上层语义。

### `loss_mask`

长度等于 `response_length`，标记哪些 response token 参与 loss。工具/环境 token、remove_sample 等路径可以产生 0 mask。

### `rollout_mask_sums`

同一 `Sample.rollout_id` 下所有 sibling samples 的 loss-mask 总和，并按 sample 广播保存。micro-batch 拆分后仍用于恢复完整 rollout 的归一化分母。

### `micro_batch_indices`

DP schedule 预计算的 rank-local sample index 列表。DataIterator 按它取 batch，不重新决定如何 pack。

## 并行与资源

### PlacementGroup（PG）

Ray 资源 bundle 集合。Slime 通常创建一个 PG，再给 actor/rollout 不同或重叠的 bundle views；critic 复用 actor PG 描述，不是固定创建三个 PG。

→ [[Slime-PlacementGroup]]

### DP / TP / PP / CP / EP / VPP

| 缩写 | 含义 | Slime 中的关键影响 |
|------|------|--------------------|
| DP | Data Parallel | rollout partition、梯度同步、跨 DP normalization |
| TP | Tensor Parallel | 层内张量/参数切分、logits collective |
| PP | Pipeline Parallel | stage 输入输出、只有 last stage 产完整 logprob/value/metrics |
| CP | Context Parallel | response/logprob/mask 的序列切片与 gather |
| EP | Expert Parallel | MoE expert/routing 分布 |
| VPP | Virtual Pipeline Parallel | model chunks 与 micro-batch group 对齐 |

这些 group 由 Megatron 建立；PG 只决定资源落点。

### colocate

actor 与 rollout 使用重叠 GPU bundle，通过 offload/onload 时分复用。当前选择 tensor/CUDA IPC updater；流水异步入口不支持 colocate。

### offload_train / offload_rollout

分别管理训练与 rollout 显存生命周期。training sleep/wake 涉及 process groups 与 memory saver；rollout 只对 `needs_offload` 的 server groups release/resume，并区分 weights 与 KV/CUDA graph。

### external rollout

SGLang engines 由外部系统提供；Slime 应用外部地址/拓扑到 args，不在本地 PG 创建 engine actors。训练与 rollout 仍通过请求、样本和权重协议交接。

### NIXL

可选 rollout-data tensor transport。它改变 Ray refs 中大 tensor 的传输方式，不改变 Sample、DP partition 或 `RolloutBatch` 的语义。

## 训练批次

### outer rollout

一次主循环 generation + training + publication 节拍。它可以包含多个 training steps。

### global batch size

当前 DP schedule/loss normalization 中，一个 training step 包含的 rollout execution 全局数量。默认一 execution 一 sample 时等于样本数；compact rollout 下不能解释成裸 sample 行数。

### micro-batch

一次 pipeline forward/backward 消费的执行切片。多个 micro-batches 累积后完成一个 optimizer step。

### dynamic batch size

按 token/workload 把样本 pack 成可执行 micro-batches。`balance_by_flops` 可改善计算均衡，但不保证 token cap，配置不当仍可 OOM。

### DP schedule

先按 `Sample.rollout_id` 保组到 training step，再 pack micro-batches，再把 micro-batches 分给 DP ranks，并保证 DP/VPP 所需对齐。

→ [[Slime-RL训练全链路]] 的 DP schedule 小节

## 概率、价值与优化

### reward / raw reward

rollout/RM/verifier 给样本的结果信号。`reward` 可为 scalar 或 dict；`reward_key` 选择训练值。raw reward 与归一化 reward 不应混用。

### rollout logprob

SGLang 生成时的 token log probability，记录采样 policy。当使用 partial/async/不同推理训练实现时，它与训练侧重算值可能有差异。

### current/train logprob

Megatron actor 对同一 token 重新计算的概率。用于 policy loss、KL、mismatch 等；根据配置也可能作为 old policy 基线。

### ref logprob

reference 参数快照产生的概率，用于 reward shaping KL 或 KL loss。ref 可以按 `ref_update_interval` 从 actor 更新，不一定永久冻结。

### value

critic 对 response positions 的价值预测。只在 pipeline last stage 形成完整返回；其他 stage 返回空 dict 是正常行为。

### advantage / return

由 reward、KL、value、mask 与 estimator 计算。当前 estimator 包括 GRPO、GSPO、CISPO、PPO、REINFORCE++ 变体或 custom function。

→ [[Slime-Advantage计算]]

### policy/value/SFT/custom loss

训练目标的分派层。actor CLI 可选 policy、SFT、custom；critic 内部设置 value loss。不要把整个 Slime loss 概括成固定 PPO/GRPO。

### TIS

Truncated Importance Sampling。根据训练/rollout probability ratio 修正或裁剪 off-policy mismatch；与 `use_rollout_logprobs` 等配置存在互斥/分支关系。

### OPSM

Off-Policy Sequence Masking。按序列级 mismatch 条件生成 mask，参与 policy loss；不是 reward filter。

### routing replay / rollout routing replay

训练侧记录/重放 MoE routing，或使用 rollout engine 返回的 routed experts。会改变 actor/ref 路由可比性与 KL 预期。

### OPD

On-Policy Distillation。teacher logprob 可来自 SGLang 或 Megatron teacher；KL penalty 叠加到 advantage，和 advantage estimator 正交。

## 权重与版本

### optimizer step

更新 Megatron 训练侧参数并推进训练 scheduler。它不自动修改 SGLang engine。

### `update_weights`

训练侧发布入口：取 updatable engines/lock，连接或恢复通信，调用具体 updater。debug-only 或没有 updatable engine 时可直接跳过。

### update weight mode

`full` 或 `delta`。delta 只支持 disk transport，需要共享发布目录和 rollout-host-local checkpoint，不支持 colocate。

### update weight transport

full 模式可用 NCCL 或 disk；colocate 由 tensor updater 处理。mode、transport 与资源关系共同决定实际 updater。

### weight version

updater/engine 的发布序号，并可记录进 `Sample.weight_versions`。版本一致用于判断发布顺序，不证明参数 checksum 或 optimizer step 一定改变了数值。

### actor / old_actor / rollout_actor / ref / teacher

训练进程内可能存在的参数备份 tags：

| tag | 作用 |
|-----|------|
| actor | 当前待训练/已训练 policy |
| old_actor | PPO/更新间隔等路径的旧策略快照 |
| rollout_actor | update interval=1 时维护的 rollout 策略队列中间态 |
| ref | KL reference，可周期更新 |
| teacher | Megatron OPD teacher |

它们是训练侧参数快照，不等同于 SGLang engine 当前装载版本。

## 执行模式

### synchronous baseline

`generate(N) → train(N) → publish → generate(N+1)`。版本关系最清晰，适合首次学习。

### pipeline async

`train_async.py` 提前启动下一轮 generation，与当前训练重叠；更新前等待在途 generation。存在受控策略陈旧度，不支持 colocate。

### fully async

生成、buffer、训练、发布可独立推进，需要显式设计版本窗口、旧样本策略与生产消费速率。不是 pipeline async 的同义词。

### debug rollout-only / train-only

rollout-only 只验证生成，不转换/训练；train-only 不启动本地 SGLang，从保存 samples replay converter、DP schedule 与训练。两者不能同时开启。

## 可观测与恢复

### trace carrier / span / event

动态绑定到 Sample 的诊断载体，包含 trace/sample/group id、attempt 与 events；可展开 SGLang PD/perf metadata。trace 操作多为尽力而为，缺记录不等于业务未执行。

### health monitor / recover

按 ServerGroup 监控 rollout engines；恢复死 engine 后，新 engine 还需重新加入权重连接/发布。健康恢复与权重一致性是两个连续步骤。

### `check_weight_update_equal`

bootstrap 或诊断时比较训练/rollout 权重的可选验证，比只看 weight version 更强，但仍要结合配置、量化与目标模型路径解释。

## 字母索引

| 字母 | 术语 |
|------|------|
| A | actor、advantage、async、attention（SGLang 所有） |
| B | Box、batch、buffer、bootstrap update |
| C | colocate、CP、critic、custom hook |
| D | DataSource、DP、dynamic batch、debug mode |
| E | EP、external rollout、engine |
| G | global batch、GRPO/GSPO、group index |
| L | logprob、loss mask、loss type |
| M | Megatron、micro-batch、model tag |
| N | NIXL |
| O | ObjectRef、offload、OPD、OPSM、optimizer |
| P | PG、policy、PP、prompt group、PPO |
| R | Ray Actor、RayTrainGroup、reward、rollout、routing replay |
| S | Sample、ServerGroup、SGLangEngine、synchronous |
| T | TIS、TP、trace、transport |
| U | update weight mode/transport/updater |
| V | value、version、VPP |
| W | weight version、weight publication |

继续阅读：[[Slime-关键概念]] 负责建立关系，[[Slime-源码地图]] 负责定位文件，[[Slime-RL训练全链路]] 负责追踪对象生命周期。
