---
title: "Slime 复杂度热点"
type: reference
framework: slime
topic: "总结复盘"
learning_role: reference
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/reference
  - source-reading
updated: 2026-07-13
---
# Slime 复杂度热点

> **读者任务：** 根据症状定位“协议汇点”，而不是看到长函数就从第一行顺读。

## 1. 真正复杂的是状态交界，不是代码长度

Slime 的热点函数通常同时跨越两种以上状态：CLI 与运行时对象、prompt 与 Sample、Sample 与 tensor、训练权重与 engine 版本。复杂度来自所有权和时序叠加，而不是单纯分支多。

| 汇点 | 汇合的状态 | 一旦出错最像什么 |
|------|------------|------------------|
| `parse_args` | SGLang、Megatron、Slime、debug 预解析 | 参数被吃掉、验证走错、组件不该启动却启动 |
| `RolloutManager.generate` | engine 健康、rollout 输出、debug dump、train-data 转换、DP split | 有 dump 无训练、shape 错、故障注入后卡住 |
| `generate_rollout` | DataSource、async generate、RM/filter、partial abort、train/eval | 样本数不齐、reward 错位、eval 行为漂移 |
| `MegatronTrainRayActor.train` | ObjectRef、offload、actor/critic、数据预处理 | 等待慢、OOM、critic value 不齐 |
| advantage / policy loss | reward、KL、value、mask、old/current/rollout logprob | advantage 异常、ratio 爆炸、指标与梯度口径分叉 |
| `update_weights` | engine 恢复、连接、transport、版本、colocate | 旧权重、半更新、collective 卡死 |

## 2. 参数热点：先认清“两阶段解析”

`parse_args` 先用 `_pre_parse_mode` 取出 `debug_rollout_only`、`debug_train_only` 和 debug dump path，再决定是否解析 SGLang 参数；随后才解析 Megatron + Slime，并把两个 namespace 合并。`debug_train_only` 或加载 debug rollout data 会跳过 SGLang parser；`debug_rollout_only` 又会跳过部分 Megatron/HF 校验。来源：`slime/utils/arguments.py` L1531-L1590。

因此“参数没生效”至少有四种不同原因：

1. 参数属于另一个 parser，当前 mode 根本没有创建它。
2. Megatron parser 以 `ignore_unknown_args=True` 放过了 SGLang 参数，但后续没有 SGLang namespace 可合并。
3. pre-parse flag 改变了验证路径，而不是只设置一个布尔值。
4. YAML role override 或后续 validation 重写了最终值。

排查时先打印最终 `args` 和当前 mode，再去看 argparse 定义；不要只在启动 shell 中确认字符串出现过。

## 3. RolloutManager：对象形状的最后一道闸

`RolloutManager.generate` 依次恢复 health monitoring、按 CI 条件注入 engine 故障、调用外层 rollout、保存原始 dump、记录指标；正常模式再把 samples 转成 train data 并按 DP 切分。`debug_rollout_only` 会在转换前直接返回。来源：`slime/ray/rollout.py` L543-L565。

```text
rollout function output
        │
        ├─ save/debug/log：仍是 Sample 语义
        │
        └─ convert → DP split：进入训练 batch 语义
```

症状路由：

| 症状 | 先检查 | 预期证据 |
|------|----------|----------|
| dump 正常、actor 没数据 | `debug_rollout_only` 与 convert 返回 | 正常训练应返回每个 DP rank 的数据引用 |
| fan-out 后 reward/group 异常 | convert 前的嵌套形状与拍平时点 | 每个下游 hook 看到的对象类型明确 |
| fault-tolerance CI 卡住 | `_try_ci_fault_injection` 的 sleep 与 health monitor | engine 被标失效，后续恢复边界可见 |
| DP rank batch 不等 | `_split_train_data_by_dp` 前的全局 batch | split 前数量和动态 batch 元数据自洽 |

## 4. 默认 SGLang rollout：并发生成不等于无状态

外层 `generate_rollout(args, rollout_id, data_source, evaluation)` 只负责选择 train/eval coroutine、运行事件循环并把 aborted samples 回填 DataSource；真正的复杂度在 `generate_rollout_async`、`generate_and_rm_group` 与 `generate_and_rm`。来源：`slime/rollout/sglang_rollout.py` L618-L641。

沿一组 prompt 阅读时，逐项跟踪：

- 输入 group 是 `list[Sample]`，还是 fan-out 后的嵌套 list；
- `custom_generate` 返回单样本还是 sibling list；
- RM 是 per-sample 还是 group RM，长度是否被严格校验；
- partial abort、filter 和 DataSource 回填保留了哪些 group 身份；
- train 返回 `RolloutFnTrainOutput`，eval 返回按 dataset 聚合的结果。

只替换单个 prompt 的生成时优先用 `custom_generate`；只有要接管队列、批次、返回结构和失败恢复时，才替换整个 `rollout_function`。

## 5. Train actor：先区分“等数据”与“算训练”

`MegatronTrainRayActor.train` 在 `debug_rollout_only` 下直接返回；需要时先 wake up，再把 ObjectRef 中的数据取回和预处理。其后由 actor/critic role 分流，最后在 `offload_train` 下删除数据并 sleep。来源：`slime/backends/megatron_utils/actor.py` L377-L400。

训练慢或 OOM 时，把总时间拆成：

| 阶段 | 典型问题 | 工具 |
|------|----------|------|
| `_get_rollout_data` | Ray 等待、CPU→GPU、列表转 tensor、动态 batch 元数据 | timer + ObjectRef 就绪时间 |
| actor/critic forward | PP/CP、routing replay、logprob/value 计算 | train profiler |
| backward/optimizer | micro-batch、recompute、loss reducer、collective | PyTorch profiler / NCCL 日志 |
| wake/sleep | colocate 或 host offload 抖动 | 显存快照 + 阶段日志 |

“actor 慢”不能只看 `train_actor`；它可能根本还没有进入模型计算。

## 6. Advantage 与 policy loss：同名算法跨两层

`compute_advantages_and_returns` 在 pipeline last stage 上构造 KL，然后选择 custom、GRPO/GSPO/CISPO 共享 returns、PPO GAE 或 REINFORCE++ 系列分支；可再叠加 OPD reverse-KL 修正和 masked DP whitening。CISPO 在这里与 GRPO/GSPO 共享 advantage 生成，不代表它的 policy loss 与 GRPO 相同。来源：`slime/backends/megatron_utils/loss.py` L661-L790。

`policy_loss_function` 才把 current logits 变成 current logprob/entropy，与 old policy baseline 形成 clipped surrogate；GSPO 需要 full-sequence gather，还可叠加 TIS/ICE-POP 类修正、KL loss 和自定义 reducer。来源：`slime/backends/megatron_utils/loss.py` L881-L1100。

排障时保留三本账：

- 信用账：reward、KL、value 如何变成 advantage/return；
- 策略账：current、old、rollout、reference logprob 各是谁；
- 归约账：token mask、sample mean、micro-batch 与 DP/CP reducer 如何对齐。

## 7. 权重更新：恢复、连接、传输、发布是四步

`update_weights` 在 debug mode 下跳过。启用 fault tolerance 时，rank 0 先让 RolloutManager 恢复可更新 engine，再用 gloo barrier 对齐；随后获取 engine、lock、GPU 布局和 actor 列表。新 engine 或远端 critic/offload 重连会触发连接重建，最后才由具体 updater 传输权重。来源：`slime/backends/megatron_utils/actor.py` L580-L650。

如果 rollout 仍使用旧策略，按因果顺序检查：

1. 本轮是否按 `update_weights_interval` 计划更新。
2. debug/无 updatable engine 分支是否提前返回。
3. fault recovery 是否找回正确的 updatable server，而不是 frozen model。
4. process group 或 engine lock 是否已重建。
5. transport 是否完整结束，engine version 是否推进。
6. 下一批样本何时开始生成；旧样本不会因后来更新而变新。

## 8. 一页排障路由

| 现象 | 第一源码入口 | 对应专题 |
|------|--------------|----------|
| 参数/组件实例化反常 | `slime/utils/arguments.py::parse_args` | [[Slime-训练与Rollout参数]] |
| rollout 空、少、重复 | `slime/ray/rollout.py::generate` | [[Slime-RolloutManager]] |
| token/logprob/reward 错位 | `slime/rollout/sglang_rollout.py` | [[Slime-SGLang-Rollout]]、[[Slime-Reward与过滤]] |
| critic value 或 actor batch 异常 | `actor.py::train` | [[Slime-训练步骤]] |
| advantage/loss NaN | `loss.py` 两层入口 | [[Slime-Advantage计算]]、[[Slime-Policy-Loss]] |
| engine 权重陈旧或更新卡住 | `actor.py::update_weights` | [[Slime-权重同步]] |

## 导航

- [[Slime-可观测性与CI]]
- [[Slime-综合学习检查]]
