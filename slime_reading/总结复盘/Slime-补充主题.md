---
title: "Slime 补充主题"
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
# Slime 补充主题

> **用途：** 给没有独立六件套专题、但会改变系统边界的实现提供补读路线。本页不是“次要文件清单”，而是防止把旁路误当默认主线。

## 1. 先按问题选入口

| 你遇到的问题 | 应补读的实现 | 为什么不能只套默认主线 |
|--------------|--------------|------------------------|
| 需要独立 teacher 计算 logprob、采样 token 或 label-token logprob | `megatron_server.py` | 它是 HTTP teacher 服务，不是 SGLang rollout engine |
| 想让第 `n+1` 轮生成与第 `n` 轮训练重叠 | `train_async.py` | 它引入权重陈旧和更新前排空，不支持 colocate |
| 想把生成彻底做成持续流 | `examples/fully_async/` | 这是比 `train_async.py` 更激进的异步方案，不能混为一谈 |
| 只训练 critic 若干轮 | actor/critic 分支与 `num_critic_only_steps` | rollout 仍发生，但 actor 是否训练、保存和清理资源会变化 |
| 使用 Megatron FSDP 相关能力 | Megatron 配置与兼容 patch | 当前没有独立的 Slime FSDP backend；它仍在 Megatron 后端内部 |
| 轨迹由外部机器生产 | `slime_plugins/rollout_buffer/` | 它替换整批 rollout，并引入 HTTP、队列和恢复协议 |
| 修改测试矩阵或发布镜像 | CI workflow template | 生成文件不是事实源，CPU/GPU job 的证明范围不同 |

## 2. `megatron_server.py`：专用 teacher logprob 服务

当前实现不是泛化的“Megatron forward-only 示例”。它把训练模型配置成不训练参数的 teacher 模式，建立 `SampleManager` 队列，让各 DP rank 的 `TeacherLogpRayActor` 计算 logprob，并暴露 HTTP：

| endpoint | 作用 | 关键边界 |
|----------|------|----------|
| `POST /generate` | 接收 `input_ids`，返回 logprob，并可返回 sampled/label-token 结果 | 输入长度、`sample_n`、label 矩阵会校验；结果轮询没有请求级总超时 |
| `POST /update_weights_from_disk` | 等队列与 inflight 清空后从磁盘换权重 | 同路径请求可合并，不同路径并发更新返回 409；超时只约束等待 idle |
| `GET /get_loads` | 查看 pending、inflight、累计 request/token | 是服务负载，不是训练 loss 指标 |
| `GET /info`、`/healthz`、`/detect` | 配置、存活和服务类型 | healthz 不证明模型数值正确 |

`configure_megatron_server_args` 会强制 `debug_train_only=True`、关闭 KL/OPD/critic 和 optimizer/RNG load，并把可训练参数列表设为 `nothing_to_train`。它适合 on-policy distillation teacher、独立 logprob 服务和热换 checkpoint；不参与默认 `generate → train → update_weights` 闭环。来源：`slime/backends/megatron_utils/server/arguments.py` L75-L131、`slime/backends/megatron_utils/server/megatron_server.py` L399-L615、L704-L770。

推荐顺序：

1. [[Slime-Policy-Loss]]：先明确 student 侧何时需要 teacher logprob。
2. `slime/backends/megatron_utils/server/arguments.py`：确认 teacher-only 强制配置。
3. `slime/backends/megatron_utils/server/megatron_server.py`：沿 HTTP request → queue → DP worker → result 回读。

## 3. `train_async.py`：一轮预取，不是 fully async

`train_async.py` 的核心是先提交下一轮 `generate.remote`，再训练当前轮；到 `update_weights_interval` 边界时，它先等待尚未完成的 generation，再更新权重，避免生成中途切版本。它显式拒绝 colocate。来源：`train_async.py` L10-L50、L77-L83。

由此得到三条必须写进实验记录的约束：

- 第 `n+1` 轮样本可能在第 `n` 轮训练结束前已开始生成，因此天然存在可控陈旧。
- `update_weights_interval > 1` 时，多个 rollout 会共享旧 engine 版本；这是策略选择，不是自动错误。
- 更新前排空 future 只能避免“半次生成换权重”，不能让已经生成的样本追溯性地变新。

`examples/fully_async/` 是另一套持续异步路径。不要因为文件名都含 async，就假设它们有相同队列、版本和反压语义。

## 4. critic-only：不是“没有 actor”的训练

同步主循环仍先生成 rollout。处于 `num_critic_only_steps` 时，critic 训练并返回 value，actor 不执行训练；保存和内存清理也根据本轮 actor 是否训练分支。过了阈值后，critic 的 value 作为 external data 交给 actor。来源：`train.py` L55-L89。

排障时应分别记录：

- rollout 是否正常产生 reward 与 logprob；
- critic 是否产出与样本对齐的 value；
- actor 本轮是否按阈值被故意跳过；
- checkpoint 中 actor/critic 哪一方应出现新版本。

## 5. Megatron FSDP：模式，不是第二套 Slime backend

当前 `--train-backend` 只接受 `megatron`。仓库中的 FSDP 相关逻辑主要是 Megatron 自身 FSDP 配置与兼容 patch，例如梯度属性和 local tensor 的版本适配；不能把它描述成与 Megatron actor 平行的独立 Slime FSDP 后端。来源：`slime/utils/arguments.py` L1531-L1544、`slime/backends/megatron_utils/megatron_patch/megatron_chunked_grad_coalesce_patch.py` L1-L119。

遇到 FSDP 问题时，先判断它属于 Megatron 版本兼容、分布式梯度表示，还是 checkpoint/optimizer 状态；不要寻找不存在的 `slime/backends/fsdp_utils/` 主线。

## 6. rollout_buffer 与 CI

外部轨迹服务的协议、生产缺口和验证入口已经集中在 [[Slime-插件与示例]]。最重要的结论是：当前 rollout_buffer 是内存实验原型，不提供持久化、ack/lease、去重和完整截止时间。

CI 则应从 `docs/en/developer_guide/ci.md` 与 `.github/workflows/pr-test.yml.j2` 阅读。CPU tests 证明纯 Python/contract 不变量；label-gated GPU e2e 才覆盖真实 Megatron + SGLang 联动。生成的 `pr-test.yml` 不是手工修改入口。

## 7. 与 SGLang 的交叉补课

| 主题 | Slime 侧 | SGLang 侧 |
|------|----------|-----------|
| rollout 请求与 engine 生命周期 | [[Slime-SGLang-Engine]] | [[SGLang-Scheduler]] |
| 权重热更新 | [[Slime-分布式权重同步]] | [[SGLang-CheckpointEngine]] |
| Agent wire message 与 tool parse | [[Slime-Agent轨迹]] | [[SGLang-OpenAI-API-源码走读]] |
| 外部服务与网关 | [[Slime-插件与示例]] | [[SGLang-model-gateway]] |

跨库职责边界见 [[三框架知识地图]]。

## 导航

- [[Slime-总结复盘]]
- [[Slime-可观测性与CI]]
- [[Slime-综合学习检查]]
