---
title: "外部推理引擎 · 学习检查"
type: exercise
framework: slime
topic: "外部推理引擎"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# 外部推理引擎 · 学习检查

这份清单用于确认你能部署和排障 external engine，而不是只知道 `--rollout-external-engine-addrs` 这个参数。

---

## 读者能做什么

- [ ] 能画出外部 SGLang server、Slime `SGLangEngine` zero GPU adapter、router、HTTP client、Megatron updater 的关系图。
- [ ] 能沿启动链复述：传 external 地址、请求 `/server_info`、写回 `args`、PG 不预留 rollout GPU、创建 zero GPU actor、sanity check、注册 router。
- [ ] 能解释 `rollout_num_gpus` 在 external 模式下是逻辑容量，不是 Ray PG 资源申请。
- [ ] 能区分 external 与 `--sglang-config`：谁拥有 server 生命周期、谁负责 recover、谁支持多模型和 frozen model。
- [ ] 能说明 generate 请求为什么走 router，而不是通过 Ray actor 做 forward。
- [ ] 能按部署现实选择 full+NCCL、full+disk 或 delta+disk 权重同步。
- [ ] 能指出 external 模式下 Slime 不负责 server kill、offload、onload 和 recover。

---

## 排障验收

| 给定症状 | 你应该能定位到 |
|----------|----------------|
| Slime 没进入 external | `args.rollout_external` 是否设置，`apply_external_engine_info_to_args` 是否运行 |
| `/server_info` 失败 | 地址格式、server 绑定地址、proxy/no_proxy、fallback endpoint |
| router 没 worker | `_init_external` sanity check、`_register_to_router`、PD bootstrap port |
| PG 申请了 rollout GPU | `_get_placement_group_layout` external 分支是否命中 |
| generate 卡在 retry | `http_utils._post`、router URL、external server 健康 |
| disk update 找不到文件 | trainer 与 external engine 是否共享同一路径 |
| engine 挂掉后没有恢复 | 外部编排系统，而不是 Slime fault tolerance |

---

## 可执行验证

| 验证 | 命令或入口 | 预期 |
|------|------------|------|
| discovery mock | `pytest slime/tests/test_external_sglang_engines.py -q` | worker type、GPU 数、PD bootstrap 被正确推导 |
| PG 布局 | `pytest slime/tests/test_placement_group.py -k external -q` | external 为 `(actor_gpus, actor_gpus)`，external debug rollout 为 `(0, 0)` |
| 手动 server info | `curl http://host:port/server_info` | 返回 TP/PP、worker type、可选 bootstrap port |
| router worker | router `/workers` | external URL 被注册，prefill/decode 类型正确 |
| HTTP proxy | 训练 job 环境变量 | `no_proxy/NO_PROXY` 包含 external host |
| disk transport | 在 trainer 写、engine 读同一路径 | external server 能读到 `--update-weight-disk-dir` |

---

## 最小口试

1. 为什么 external 模式仍然创建 `SGLangEngine` actor？
2. 为什么 external actor 的 `num_gpus=0` 不等于 external server 没有 GPU？
3. `/server_info` 的哪些字段会影响 Slime 后续行为？
4. 为什么 `--rollout-external-engine-addrs` 和 `--sglang-config` 互斥？
5. external PD prefill worker 为什么需要 `disaggregation_bootstrap_port`？
6. 为什么跨集群 external 常选 delta+disk，而不是 NCCL？

---

## 改代码前的不变量

- [ ] 改 discovery 时，必须同步更新 mock test 对 worker type、GPU 数和 bootstrap port 的断言。
- [ ] 改 PG 布局时，必须保留 external 不占 rollout GPU 的语义。
- [ ] 改 router 注册时，必须同时验证 regular、prefill、decode、encoder 分支。
- [ ] 改 HTTP client sizing 时，必须优先使用 `rollout_num_engines`。
- [ ] 改权重同步时，必须写清 metadata 控制面和 tensor/checkpoint 数据通道分别走哪里。
- [ ] 改 fault tolerance 时，必须先明确外部 server 生命周期是否仍归外部系统。

---

## 下一篇

如果你已经掌握 external 接入，继续读 [[Slime-磁盘权重同步]]，重点看 full checkpoint、disk delta、local checkpoint 和 shared filesystem visibility 如何配合 external serving。
