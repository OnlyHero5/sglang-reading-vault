---
title: "阅读方法 · 核心概念"
type: concept
framework: slime
topic: "阅读方法"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/concept
  - source-reading
updated: 2026-07-12
---
# 阅读方法 · 核心概念

## 你为什么要读

Slime 的难点不在某个 loss 公式，而在多个系统共同维护一条 RL 闭环。最有效的读法不是背模块名，而是追对象、所有权、等待点和权重版本。本页建立后续所有专题共用的坐标系。

## 1. 先分清“系统承诺”和“实现契约”

README 把 Slime 概括为两项互相强化的能力：高性能训练与灵活数据生成。它们通过 Training / Rollout / Data Buffer 路径闭环，而不是两个互不相关的工具箱。

| 官方角色 | 当前实现中的主要承载者 | 不应误解为 |
|----------|------------------------|------------|
| Training | Megatron actor/critic、optimizer、RayTrainGroup | 单一 trainer class |
| Rollout | RolloutManager、rollout function、SGLang server/router | 只有一次 `/generate` 请求 |
| Data Buffer | DataSource、Sample/group、转换逻辑、Ray ObjectRef | 必须独立部署的 daemon |

官方角色帮助建立全局模型；实现契约则由当前源码中的对象、字段和等待关系决定。两者不能混写成一张“模块等于进程”的图。

## 2. 五本账

### 2.1 资源账：谁占 GPU

Ray PlacementGroup 预订 bundle，Slime 再为 training、rollout，必要时 critic 构造不同 view。colocate 表示资源视图可以重叠，不表示 Megatron 与 SGLang 变成同一 Python 进程，更不表示共享同一个参数对象。

### 2.2 样本账：谁生成和改写字段

DataSource 提供 prompt/sample group；rollout function 驱动 SGLang 或用户环境；reward、filter 与 converter 继续补齐或整理字段。要始终区分：

- 外层训练循环的 `rollout_id`；
- `Sample.rollout_id` 代表的样本组身份；
- 一次 rollout execution 可能产出的多条 training samples；
- 转换后的按 rank `rollout_data`。

名称相同或相近，不代表身份相同。

### 2.3 训练账：训练真正消费什么

RolloutManager 的 `generate()` 不只“生成文本”。它取得 rollout 结果、记录/保存调试数据、转换为训练字段、计算 DP/micro-batch schedule，并为各 DP rank 放入对象存储。训练侧消费的是这些 rank-local 数据引用，而不是直接消费 HTTP response。

### 2.4 版本账：哪一版 policy 可见

optimizer step 只改变训练侧参数；`update_weights()` 才把 actor 状态发布给 rollout。当前实现存在 tensor/CUDA IPC、distributed NCCL、full disk、delta disk 等路线。`weight_version` 是发布序号，不是参数正确性的校验和；版本递增也不保证参数数值必然改变，例如 critic-only 阶段仍可能发布未训练的 actor。

### 2.5 等待账：谁真的等谁

Ray `.remote()` 返回 future；真正的 happens-before 通常由 `ray.get` 或 actor 的串行执行语义建立。函数名带 `async_` 不等于整轮异步：同步 `train.py` 也调用 `async_train()`，随后立即 `ray.get`，因此训练完成后才继续发布权重。

## 3. Native 的准确含义

Slime 选择 Megatron + SGLang native，不是“完全没有适配层”，而是尽量保留上游控制面和能力：

- Megatron 参数直接进入训练参数体系；
- SGLang 的 `ServerArgs.add_cli_args` 被复用并加 `--sglang-` 前缀；
- Slime 仍跳过自己接管的 model path、拓扑、端口和分布式字段；
- Slime 仍负责资源编排、数据契约、权重 transport 与正确性边界。

因此，native 是“减少公共最小子集抽象并贴近上游”，不是“零转换、零约束、零框架逻辑”。

## 4. 同步、pipeline async 与 fully async

| 模式 | 关键等待关系 | 版本边界 |
|------|--------------|----------|
| 同步 `train.py` | generate 完成 → train 完成 → publish | 下一轮 generation 才看到新发布 |
| `train_async.py` | 下一批 generation 可与当前 train 重叠 | 更新前等待在途 generation，避免生成中途改权重 |
| fully async example | producer/consumer 与版本策略进一步解耦 | 必须显式分析 staleness，不可套用同步结论 |

异步不是“少一个 `ray.get`”这么简单。移动等待点会同时改变样本版本、显存峰值、故障传播和吞吐，需要连同五本账一起读。

## 5. 自定义接口是数据契约，不是魔法插件

`data_source_path`、`rollout_function_path`、`custom_generate_function_path`、`custom_rm_path` 等 hook 进入不同层级。它们不是固定的 `DataSource → custom_generate → custom_rm` 串行流水线：完整 rollout function 可以替换外层组织方式，custom generate/RM 则通常嵌入默认 rollout 实现。

判断一个自定义实现是否可用，要问：

1. 输入对象由谁拥有，是否允许回填 buffer；
2. 返回的是单条 Sample、Sample group，还是 rollout function 输出；
3. token、logprob、reward、loss mask 与 metadata 是否满足 converter/训练侧契约；
4. 异常、取消和 partial rollout 如何回收；
5. 生成样本对应哪一版权重。

## 6. 类比及其边界

可以把 Slime 想成一条“采样—加工—训练—发布”的工厂线：DataSource 提供工单，SGLang 生产轨迹，训练侧改进配方，weight updater 发布新配方。

这个类比只帮助记方向，不能推导实现：Data Buffer 不是实体仓库，weight update 也不是数据库事务；Ray object store、CUDA IPC、NCCL 和共享文件系统具有不同所有权与失败语义。进入排障时必须回到具体对象和 transport。

## 7. 阅读结论模板

每读完一条路径，至少写清：

> 在 baseline `22cdc6e1` 下，主体 A 持有对象 X，经边界 B 把它转换/发布为 Y；等待点 C 保证 D 发生在 E 之后。该结论由源码/测试证明；性能或跨故障恢复仍需在指定环境验证。

这个模板会强迫结论同时包含版本、主体、对象、边界、顺序和证据等级，避免“看起来会同步”“应该是异步”这类无法审计的表述。
