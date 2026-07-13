---
title: "数据源 · 学习检查"
type: exercise
framework: slime
topic: "数据源"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# 数据源 · 学习检查

这份清单用来判断你是否真的掌握了 DataSource，而不是只记住“它负责读 prompt”。

## 读者能画出来

- [ ] 能画出 `prompt-data -> Dataset -> RolloutDataSource -> get_samples -> generate_rollout_async -> RolloutFnTrainOutput` 主线。
- [ ] 能在图上标出 buffer 回灌口：`abort(partial) -> add_samples -> 下一轮 get_samples`。
- [ ] 能说明 `Dataset` 与 `process_rollout_data` 虽在同一工具文件里，但分别服务 prompt 加载和训练数据解包。

## 读者能解释清楚

- [ ] `get_samples(N)` 中的 N 是 prompt group 数，不是单条 Sample 数。
- [ ] 每个 group 的长度为什么必须等于 `n_samples_per_prompt`。
- [ ] `sample_offset`、`epoch_id`、`sample_group_index`、`sample_index` 四个字段分别管哪本账。
- [ ] 为什么同一 prompt 的多条 Sample 要 `deepcopy`。
- [ ] 为什么 buffer group 和 fresh dataset group 形状相同，但字段成熟度不同。
- [ ] 能解释默认 `get_samples(N)` 只跨一个 epoch，为什么超大 N 会少返回并留下越界 offset。
- [ ] 能说明 dataset shuffle 的排列可复现与“污染进程全局 RNG”可以同时成立。

## 读者能排障

- [ ] 默认 rollout 断在 `assert args.rollout_global_dataset` 时，能判断是模式不匹配，而不是数据文件坏。
- [ ] dynamic filter drop 后 dataset 消费快于有效训练样本时，能指出默认没有回写 buffer。
- [ ] partial rollout 没有复用半成品时，能检查是否使用 `RolloutDataSourceWithBuffer`、是否开启 `partial_rollout`、是否有 pending task。
- [ ] 续训后 prompt 顺序错位时，能检查 checkpoint 文件、`epoch_id`、`sample_offset`、`rollout_seed` 和 shuffle。
- [ ] 多模态长度过滤没有生效时，能检查 `apply_chat_template` 和 processor 路径。
- [ ] 空 dataset 时能预判默认 rollout/fully-async 都没有 EOF 终态，而不是等待它们自然结束。
- [ ] 负数广义路径切片报错时，能区分正则接受语法与 `islice` 不支持语义。
- [ ] 多模态配置存在但行内媒体为空时，能定位空正则导致逐字符 content 的问题。
- [ ] fully-async 没有 dynamic filter metrics 时，能说明它绕过默认生成主循环而非指标系统故障。

## 读者能做最小验证

```powershell
Push-Location slime
python -m pytest tests/plugin_contracts/test_plugin_path_loading_contracts.py -q
Pop-Location
```

预期现象：

- 默认 `RolloutDataSourceWithBuffer` 可动态加载。
- 默认 `pop_first` 签名稳定。
- 自定义 data source 的构造函数和 `get_samples` 最小返回形状被验证。

契约测试不覆盖超大 N 跨多 epoch、空 dataset 进展性、buffer filter 超额返回、全局 RNG 副作用或 fully-async 控制面缺失；这些必须用下面的边界场景单独验证。

静态替代：

```powershell
rg -n 'class RolloutDataSource|class RolloutDataSourceWithBuffer|def get_samples|def add_samples|def save|def load' slime/slime
rg -n 'rollout_global_dataset|partial_rollout|sample_offset|epoch_id' slime/slime
```

预期现象：能定位 fresh dataset、buffer 回灌、checkpoint 状态与 partial rollout 的分叉；只证明类可 import，不能证明 group shape 正确。

## 复盘迁移

- [ ] 读 [[Slime-Sample数据契约]] 时，能把 DataSource 产出的 prompt 阶段 Sample 接到 response 时间轴账本。
- [ ] 读 [[Slime-SGLang-Rollout]] 时，能看懂为什么默认主循环只接收 `get_samples` bound method。
- [ ] 读 [[Slime-其他Rollout路径]] 时，能解释 fully-async 为什么把 buffer 当跨 step 队列。
- [ ] 同时能解释 fully-async 为什么不自动继承默认 dynamic filter、drop metrics 和 all-samples hook。
- [ ] 读 [[Slime-RolloutManager]] 时，能定位 DataSource 的 save/load 在 checkpoint 生命周期里的位置。

## 通过标准

全部满足后，应该能借助主线笔记完成这三个动作；修改实现或遇到版本争议时仍需回到 upstream：

- 给别人讲清楚一条 prompt 从文件行到训练样本的生命周期。
- 根据症状选择检查 dataset、buffer、filter、checkpoint 还是 rollout 函数。
- 写出一个最小自定义 DataSource，并知道哪些形状错误会在默认路径中爆出来。
