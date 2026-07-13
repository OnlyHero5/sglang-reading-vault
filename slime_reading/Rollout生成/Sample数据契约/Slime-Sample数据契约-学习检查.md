---
title: "Sample数据契约 · 学习检查"
type: exercise
framework: slime
topic: "Sample数据契约"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# Sample数据契约 · 学习检查

这份清单检查你是否能维护 Sample 的 response 时间轴，而不是背字段名。

## 读者能做什么

- [ ] 能画出 `DataSource → generate → Sample.append_response_tokens → RolloutManager flatten → train_data → DP ObjectRef`。
- [ ] 能解释 `tokens`、`response_length`、`loss_mask`、`rollout_log_probs` 四者如何对齐。
- [ ] 能区分 trainable token 与 non-trainable token 的 logprob 和 loss mask 规则。
- [ ] 能说明 top-p replay 按 response offsets 对齐，也能解释 routed experts 为什么按 `len(tokens)-1` 对齐且不增量 merge。
- [ ] 能解释 `finish_reason` 如何映射到 `Sample.Status`。
- [ ] 能说明 `to_dict/from_dict` 为什么要处理 enum、nested info 和未知字段。
- [ ] 能解释普通 rollout 与 compact/subagent rollout 对 `rollout_id` 的不同要求。
- [ ] 能说明 `remove_sample=True` 为什么不是删除样本，而是把 loss mask 置零。
- [ ] 能描述 `RolloutFnTrainOutput`、legacy list、`call_rollout_fn` 三者关系。

## 排障验收

- [ ] 看到 “trainable response tokens require rollout log probabilities” 时，能定位到自定义 generate 缺 `log_probs`。
- [ ] 看到 top-p offsets 报错时，能检查 offsets 首项、末项、长度和 token id 总数。
- [ ] 看到 compact rollout_id assert 时，能回到 flatten 前的嵌套 sibling list。
- [ ] 看到 reward dict 进入训练时报错时，能检查 `args.reward_key`。
- [ ] 看到 debug load 字段丢失时，能用 `to_dict/from_dict` round-trip 定位。
- [ ] 看到 tool token 参与 loss 时，能检查 append 时是否使用 `trainable=False`。
- [ ] 看到 round-trip 通过时，仍能说明它只证明字段保真，不证明 response 时间轴合法。

## 可执行验证

优先跑轻量 CPU 单测：

```powershell
$env:PYTHONPATH='F:\源码阅读\slime'
python -m pytest slime/tests/test_sample.py -q
```

预期覆盖：

- status enum 序列化和恢复。
- `spec_info`、`prefix_cache_info` round-trip。
- 未知字段作为动态属性保留。
- `finish_reason` 到 `Sample.Status` 的映射。
- prefix cache 统计跨调用累加。
- speculative metrics 只在对应 flag 开启时累加。

再跑 top-p 与 metric 相关单测：

```powershell
$env:PYTHONPATH='F:\源码阅读\slime'
python -m pytest slime/tests/test_rollout_metrics.py -q
```

预期覆盖：

- top-p kept vocab metric 只统计 loss mask 为 1 的 token。
- `remove_sample=True` 的样本不贡献 top-p metric。
- base64 int32 metadata 能解码成 tensor。
- 多段 top-p replay offsets 能正确合并。
- non-trainable token 会 padding top-p offsets。
- trainable 缺 logprob、non-trainable 带 logprob 都会报错。

插件契约可选验证：

```powershell
$env:PYTHONPATH='F:\源码阅读\slime'
python -m pytest slime/tests/plugin_contracts/test_plugin_rollout_contracts.py -q
```

这类测试需要环境能正常导入相关模块。当前基线直接运行时：`test_sample.py` 缺 `httpx`，`test_rollout_metrics.py` 缺 `ray`。只 stub Sample 未使用的 `http_utils.is_port_available` 后，原测试文件 `12 passed`；top-p metric 从当前源码 AST 抽取函数体的两项检查通过。两者仍伴随 Torch/NumPy ABI 警告，不能替代完整 Ray/SGLang 测试。

## 源码复述题

- [ ] 为什么 `append_response_tokens` 不允许 trainable token 缺 logprob？
- [ ] 为什么非训练 token 仍然增加 `response_length`？
- [ ] 为什么 top-p replay 不能只有 token ids？
- [ ] 为什么 compact rollout 校验必须发生在 flatten 前？
- [ ] 为什么 `rollout_mask_sums` 要按 `rollout_id` 聚合后再广播回每条 sample？

## 下一步

继续读 [[Slime-SGLang-Rollout]] 可以看到默认 generate 如何填充 `Sample`；继续读 [[Slime-训练数据]] 可以看到列式 `train_data` 如何被 Megatron data iterator 消费。
