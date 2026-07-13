---
title: "RolloutManager · 排障指南"
type: troubleshooting
framework: slime
topic: "RolloutManager"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# RolloutManager · 排障指南

## 你为什么要读

本页是 RolloutManager 的排障入口。读完后，你应该能把 generate 返回形态、`rollout_id` 分组、debug 路径、reward normalization、remove sample 和 DP schedule 问题分别落到对应源码入口。

## 1. generate 返回什么？

**症状：** 以为 `generate` 返回 `list[Sample]`，训练侧却拿到 `Box`。

**原因：** 正常训练路径会把 Sample 转成 `train_data`，再按 DP rank `ray.put`。返回值是 `list[Box]`，长度等于 `dp_size`。

源码入口：来源：slime/ray/rollout.py L546-L559

源码入口：来源：slime/ray/rollout.py L853-L895

**验证：** 打印返回对象长度应等于 `train_parallel_config["dp_size"]`；每个 Box 内部是 Ray ObjectRef。

## 2. `generate(rollout_id)` 和 `Sample.rollout_id` 是一回事吗？

**症状：** compact/subagent rollout 的 loss 分母明显偏大或偏小。

**原因：** `generate(rollout_id)` 是训练循环 step；`Sample.rollout_id` 是 loss 聚合分组。一次 rollout execution 产出多条训练样本时，这些 sibling 必须共享同一个 `Sample.rollout_id`。

源码入口：来源：slime/utils/types.py L97-L106

源码入口：来源：slime/ray/rollout.py L898-L927

**验证：** 对 compact 三层输出构造缺失 `rollout_id` 的 sibling，预期 `_validate_rollout_id_annotated` 报错。

## 3. debug_rollout_only 和 debug_train_only 有何区别？

**症状：** 开了 debug 后训练没跑，或者 SGLang server 没启动。

**原因：**

- `debug_train_only` 在 RolloutManager 初始化时跳过 SGLang servers。
- `debug_rollout_only` 允许生成 Sample 和日志，但在 convert/split 前直接返回。

源码入口：来源：slime/ray/rollout.py L430-L436

源码入口：来源：slime/ray/rollout.py L555-L559

**验证：** debug rollout-only 下应看到 Sample 日志或 debug dump，但没有 `list[Box]` 返回给训练。

## 4. load_debug_rollout_data 如何复现问题？

**症状：** 想复现训练数据构造 bug，但不想重新跑 SGLang。

**原因：** `_get_rollout_data` 可以从磁盘读取 `samples`，恢复成 `Sample` 后继续走 convert/split。这个路径跳过 SGLang 请求和 rollout 函数。

源码入口：来源：slime/ray/rollout.py L635-L665

源码入口：来源：slime/ray/rollout.py L667-L684

**验证：** 使用保存出的 debug 文件，预期参数归一化强制 `debug_train_only=True`、metrics 为 None，但 `_convert_samples_to_train_data` 和 `_split_train_data_by_dp` 仍执行。注意这条路径恢复的是扁平 Sample，不会重新验证 compact sibling 的嵌套 `rollout_id` 契约。

## 5. reward normalization 为什么要在 RolloutManager 做？

**症状：** 把 reward 后处理放到训练 rank 后，GRPO 类算法结果不一致。

**原因：** group normalization 需要看到整个 rollout batch。DP split 后，每个 rank 只看到局部样本，无法可靠按 prompt group 归一化。但默认实现只在固定 fanout 总数匹配时 reshape 成 prompt groups；可变 fanout fallback 会把整批当一组。

源码入口：来源：slime/ray/rollout.py L686-L711

**验证：** 固定 fanout 检查 `raw_reward` 与 `rewards`；可变 fanout 应启用按 `group_index` 分组的 custom reward hook，并用每组均值接近 0 验证。不要把默认 fallback 的整批均值为 0 当作逐组正确。

## 6. 为什么 remove_sample 不直接删除样本？

**症状：** `remove_sample=True` 的样本仍然出现在 train_data 中。

**原因：** 源码选择保留样本结构，但把 loss mask 置零。这样日志、partition、rollout_id 和 batch 形态仍稳定。

源码入口：来源：slime/ray/rollout.py L747-L761

**验证：** 对 remove sample 检查 `loss_masks` 应全 0，`tokens` 仍在。

## 7. DP schedule 为什么按 rollout 数切 step？

**症状：** sample 数很多，但 `global_batch_size` 判断仍然失败。

**原因：** 训练 step 的单位是 rollout，不是 sample。同一 rollout 的多个 sample 必须在同一步内，避免 loss 聚合被拆散。

源码入口：来源：slime/utils/dp_schedule.py L82-L209

**验证：** 看 unique `rollout_ids` 数量，而不是 `len(samples)`。如果 unique rollout 数小于 `global_batch_size`，会直接 assert。

补充：若 unique rollout 数大于但不能整除 `global_batch_size`，尾部不足一整 step 的 rollout 会被静默排除。对照 partitions 的并集确认哪些 samples 真正进入训练。

## 8. static micro batch 为什么会报对齐错误？

**症状：** 静态 micro batch size 下报 `num_mbs` 不是 `dp_size * mb_group` 的倍数。

**原因：** 静态路径不能随便拆 micro-batch，否则破坏固定大小不变量；源码在无法对齐时直接抛 AssertionError。动态路径才会尝试拆分最大 bin。

源码入口：来源：slime/utils/dp_schedule.py L167-L185

**验证：** 调整 `global_batch_size`、`micro_batch_size`、`dp_size` 或 VPP mb group，使 step 内 mbs 数天然对齐。

若开启 `balance_by_flops`，它只允许动态 batch，并明确不保证 token cap；出现 OOM 时要打印每个实际 bin 的 token 和，不能仅检查 `max_tokens_per_gpu` 配置值。

## 9. train_parallel_config 缺失该查哪里？

**症状：** `_split_train_data_by_dp` 访问 `self.train_parallel_config` 失败。

**原因：** 训练 actor 的 rank 0 会在 `set_rollout_manager` 时把 config 写回 RolloutManager。正常启动顺序要求 training models 先创建，再进入首次 generate。

源码入口：来源：slime/ray/train_actor.py L125-L128

源码入口：来源：slime/ray/rollout.py L826-L827

**验证：** 确认 `create_training_models` 已完成，且不是跳过了 `set_rollout_manager`。

## 10. object-store 和 nixl 有何区别？

**症状：** 切换 `rollout_data_transport` 后不清楚是否改变数据结构。

**原因：** 两者只改变 Ray put 的传输方式，不改变 `rollout_data` 字段结构。`nixl` 路径需要创建 RolloutManager 时启用 tensor transport。

源码入口：来源：slime/ray/placement_group.py L220-L230

源码入口：来源：slime/ray/rollout.py L887-L895

**验证：** 两种模式下 `list[Box]` 的长度和字段应一致；区别在 ObjectRef 的 tensor transport。

## 11. 为什么权重更新只拿第一个 updatable server？

**症状：** 多模型 rollout 中 reference/reward 没有收到权重更新。

**原因：** 源码明确只返回第一个 `update_weights=True` 的 server。frozen 模型被排除，多 policy 同时更新尚未支持。

源码入口：来源：slime/ray/rollout.py L511-L540

**验证：** 检查 SGLang config 中只有 policy server 配 `update_weights=True`。

若确实配置了多个可更新模型，当前只取 `servers` 插入顺序中的第一个；需要多模型同步时不能假设 manager 会自动遍历全部目标。

## 12. custom converter 或混合可选字段为什么晚到 split/训练才报错？

**症状：** custom converter 已成功返回，却在 `_split_train_data_by_dp` 报 `KeyError`；或混合来源 batch 的 logprob、teacher、MoE 字段被整列忽略。

**原因：** custom converter 直接绕过默认转换，当前没有独立 schema validator，split 隐式要求 `tokens/rollout_ids`。默认 converter 的多数可选列又用 `samples[0]` 判断是否启用，第一条样本代表了整批字段形态。

**验证：** custom 输出至少检查 `len(tokens) == len(rollout_ids)`，并为所有需要按 partition 切的列检查等长；混合来源数据应先统一字段存在性，不能依赖后续自动补齐。

## 13. offload_rollout 的时序是什么？

**症状：** 训练和 rollout 抢显存，或者 generate 时 engine 还没恢复。

**原因：** 训练主循环中 generate 后 offload rollout，训练后 onload weights，update_weights，再 onload KV。RolloutServer/ServerGroup 只对 `needs_offload=True` 的组执行内存释放和恢复。

源码入口：来源：train.py L22-L33

源码入口：来源：train.py L67-L92

源码入口：来源：slime/ray/rollout.py L578-L593

**验证：** generate 前应已 onload weights/KV；generate 后才能 offload 给训练。

## 排障顺序

1. 先确认是否 debug-only 或 load-debug 路径。
2. 再确认 rollout 函数输出形态和 `Sample.rollout_id`。
3. 再看 reward/mask 是否在 convert 阶段符合预期。
4. 再看 unique rollout 数、`global_batch_size`、DP/VPP 对齐。
5. 最后检查 ObjectRef transport 和训练 actor config 回填。

## 运行验证

RolloutManager FAQ 的排障可以先用源码检索覆盖 debug-only、rollout id、reward normalization、DP schedule、ObjectRef transport、权重更新和 offload 时序。

```powershell
rg -n 'debug_rollout_only|debug_train_only|load_debug_rollout_data|def generate\(|_validate_rollout_id_annotated|_post_process_rewards|remove_sample|build_dp_schedule|ray\.put|get_updatable_engines_and_lock|offload|onload|update_weights' slime/train.py slime/slime/ray/rollout.py slime/slime/utils/arguments.py slime/slime/utils/dp_schedule.py
```

读输出时先看 `arguments.py` 是否改写 debug 路径；再看 `rollout.py` 的 `generate/_post_process_rewards/_split_train_data_by_dp`。如果问题发生在训练前，优先看 `ray.put` 和 `build_dp_schedule`；如果问题发生在权重同步前后，看 `get_updatable_engines_and_lock` 与 `offload/onload`。
