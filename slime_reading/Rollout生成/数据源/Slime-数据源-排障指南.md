---
title: "数据源 · 排障指南"
type: troubleshooting
framework: slime
topic: "数据源"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# 数据源 · 排障指南

## 你为什么要读

这篇按症状排障。每个问题都给出判断方式、源码入口和验证方法，避免停留在“看起来是 DataSource 问题”的泛泛说法。

## 症状一：关掉 global dataset 后默认 rollout 直接失败

判断：默认 `sglang_rollout.generate_rollout` 入口要求 `args.rollout_global_dataset`。关闭 global dataset 意味着 prompt 来源必须由自定义 rollout 或自定义 data source 管理。

```python
# 来源：slime/rollout/sglang_rollout.py L618-L633
def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to get and store samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
    """
    assert args.rollout_global_dataset
```

处理：

- 使用默认 SGLang rollout 时，保持 `--rollout-global-dataset` 并配置 `--prompt-data`。
- 如果 prompt 来自外部服务、睡眠 rollout 或自定义 buffer，同时替换 `--rollout-function-path` 和必要的 `--data-source-path`。

验证：启动前检查参数；启动后若断在 `generate_rollout` 的 assert，说明不是 tokenizer 或文件格式问题，而是 rollout 函数与数据源模式不匹配。

## 症状二：buffer 有样本，但好像仍然读了 dataset

判断：这是默认语义。`get_samples(N)` 先从 buffer 取，buffer 不足时只向 dataset 补缺口。它不会因为 buffer 非空就完全停止读 dataset。

```python
# 来源：slime/rollout/data_source.py L177-L189
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples
```

处理：

- 想让 buffer 完全承担读取，必须保证 buffer 长度始终不小于请求 N，或自定义 DataSource。
- 想改变出队顺序，替换 `--buffer-filter-path`，不要改 dataset 游标逻辑。

验证：给 buffer 放 B 组，调用 `get_samples(N)` 后，预期返回 N 组，其中前 B 组来自 buffer，dataset 的 `sample_offset` 只增加 `N-B`。

## 症状三：dynamic filter drop 后数据集消耗变快

判断：默认主循环不会把 filtered-out group 自动塞回 buffer。源码里的注释明确说明 unused samples 没有存回 data buffer。

```python
# 来源：slime/rollout/sglang_rollout.py L429-L437
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
```

处理：

- 这是过采样设计的代价，不是 DataSource 重复消费。
- 需要观察全部完成样本时，用 `--rollout-all-samples-process-path` 接收 `all_samples`。
- 需要回收 drop 样本时，需要自定义后处理或 fork 默认 rollout 主循环。

验证：比较 `rollout/dynamic_filter` 类 metric、有效训练样本数和 `sample_offset` 增长；drop 越多，offset 相对有效 batch 前进越快。

## 症状四：partial rollout 没有复用半成品

判断：回灌只在 `abort` 返回 `aborted_samples` 后发生，且 data source 必须支持 `add_samples`。纯 `RolloutDataSource` 会抛 `RuntimeError`。

```python
# 来源：slime/rollout/sglang_rollout.py L637-L640
    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output
```

```python
# 来源：slime/rollout/data_source.py L120-L121
    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")
```

处理：

- 保持默认 `RolloutDataSourceWithBuffer`。
- 确认开启 `--partial-rollout`。
- 确认 `abort` 时确实还有 pending task，且 task result 中有可回收 group。

验证：看日志中是否出现 `Collected <count> partial samples into the data buffer` 这类信息；下一轮 `get_samples` 应先从 buffer 弹出。

## 症状五：自定义 buffer filter 后样本重复出现

判断：默认 `pop_first` 会原地删除返回的 group。自定义 filter 如果只返回不删除，就会让同一组反复被取出。

```python
# 来源：slime/rollout/data_source.py L225-L229
def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
```

处理：

- 自定义 filter 必须维护 buffer 状态，或明确实现“可重复采样”的业务语义。
- 返回数量不要超过 `num_samples`。
- 返回值必须仍是 `list[list[Sample]]`。

验证：调用 filter 前后比较 `len(buffer)`；正常 FIFO 下长度会减少返回 group 数。

## 症状六：`len(data_source)` 是 0，epoch 步数不对

判断：默认 `__len__` 只看 dataset，不看 buffer。dataset 不存在时返回 0。

```python
# 来源：slime/rollout/data_source.py L162-L165
    def __len__(self) -> int:
        if self.dataset is None:
            return 0
        return len(self.dataset)
```

处理：

- 检查 `--prompt-data` 是否存在且格式受支持。
- 检查 `--rollout-max-prompt-len` 是否把全部样本过滤掉。
- 不要用 buffer 长度推断 `get_num_rollout_per_epoch`。

验证：打印或断点查看 `data_source.dataset is None`、`len(data_source.dataset.samples)`、`len(data_source.buffer)`，这三个值含义不同。

## 症状七：续训后 prompt 顺序错位

判断：续训需要同时恢复 `epoch_id` 和 `sample_offset`。如果开启 shuffle，load 后还要重放同一 epoch 的 shuffle。

```python
# 来源：slime/rollout/data_source.py L152-L160
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle and self.dataset is not None:
            self.dataset.shuffle(self.epoch_id)
```

处理：

- 确认 `args.load` 指向正确 checkpoint 根目录。
- 确认 `global_dataset_state_dict_{rollout_id}.pt` 存在。
- 确认续训使用同一份 prompt 数据和同一 `rollout_seed`。

验证：load 后断点检查 `epoch_id`、`sample_offset`、`dataset.epoch_id` 是否一致；然后手动调用一次 `get_samples(1)` 看 prompt 是否符合预期。

## 症状八：多模态 prompt 长度过滤没有生效

判断：`filter_long_prompt` 只在 prompt 是字符串时做长度检查。未应用 chat template 的 list prompt 会直接 warning 并跳过检查。

```python
# 来源：slime/utils/data.py L81-L90
def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_length: int | None) -> list[Sample]:
    if max_length is None:
        return origin_samples

    if not isinstance(origin_samples[0].prompt, str):
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return origin_samples
```

处理：

- 对 messages 数据开启 `--apply-chat-template`，让 prompt 变成可估长字符串。
- 对多模态数据确认 processor 可用，避免用纯 tokenizer 低估长度。

验证：看日志中的 skip warning 和 `Filtered <count> samples longer than max_length=<limit>` 这类信息；没有过滤日志不代表所有样本都短，可能是路径被跳过。

## 症状九：jsonl 行看起来没问题，但生成阶段 prompt 是空的

判断：`Dataset` 用 `prompt_key` 从记录取 prompt。字段名不匹配时，`data.get(prompt_key)` 会得到 `None`，错误可能延迟到模板、processor 或 tokenizer 阶段。

```python
# 来源：slime/utils/data.py L130-L138
def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]
```

处理：

- 检查 `--input-key` 是否与 jsonl 字段一致。
- 检查 `--label-key`、`--metadata-key`、`--tool-key` 是否存在或允许为空。
- messages 数据配合 `--apply-chat-template` 使用。

验证：在 `Dataset.__init__` 后看第一条 `dataset.samples[0].prompt`，不要等到 SGLang 请求失败才查。

## 自定义扩展契约

| 扩展点 | 签名/形状 | 核心不变量 |
|--------|-----------|------------|
| `--data-source-path` | 类构造 `__init__(args)` | 实现 `get_samples/add_samples/save/load/__len__` |
| `get_samples` | `N -> list[list[Sample]]` | 外层 group 数不超过语义请求，内层长度为 `n_samples_per_prompt` |
| `add_samples` | `list[list[Sample]] -> None` | 不破坏 group 形状 |
| `--buffer-filter-path` | `(args, rollout_id, buffer, num_samples)` | 返回后 buffer 状态要一致 |
| `--rollout-all-samples-process-path` | `(args, all_groups, data_source)` | 可观察被 filter drop 的组 |

契约测试入口：

```powershell
$env:PYTHONPATH='F:\源码阅读\slime'
python -m pytest slime/tests/plugin_contracts/test_plugin_path_loading_contracts.py -q
```

预期现象：测试会校验默认 data source、buffer filter、自定义 data source 路径加载和最小返回形状。若缺少依赖导致 collection 失败，先记录缺失模块，不要把它误判为 DataSource 逻辑错误。
