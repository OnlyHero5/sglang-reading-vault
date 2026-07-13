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
updated: 2026-07-13
---
# 数据源 · 排障指南

## 你为什么要读

这篇按症状排障。每个问题都给出判断方式、源码入口和验证方法，避免停留在“看起来是 DataSource 问题”的泛泛说法。

## 症状一：关掉 global dataset 后默认 rollout 直接失败

判断：默认 `sglang_rollout.generate_rollout` 入口要求 `args.rollout_global_dataset`。关闭 global dataset 意味着 prompt 来源必须由自定义 rollout 或自定义 data source 管理。

```python
# 来源：slime/rollout/sglang_rollout.py L618-L632
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

补充边界：`rollout_global_dataset=True` 但 `prompt_data=None` 时，默认 DataSource 会生成空 `Sample()`，并不是解析阶段 fail fast。普通训练仍应显式配置 prompt 数据；空 Sample 只适合确知后续 custom generate 会完整填充输入的扩展路径。

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

这个预期要求自定义 buffer filter 不超额返回。若它返回多于 N 组，剩余数会变成负值，源码没有断言，dataset offset 甚至可能倒退。

## 症状三：dynamic filter drop 后数据集消耗变快

判断：默认主循环不会把 filtered-out group 自动塞回 buffer。源码里的注释明确说明 unused samples 没有存回 data buffer。

```python
# 来源：slime/rollout/sglang_rollout.py L429-L436
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
# 定位骨架（据 `slime/utils/data.py` L130-L138 删节）：
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

## 症状十：请求 N 组却少返回，offset 还大于 dataset 长度

判断：默认跨 epoch 代码只执行一次“尾部 + 新 epoch 头部”，不是 while 循环。若 dataset 只有 3 条而请求 10 组，当前实现会返回 6 组，并把 `sample_offset` 写成 7。这个状态已经不再是合法的 dataset 内 offset。

处理与预期：保证单次请求不超过“当前尾部 + 一个完整 epoch”的容量，生产配置通常还应让 dataset 长度不小于 `over_sampling_batch_size`。自定义 source 若承诺精确 N，应用循环明确实现多 epoch wrap，并对最终返回长度和 offset 范围做断言。

## 症状十一：空数据集时 rollout 不报 EOF，却一直不前进

jsonl 坏行会被打印并跳过；若全部行无效，或长度过滤移除全部样本，dataset 可能为空。开启 max-length 时还可能先在 `origin_samples[0]` 触发 `IndexError`；未开启时 `get_samples` 持续返回空列表。默认 rollout 与 fully-async 都没有 EOF 终态：前者反复拉取，后者后台重试而前台持续等待。

验证必须在启动生成前断言 `len(data_source) > 0`，并检查有效解析行数、过滤前后数量。不要把空 list 当作可以被默认控制循环正确处理的 EOF 信号。

## 症状十二：配置了多模态字段，但纯文本行被拆成逐字符 content

当 `multimodal_keys` 非空、某条记录却没有任何对应媒体数据时，`multimodals` 字典为空，源码仍构造 `pattern="()"`。`re.split` 会按空匹配切开字符串，例如 `abc` 变成三个 text item。应在数据预检中保证媒体字段与配置一致，或在自定义预处理里对空 `multimodals` 直接保留原文本结构。

## 症状十三：切片语法解析成功，读取时却报 `islice` ValueError

路径正则允许 `@[-2:]`、`@[:-1]` 这样的负数，但后续 `itertools.islice` 不接受负 start/stop。当前广义路径只应使用非负边界；需要尾部切片时先离线确定总行数或使用显式预处理，不能把 Python list 的负切片语义外推到流式 reader。

## 症状十四：shuffle 后随机 RM 或自定义插件的随机序列变化

`Dataset.shuffle` 会执行全局 `random.seed(seed + epoch_id)`。它能重建 dataset permutation，却同时覆盖进程级 RNG。若同一进程还有 `rm_type=random` 或插件使用 Python `random`，跨 epoch/load 会改变它们的后续序列。验证时在 shuffle 前后保存 `random.getstate()`；需要隔离时，自定义 Dataset 应使用局部 `random.Random` 实例。

## 症状十五：fully-async 开了 dynamic filter，却没有 drop metrics

fully-async worker 直接调用 `generate_and_rm_group`，不进入默认 `generate_rollout_async`，因此不会运行 dynamic sampling filter、`MetricGatherer` 或 all-samples hook。它的全局 worker 还复用首次 args/data source。若业务依赖这些控制面能力，应扩展 fully-async 实现或继续使用默认 rollout，不能只复用相同 CLI 参数就假定语义相同。

## 自定义扩展契约

| 扩展点 | 签名/形状 | 核心不变量 |
|--------|-----------|------------|
| `--data-source-path` | 类构造 `__init__(args)` | 实现 `get_samples/add_samples/save/load/__len__` |
| `get_samples` | `N -> list[list[Sample]]` | 若调用方要求固定 batch，应精确返回 N；默认 source 对超大 N/空 dataset 不满足该承诺，内层长度仍须为 `n_samples_per_prompt` |
| `add_samples` | `list[list[Sample]] -> None` | 不破坏 group 形状 |
| `--buffer-filter-path` | `(args, rollout_id, buffer, num_samples)` | 返回后 buffer 状态要一致 |
| `--rollout-all-samples-process-path` | `(args, all_groups, data_source)` | 可观察被 filter drop 的组 |

契约测试入口：

```powershell
Set-Location 'F:\源码阅读\slime'
python -m pytest tests/plugin_contracts/test_plugin_path_loading_contracts.py -q
```

预期现象：测试会校验默认 data source、buffer filter、自定义 data source 路径加载和最小返回形状。若缺少依赖导致 collection 失败，先记录缺失模块，不要把它误判为 DataSource 逻辑错误。
