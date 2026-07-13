---
title: "ModelLoader · 学习检查"
type: exercise
framework: sglang
topic: "ModelLoader"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-11
---
# ModelLoader · 学习检查

## 你为什么要读

本篇用来判断你是否真的读懂了 ModelLoader，而不是只看过 loader 类列表。验收目标是：拿到一个加载异常或一个新 checkpoint 格式时，能沿权重装载生产线定位问题。

## 1. 能画出主线

- [ ] 能画出默认 HF 基线，并在写入点画出 model/parameter loader、rank-local state direct copy、remote parameter transfer 三条分支。
- [ ] 能说明 `DefaultModelLoader` 负责文件账和 iterator，不负责所有模型名字规则。
- [ ] 能说明模型类 `load_weights` 负责 checkpoint name 到参数名的映射、跳过和融合。
- [ ] 能说明默认全量 tensor 常在 parameter loader 切片，同时列出 presharded、BitsAndBytes、ShardedState、Remote KV/Instance 例外。

自测题：如果某个 rank 的 `q_proj.weight` shape mismatch，你会先打开 `loader.py`、模型类 `load_weights`，还是 `layers/linear.py`？请说出理由。

## 2. 能定位五类失败

| 症状 | 你应能定位到 | 预期判断 |
|------|--------------|----------|
| `Cannot find any model weights` | `_prepare_weights` | allow pattern、路径、HF 下载、safetensors index 过滤 |
| `Unexpected extra config keys` | loader `__init__` | 当前 loader 不接受这个 `model_loader_extra_config` |
| `Parameter <name> not found in params_dict` | 模型类 `load_weights` | name remap、跳过规则、架构不匹配 |
| `param_data.shape != loaded_weight.shape` | 当前写入协议 | TP narrow、padding、presharded、quant iterator、rank state/direct transfer |
| 量化模型启动后输出异常 | `load_weights_and_postprocess` | 后处理没有执行、设备上下文不对、layout 未 repack |

验收标准：任选一个症状，能说清“错误发生在配置账、文件账、transport 账、参数账、后处理账中的哪一本”。

## 3. 能区分冷启动和热更新

- [ ] 冷启动通过 `ModelRunner.load_model` 选择 loader 并返回 `nn.Module`。
- [ ] 从磁盘热更新复用 `DefaultModelLoader._get_weights_iterator`，但不重建模型。
- [ ] 分布式热更新和 tensor 热更新绕过文件层，直接构造 `(name, tensor)`。
- [ ] `FlattenedTensorBucket` 只是 transport 容器，重建后仍回到 `(name, tensor)`。
- [ ] 能解释磁盘更新的“Rolling back”为什么不证明恢复旧 checkpoint，并按可能部分写入处置。

自测题：`load_format="flattened_bucket"` 时，模型类会看到 bucket 还是 reconstructed tensors？请指出数据形态变化。

## 4. 能解释特殊 loader 改了哪一段

| loader | 改变的边界 | 不变的约束 |
|--------|------------|------------|
| `DummyModelLoader` | 不读真实 checkpoint | 仍交付可执行模型对象 |
| `ShardedStateLoader` | rank 文件 + state dict direct copy | 参数集合/shape 必须匹配，并显式 post-load |
| `BitsAndBytesModelLoader` | iterator 内量化/部分 TP 切片 | 预量化 TP>1 被拒绝，不能再二次 narrow |
| `GGUFModelLoader` | 文件解析和参数 materialize | 参数 shape/dtype 仍要匹配 |
| `LayeredModelLoader` | meta 初始化、逐层 materialize | 模型类必须支持逐模块加载 |
| `RemoteModelLoader` | FS 走模型 loader，KV 按 rank direct copy | 先识别 connector 类型，不能假定同一写入协议 |
| `RemoteInstanceModelLoader` | 按参数顺序/地址从另一实例取值 | 源/目标参数集合、顺序或地址 metadata 必须一致 |
| `ModelOptModelLoader` | 合并加载和量化工作流 | 后续执行态仍受量化 layout 约束 |

验收标准：看到新 loader 时，先问它替换的是“权重来源、文件 iterator、模型初始化、参数写入、量化后处理”中的哪一段。

## 5. 能做最小验证

任选一种验证方式执行或口述预期现象：

| 验证 | 操作 | 预期 |
|------|------|------|
| 文件选择 | 本地列出 `*.safetensors`、`model.safetensors.index.json`、`*.bin`、`*.pt` | 能解释 `auto`、`safetensors`、`pt` 会匹配哪些文件 |
| iterator 内存 | 切换 threads、mmap、prefetch、drop-cache 并记录 RSS/page cache | 能解释 generator 为何仍可能同时持有多个完整 shard |
| rank-local 化 | 对比默认 RowParallel、presharded 与 BnB/RankState | 能证明每条路线恰好切一次，而非背诵固定切片点 |
| 名字映射 | 选一个 checkpoint name 手算模型类 `load_weights` 的 remap | 能判断最终是否进入 `params_dict` |
| 热更新 | 对比 `update_weights_from_disk` 的 iterator 失败和写入失败返回消息 | 能区分文件层失败和参数层失败 |

## 6. 能迁移到相邻专题

- [ ] 读 [[SGLang-ModelRunner]] 时，能把 ModelLoader 放回启动和执行态初始化之间。
- [ ] 读 [[SGLang-通用模型]] 时，能重点看每个模型类的 `load_weights`。
- [ ] 读 [[SGLang-Quantization]] 时，能把加载后处理和 quant kernel layout 连接起来。
- [ ] 读 [[Slime-分布式权重同步]] 时，能区分冷启动装载和运行时权重同步。

## 通过标准

如果你能不用翻文档回答下面四个问题，就算本专题通过：

1. 为什么 loader 找到权重文件，不代表本 rank 已经拿到正确参数分片？不同路线在哪里完成 rank-local 化？
2. 为什么 `remote_instance` 配置不完整时可能悄悄回到 `auto`？
3. 为什么 `FlattenedTensorBucket` 不应该被理解成新的模型权重语义？
4. 为什么不能只检查 `model.load_weights`，还要区分 quant process、model post-load fixup 与 KV scale？
