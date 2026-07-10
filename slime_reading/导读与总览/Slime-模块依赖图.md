---
title: "Slime 模块依赖图"
type: map
framework: slime
topic: "导读与总览"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/map
  - source-reading
updated: 2026-07-10
---
# Slime 模块依赖图

> Mermaid 模块关系 + import 示例 · 用来判断跨层依赖边界

---

## 你为什么要读

Slime 的依赖关系不能只看 Python import：PlacementGroup 先决定资源，RolloutManager 和 RayTrainGroup 再占座，训练结果最后通过权重同步回到 SGLang。本页画的是创建顺序、远程引用和数据依赖，帮助你区分“代码能 import”与“系统能闭环”这两件完全不同的事。

## 分层依赖（Mermaid）

```mermaid
flowchart TB
    subgraph Entry["入口与编排"]
        train["train.py"]
        args["utils/arguments.py"]
        pg["ray/placement_group.py"]
        rg["ray/actor_group.py"]
    end

    subgraph Rollout["Rollout 生成"]
        rm["ray/rollout.py"]
        sgr["rollout/sglang_rollout.py"]
        ds["rollout/data_source.py"]
        types["utils/types.py"]
    end

    subgraph SGLang["SGLang 后端"]
        eng["sglang_utils/sglang_engine.py"]
        cfg["sglang_utils/sglang_config.py"]
    end

    subgraph Megatron["Megatron 训练"]
        actor["megatron_utils/actor.py"]
        model["megatron_utils/model.py"]
        loss["megatron_utils/loss.py"]
    end

    subgraph Sync["权重同步"]
        uwd["update_weight_from_distributed.py"]
        m2hf["megatron_to_hf"]
    end

    train --> args
    train --> pg
    train --> rg
    pg --> rm
    pg --> rg
    rm --> sgr
    rm --> ds
    rm --> types
    rm --> eng
    rm --> cfg
    rg --> actor
    actor --> model
    actor --> loss
    actor --> uwd
    actor --> m2hf
    uwd --> eng
```

---

## train.py 导入链

**依赖读法：** 入口层只依赖 Ray 编排与 arguments；不直接 import Megatron/SGLang 实现。

**源码锚点：**

```python
## 来源：train.py L1-L6
import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action
```

---

## RolloutManager 导入链

**依赖读法：** rollout.py 是 Rollout 层 hub，连接 sglang backend、rollout fn、types。

**源码锚点（典型 import 模式）：**

```python
## 来源：slime/ray/rollout.py L1-L30（节选）
from slime.backends.sglang_utils.sglang_config import SglangConfig
from slime.rollout.base_types import call_rollout_fn
from slime.utils.misc import load_function
from slime.utils.types import Sample
```

---

## Megatron Actor 导入链

**依赖读法：** actor.py 依赖 model、loss、update_weight 子模块；通过 Ray 持有 rollout_manager 引用。

**源码锚点：**

```python
## 来源：slime/backends/megatron_utils/actor.py L1-L40（节选）
from slime.backends.megatron_utils.loss import compute_advantages_and_returns, loss_function
from slime.backends.megatron_utils.model import initialize_model_and_optimizer, train
from slime.utils.misc import Box
```

---

## 权重同步依赖

```mermaid
flowchart LR
    actor["MegatronTrainRayActor.update_weights"]
    common["update_weight/common.py"]
    dist["update_weight_from_distributed.py"]
    hf["hf_weight_iterator_direct.py"]
    m2hf["megatron_to_hf.convert_to_hf"]
    engine["SGLangEngine.update_weights"]

    actor --> common
    actor --> dist
    dist --> hf
    dist --> m2hf
    dist --> engine
```

→ [[Slime-分布式权重同步-数据流]]

---

## 定制 hook 依赖

**依赖读法：** 所有 `--*-path` 参数经 `load_function` 动态加载，编译期无 import。

**源码锚点：**

```python
## 来源：slime/utils/misc.py L37-L45
def load_function(path):
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
```

→ [[Slime-自定义扩展-源码走读]]

---

## 与 SGLang upstream 边界

| Slime 模块 | SGLang 依赖方式 |
|-----------|----------------|
| `sglang_engine.py` | subprocess 启动 `sglang serve` |
| `sglang_rollout.py` | HTTP 调 router `/generate` |
| `arguments.py` | `sglang_parse_args()` 透传 CLI |

Slime **不 fork** SGLang 源码；见 [[Slime与SGLang-阅读对照]]。

---

## 导航

- [[Slime-架构分层]]
- [[Slime-源码地图]]
