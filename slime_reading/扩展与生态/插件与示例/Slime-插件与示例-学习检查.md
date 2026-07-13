---
title: "插件与示例 · 学习检查"
type: exercise
framework: slime
topic: "插件与示例"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-13
---
# 插件与示例 · 学习检查

这一页验收读者是否能把 example 的边界迁移到自己的项目，并区分“接口 smoke 通过”和“依赖服务下的真实 workflow 可运行”。

## 1. 读者自测

- [ ] 能说明 Search-R1 为什么是 `custom_generate + custom_rm`。
- [ ] 能指出 multi_agent 当前使用 `custom_generate`，不是 `rollout_function`。
- [ ] 能解释 multi_agent 为什么共同 `rollout_id` 仍不足以修复变量 fan-out 的 reward normalization。
- [ ] 能解释 Search-R1 中 model token 与 observation token 的 loss mask 分界。
- [ ] 能说明 rollout_buffer 的 `/start_rollout`、`/buffer/write`、`/get_rollout_data` 各自职责。
- [ ] 能说出 generator 的必需符号：`TASK_TYPE` 和 `run_rollout`。
- [ ] 能指出 README 的 `_generator.py`/五个 optional 与发现器的 `*.py`/三个 optional 漂移。
- [ ] 能解释 rollout_buffer wrapper 如何把 OpenAI messages 转成 `Sample`。
- [ ] 能说明 rollout_buffer 为什么不是持久化、可恢复的生产队列。
- [ ] 能区分 runnable example 和 GLM5 这种模型结构插件。

## 2. 示例入口定位

在 vault 根目录运行：

```powershell
rg -n 'custom-generate-function-path|custom-rm-path|rollout-function-path' slime/examples slime/slime_plugins
rg -n 'TASK_TYPE|run_rollout|start_rollout|buffer/write|get_rollout_data' slime/examples slime/slime_plugins
```

期望：

- 能判断每个 example 替换的是单样本 generate、reward、整轮 rollout 还是模型结构。
- 能定位外部检索、agent system、rollout buffer HTTP 服务等必须单独启动的依赖。
- 看到 README 中的命令后，能反查到真正 callable，而不是把示例目录名当插件路径。

## 3. example 级静态 smoke

这些 example 需要真实 tokenizer、模型参数或 HTTP 服务，不能直接拿 plugin contract test 的 reference args 调用。先在 vault 根目录运行不会启动服务的检查：

```powershell
python -m py_compile slime/examples/search-r1/generate_with_search.py slime/examples/multi_agent/rollout_with_multi_agents.py slime/examples/multi_agent/agent_system.py slime/slime_plugins/rollout_buffer/buffer.py slime/slime_plugins/rollout_buffer/rollout_buffer_example.py slime/slime_plugins/rollout_buffer/generator/base_generator.py slime/slime_plugins/models/glm5/glm5.py
```

再运行结构断言：

```powershell
@'
import ast
from pathlib import Path

def functions(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return {node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

search = functions("slime/examples/search-r1/generate_with_search.py")
multi = functions("slime/examples/multi_agent/rollout_with_multi_agents.py")
agent = Path("slime/examples/multi_agent/agent_system.py").read_text(encoding="utf-8")
buffer = Path("slime/slime_plugins/rollout_buffer/buffer.py").read_text(encoding="utf-8")
wrapper = Path("slime/slime_plugins/rollout_buffer/rollout_buffer_example.py").read_text(encoding="utf-8")
generator = Path("slime/slime_plugins/rollout_buffer/generator/base_generator.py").read_text(encoding="utf-8")

assert isinstance(search["generate"], ast.AsyncFunctionDef)
assert [a.arg for a in search["generate"].args.args] == ["args", "sample", "sampling_params"]
assert isinstance(multi["generate_with_multi_agents"], ast.AsyncFunctionDef)
assert "s.rollout_id = input_rollout_id" in agent
assert "zip(args.results_dict" in agent and "strict=False" in agent
assert 'glob.glob(str(generator_dir / "*.py"))' in buffer
assert '"transform_group"' in buffer and "def normalize_group_data" in generator
assert "time.sleep(5)" in wrapper and "requests.post" in wrapper
print("PASS: example signatures and documented prototype boundaries are present")
'@ | python -
```

预期：编译命令退出码为 0，结构脚本输出 `PASS:`。若失败，说明 upstream 已变，需重新阅读而不是只改行号。

## 4. 真实 workflow 验收

不要直接在共享开发机运行 example shell；脚本会强制终止 SGLang、Ray 和 Python 进程。应在专用容器里按以下顺序验证：

1. Search-R1：先启动 retrieval service，再用 1 prompt、1 response 验证 stop tag、token/logprob/loss-mask 等长；预期 observation mask 全为 0。
2. multi_agent：先把 `num_parallel` 降到 1，记录每个输入实际 fan-out 数、共同 `rollout_id` 和 normalization 口径；预期空 list、`None` reward 都被显式拒绝。
3. rollout_buffer：先独立请求 `/start_rollout`、`/buffer/write`、`/get_rollout_data`，再启动训练 wrapper；预期服务重启、重复 start、超额 group 和不可达场景都有有界失败。
4. GLM5：先只构造 spec，再做单 microbatch forward；预期每个 PP stage 从 computing layer 开始，skip layer 不含 indexer 参数。

如果环境不具备模型/GPU/外部服务，只能完成静态 smoke，并必须把真实 workflow 标记为未运行，不能用 `py_compile` 冒充训练通过。

## 5. 收官衔接

完成本专题后，扩展生态主线已经接到 customization。继续阅读 [[Slime-总结复盘]]，把训练闭环、rollout、权重同步和扩展点统一复盘。
