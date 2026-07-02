---
type: batch-doc
module: 28-Frontend-lang
batch: "28"
doc_type: walkthrough
title: "Frontend Language · 源码走读"
tags:
 - sglang/batch/28
 - sglang/module/frontend-lang
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# Frontend Language · 源码走读

> 走读顺序：`api.py` → `ir.py` → `interpreter.run_program` → `StreamExecutor` → `RuntimeEndpoint.generate` → `tracer.py` → `run_program_batch`

---

## 1. 公开 API

### 1.1 `gen()` — 构造 SglGen IR

**Explain：** `gen()` 是用户最常用的生成原语。若传 `choices` 则返回 `SglSelect`；否则校验 regex 后返回 `SglGen`。

**Code：**

```python
# 来源：python/sglang/lang/api.py L102-L139
    if choices:
        return SglSelect(
            name,
            choices,
            0.0 if temperature is None else temperature,
            token_length_normalized if choices_method is None else choices_method,
        )

    # check regex is valid
    if regex is not None:
        try:
            re.compile(regex)
        except re.error as e:
            raise e

    return SglGen(
        name,
        max_tokens,
        min_tokens,
        n,
        stop,
        stop_token_ids,
        stop_regex,
        temperature,
        top_p,
        top_k,
        min_p,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        return_text_in_logprobs,
        dtype,
        regex,
        json_schema,
    )
```

**Comment：**

- 参数默认值在 `SglGen`/`SglSamplingParams` 层合并。
- `gen_int`/`gen_string` 等是 `dtype` + regex 的语法糖。

### 1.2 `SglFunction.run()` — 程序入口

**Explain：** 构造 `SglSamplingParams`，调用 `run_program` 启动解释器。

**Code：**

```python
# 来源：python/sglang/lang/ir.py L212-L221
        backend = backend or global_config.default_backend
        return run_program(
            self,
            backend,
            args,
            kwargs,
            default_sampling_para,
            stream,
            use_thread=use_thread,
        )
```

**Comment：**

- `bind(**kwargs)` 预绑定部分参数，合并进 `func_kwargs`。
- `__call__` 在 TracingScope 内自动走 `trace()` 而非 `run()`。

---

## 2. 解释器启动

### 2.1 `run_program`

**Explain：** 解包 `Runtime.endpoint`（若传入的是 Runtime 包装类），创建 `StreamExecutor` 与 `ProgramState`，同步或异步执行用户函数。

**Code：**

```python
# 来源：python/sglang/lang/interpreter.py L57-L90
def run_program(
    program,
    backend,
    func_args,
    func_kwargs,
    default_sampling_para,
    stream,
    sync=False,
    use_thread=True,
):
    if hasattr(backend, "endpoint"):
        backend = backend.endpoint
    assert backend is not None, "Please specify a backend"
    func_kwargs.update(program.bind_arguments)
    stream_executor = StreamExecutor(
        backend,
        func_kwargs,
        default_sampling_para,
        chat_template=None,
        stream=stream,
        num_api_spec_tokens=program.num_api_spec_tokens,
        use_thread=use_thread,
    )
    state = ProgramState(stream_executor)

    if stream:
        t = threading.Thread(
            target=run_internal, args=(state, program, func_args, func_kwargs, sync)
        )
        t.start()
        return state
    else:
        run_internal(state, program, func_args, func_kwargs, sync)
        return state
```

**Comment：**

- `run_internal` 调用 `program.func(state, *args, **kwargs)`，用户代码中的 `s += ...` 在此执行。
- `stream=True` 时用户函数在独立线程跑，主线程可 `text_iter()` 消费流。

### 2.2 `StreamExecutor.__init__` — 线程模型

**Code：**

```python
# 来源：python/sglang/lang/interpreter.py L321-L332
        # Worker thread
        self.use_thread = use_thread
        if self.use_thread:
            self.queue = queue.Queue()

            def _run_worker_in_context():
                self._thread_worker_func()

            self.worker = threading.Thread(
                target=contextvars.copy_context().run, args=(_run_worker_in_context,)
            )
            self.worker.start()
```

**Comment：**

- 默认 `use_thread=True`：`submit` 入队，后台 `_execute`。
- `use_thread=False` 用于 trace/debug，同步 `_execute`。

---

## 3. 表达式执行

### 3.1 `submit` 与 `_execute` 分派

**Explain：** 所有 IR 类型在 `_execute` 中分派到专用 handler。

**Code：**

```python
# 来源：python/sglang/lang/interpreter.py L342-L348
    def submit(self, expr: SglExpr):
        self._init_var_event(expr)

        if self.use_thread:
            self.queue.put(expr)
        else:
            self._execute(expr)
```

```python
# 来源：python/sglang/lang/interpreter.py L461-L503
    def _execute(self, other):
        if isinstance(other, str):
            other = SglConstantText(other)

        assert isinstance(other, SglExpr), f"{other}"

        if isinstance(other, SglConstantText):
            self._execute_fill(other.value)
        elif isinstance(other, SglGen):
            self._execute_gen(other)
        elif isinstance(other, SglSelect):
            self._execute_select(other)
        elif isinstance(other, SglExprList):
            for x in other.expr_list:
                self._execute(x)
        elif isinstance(other, SglRoleBegin):
            self._execute_role_begin(other)
        elif isinstance(other, SglRoleEnd):
            self._execute_role_end(other)
        elif isinstance(other, SglImage):
            self._execute_image(other)
        elif isinstance(other, SglVideo):
            self._execute_video(other)
        elif isinstance(other, SglVariable):
            self._execute_variable(other)
        elif isinstance(other, SglVarScopeBegin):
            self._execute_var_scope_begin(other)
        elif isinstance(other, SglVarScopeEnd):
            self._execute_var_scope_end(other)
        elif isinstance(other, SglCommitLazy):
            self._execute_commit_lazy_operations(other)
        elif isinstance(other, SglConcateAndAppend):
            if (
                global_config.enable_parallel_encoding
                and self.backend.support_concate_and_append
            ):
                self._execute_concatenate_and_append_kv_cache(other)
            else:
                self._execute_concatenate_and_append_text(other)
        elif isinstance(other, SglSeparateReasoning):
            self._execute_separate_reasoning(other)
        else:
            raise ValueError(f"Unknown type: {type(other)}")
```

**Comment：**

- `SglExprList` 顺序执行，保证 prompt 左到右语义。
- `SglCommitLazy` 触发 `commit_lazy_operations` 把 lazy API spec 落盘。

### 3.2 `_execute_gen` — 调用 backend

**Explain：** 非流式路径调用 `backend.generate`；流式路径迭代 `generate_stream`。结果写入 `text_`、`variables[name]`、`meta_info`。

**Code：**

```python
# 来源：python/sglang/lang/interpreter.py L593-L625
    def _execute_gen(self, expr: SglGen):
        sampling_params = self._resolve_sampling_params(expr.sampling_params)
        name = expr.name
        if not self.stream:
            if self.num_api_spec_tokens is None:
                comp, meta_info = self.backend.generate(
                    self,
                    sampling_params=sampling_params,
                )

            else:
                if self.backend.is_chat_model:
                    # Speculative execution on models with only chat interface.
                    # Store the calls into a temporary list.
                    # They will be lazily executed later.
                    comp, meta_info = self.backend.generate(
                        self,
                        sampling_params=sampling_params,
                        spec_var_name=name,
                    )
                    return

                else:  # Speculative execution on models with completion interface
                    comp, meta_info = self._spec_gen(sampling_params)
            if isinstance(comp, list):
                self.text_ += comp[0]
            else:
                assert isinstance(comp, str)
                self.text_ += comp

            self.variables[name] = comp
            self.meta_info[name] = meta_info
            self.variable_event[name].set()
```

**Comment：**

- `variable_event` 供 `s["var_name"]` 阻塞等待生成完成。
- API speculative：`num_api_spec_tokens` 延迟多次 gen 合并为一次 HTTP。

---

## 4. ProgramState 语法糖

### 4.1 Role 与 `__iadd__`

**Explain：** `s.system("...")` / `with s.user():` 提交 `SglRoleBegin/End`，由 chat template 格式化为 messages。

**Code：**

```python
# 来源：python/sglang/lang/interpreter.py L858-L871
    def _role_common(self, name: str, expr: Optional[SglExpr] = None):
        if expr is not None:
            role_expr = SglExprList([SglRoleBegin(name), expr, SglRoleEnd(name)])
            self.stream_executor.submit(role_expr)
            return role_expr
        else:

            @contextmanager
            def role_scope():
                self.stream_executor.submit(SglRoleBegin(name))
                yield
                self.stream_executor.submit(SglRoleEnd(name))

            return role_scope()
```

**Comment：**

- `_execute_role_begin/end` 更新 `messages_` 与 `cur_role`。
- RuntimeEndpoint 在 generate 时将 `messages_` 或 `text_` 发给 srt。

### 4.2 `fork()` — 并行分支

**Code：**

```python
# 来源：python/sglang/lang/interpreter.py L370-L402
    def fork(
        self,
        size: int = 1,
        position_ids_offset: Optional[List[int]] = None,
    ):
        if size > 1 and str(self.text_):
            self.submit(SglCommitLazy())

        self.sync()
        size = int(size)

        exes = [
            StreamExecutor(
                self.backend,
                self.arguments,
                self.default_sampling_para,
                self.chat_template,
                self.stream,
            )
            for _ in range(size)
        ]
        for i in range(size):
            exes[i].variables = dict(self.variables)
            exes[i].text_ = str(self.text_)
            exes[i].messages_ = list(self.messages_)
            exes[i].cur_role = self.cur_role
            exes[i].cur_role_begin_pos = self.cur_role_begin_pos
            exes[i].fork_start_text_pos = len(self.text_)
            exes[i].images_ = list(self.images_)

            # TODO(ying): handle API speculative execution

        return exes
```

**Comment：**

- fork 前 `SglCommitLazy` 确保 KV 状态一致。
- `ProgramStateGroup.join()` 合并分支结果。

---

## 5. RuntimeEndpoint Backend

### 5.1 初始化 — 拉取 model_info

**Code：**

```python
# 来源：python/sglang/lang/backend/runtime_endpoint.py L27-L54
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        verify: Optional[str] = None,
        chat_template_name: Optional[str] = None,
    ):
        super().__init__()
        self.support_concate_and_append = True

        self.base_url = base_url
        self.api_key = api_key
        self.verify = verify

        res = http_request(
            self.base_url + "/get_model_info",
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        self.model_info = res.json()

        if chat_template_name:
            self.chat_template = get_chat_template(chat_template_name)
        else:
            self.chat_template = get_chat_template_by_model_path(
                self.model_info["model_path"]
            )
```

**Comment：**

- 启动时探测 server，自动匹配 chat template。
- `support_concate_and_append=True` 启用 KV 拼接 fast path。

### 5.2 `generate` — POST /generate

**Explain：** 将 `StreamExecutor.text_` 与 sampling params 打包 JSON，POST 到 srt `/generate`。

**Code：**

```python
# 来源：python/sglang/lang/backend/runtime_endpoint.py L159-L196
    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        self._handle_dtype_to_regex(sampling_params)
        data = {
            "text": s.text_,
            "sampling_params": {
                "skip_special_tokens": global_config.skip_special_tokens_in_output,
                "spaces_between_special_tokens": global_config.spaces_between_special_tokens_in_out,
                **sampling_params.to_srt_kwargs(),
            },
        }

        for item in [
            "return_logprob",
            "logprob_start_len",
            "top_logprobs_num",
            "return_text_in_logprobs",
        ]:
            value = getattr(sampling_params, item, None)
            if value is not None:
                data[item] = value

        self._add_images(s, data)

        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

        obj = res.json()
        comp = obj["text"]
        return comp, obj["meta_info"]
```

**Comment：**

- 多模态：`_add_images` 把 base64 图片附到 JSON。
- `max_new_tokens=0` 的 `commit_lazy_operations` 也走同一 endpoint 做 prefill-only。

### 5.3 `cache_prefix` — Radix 预热

**Code：**

```python
# 来源：python/sglang/lang/backend/runtime_endpoint.py L80-L87
    def cache_prefix(self, prefix_str: str):
        res = http_request(
            self.base_url + "/generate",
            json={"text": prefix_str, "sampling_params": {"max_new_tokens": 0}},
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
```

**Comment：**

- `max_new_tokens: 0` 仅做 prefill，填充 RadixAttention cache。
- batch trace 后对共享 prefix 调用一次即可。

---

## 6. Batch 执行

### 6.1 `run_program_batch`

**Explain：** 多样本并行跑同一 `SglFunction`，可选 trace prefix cache，线程池并发。

**Code：**

```python
# 来源：python/sglang/lang/interpreter.py L105-L114
    # Pre-cache the common prefix for a batch. The prefix is extracted by tracing the program.
    if global_config.enable_precache_with_tracing and len(batch_arguments) > 1:
        cache_program(program, backend)

    # Run all programs
    if num_threads == "auto":
        num_threads = max(96, multiprocessing.cpu_count() * 16)
    num_threads = min(num_threads, len(batch_arguments))

    if generator_style:
```

**Comment：**

- `num_threads="auto"` 默认极高并发（IO bound HTTP）。
- `generator_style=True` 用 yield 逐个返回完成样本。

---

## 7. Tracer

### 7.1 `SglFunction.__call__` tracing 分支

**Code：**

```python
# 来源：python/sglang/lang/ir.py L316-L324
    def __call__(self, *args, **kwargs):
        from sglang.lang.tracer import TracingScope

        tracing_scope = TracingScope.get_current_scope()
        if tracing_scope is None:
            return self.run(*args, **kwargs)
        else:
            kwargs["backend"] = tracing_scope.tracer_state.backend
            return self.trace(*args, **kwargs)
```

**Comment：**

- Tracer 用 dummy `SglArgument` 占位运行时参数，避免真实 API 调用。
- `StopTracing` 异常用于提前终止 trace。

---

## 8. dtype → regex 转换

**Code：**

```python
# 来源：python/sglang/lang/backend/runtime_endpoint.py L127-L157
    def _handle_dtype_to_regex(self, sampling_params: SglSamplingParams):
        if sampling_params.dtype is None:
            return

        if sampling_params.stop == ():
            sampling_params.stop = []

        dtype_regex = None
        if sampling_params.dtype in ["int", int]:

            dtype_regex = REGEX_INT
            sampling_params.stop.extend([" ", "\n"])
        elif sampling_params.dtype in ["float", float]:

            dtype_regex = REGEX_FLOAT
            sampling_params.stop.extend([" ", "\n"])
        elif sampling_params.dtype in ["str", str]:

            dtype_regex = REGEX_STR
        elif sampling_params.dtype in ["bool", bool]:

            dtype_regex = REGEX_BOOL
        else:
            raise RuntimeError(f"Invalid dtype: {sampling_params.dtype}")

        if dtype_regex is not None and sampling_params.regex is not None:
            warnings.warn(
                f"Both dtype and regex are set. Only dtype will be used. dtype: {sampling_params.dtype}, regex: {sampling_params.regex}"
            )

        sampling_params.regex = dtype_regex
```

**Comment：**

- `gen_int()` 等依赖此路径把结构化类型转为 srt regex 约束。
- dtype 与 regex 同时设置时 dtype 优先（warning）。

---

## 9. Lazy Commit

**Code：**

```python
# 来源：python/sglang/lang/backend/runtime_endpoint.py L105-L114
    def commit_lazy_operations(self, s: StreamExecutor):
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
```

**Comment：**

- fork 前、concate 前常插入 `SglCommitLazy()` 同步 KV 状态。
- 与 `cache_prefix` 类似，max_new_tokens=0。

---

## 10. Select（多选一类）

**Code：**

```python
# 来源：python/sglang/lang/interpreter.py L647-L655
    def _execute_select(self, expr: SglSelect):
        choices_decision = self.backend.select(
            self, expr.choices, expr.temperature, expr.choices_method
        )
        if expr.name is not None:
            name = expr.name
            self.variables[name] = choices_decision.decision
            self.meta_info[name] = choices_decision.meta_info
            self.variable_event[name].set()
```

**Comment：**

- RuntimeEndpoint 通过 logprob 比较 choices（见 `choices.py` 采样方法）。
- 等价于 constrained classification 而非自由生成。
