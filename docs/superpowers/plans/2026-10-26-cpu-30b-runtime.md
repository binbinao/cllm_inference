# CPU ≤30B Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 cllm_inference 在 CPU + 大内存上可运行 ≤30B 多轮对话：前缀 KV 复用、batched prefill、x86 AVX2 点积、断连停算、原生多轮模板。

**Architecture:** 引擎维护 `cached_tokens_` 与按 `n_ctx` 预留的 KV；`prefill` 做 LCP 截断后对后缀做权重静止的 batch 矩阵乘；量化快路径在 AVX2 上为 Q8_0/Q4_K/Q5_K/Q6_K 提供 SIMD；Tokenizer 用哈希词表 + 多轮模板 + 多 EOS + UTF-8 流式缓冲；HTTP 用 `MSG_NOSIGNAL` 并 detach 连接线程。

**Tech Stack:** C++20，零第三方依赖，POSIX sockets，Makefile/CMake，现有手写断言测试。

## Global Constraints

- 零第三方运行时依赖，仅标准库 + POSIX。
- 本轮不做 Sampler 加速、冷门量化 vec_dot、FP16 LUT、SP 重写、Attention head 并行、scratch/RoPE 全表、多会话。
- AVX2 仅覆盖 Q8_0 / Q4_K / Q5_K / Q6_K；选择顺序 NEON SDOT → AVX2 → 标量。
- `n_ctx` 溢出抛 `std::runtime_error("context length exceeded")`。
- Release 编译：`-O3 -march=native`（CMake 可用 `CLLM_NATIVE=OFF`）。
- 每个行为先写失败测试再写实现（TDD）。
- 单引擎 Prefix Cache，无 session map。

---

### Task 1: Tokenizer 哈希、多轮模板、多 EOS、UTF-8 流式

**Files:**
- Create: `tests/tokenizer_test.cpp`
- Modify: `include/cllm/core/tokenizer.hpp`
- Modify: `src/core/tokenizer.cpp`
- Modify: `tests/run_tests.sh`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `ChatMessage`, `Tokenizer::apply_chat_template(const std::vector<ChatMessage>&)`, `is_eos(int)`, `Utf8StreamDecoder::{feed,flush}`, O(1) `find_token`

- [ ] **Step 1: Write the failing test** `tests/tokenizer_test.cpp`

构造最小假 `GgufModel`（tokens 含 ChatML 特殊符与中文 BPE 字节），断言：`find_token` 命中/未命中、两轮 ChatML 含两个 user 头、`is_eos` 识别 `<|im_end|>`、`Utf8StreamDecoder` 先 feed 半个「你」再补齐才输出。

- [ ] **Step 2: Run test to verify it fails**（头文件尚无新 API）

```bash
clang++ -std=c++20 -O2 -Iinclude tests/tokenizer_test.cpp src/core/tokenizer.cpp src/core/gguf.cpp src/core/quant.cpp -o /tmp/tokenizer_test -pthread
```

Expected: compile fail on `ChatMessage` / `is_eos` / `Utf8StreamDecoder`.

- [ ] **Step 3: Minimal implementation**

`tokenizer.hpp` 增加：

```cpp
struct ChatMessage {
    std::string role;
    std::string content;
};

class Utf8StreamDecoder {
public:
    std::string feed(std::string_view bytes);
    std::string flush();
private:
    std::string buf_;
};

class Tokenizer {
    std::vector<int> apply_chat_template(const std::vector<ChatMessage>& messages) const;
    bool is_eos(int token_id) const;
    // token_to_id_, stop_ids_
};
```

实现：构造填 `token_to_id_` 与 `stop_ids_`；`find_token` 查 map；多轮 ChatML/Llama3/Llama2 按 message 循环；单轮重载转 messages；`Utf8StreamDecoder` 按 UTF-8 首字节长度缓冲。

- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit** `test+feat(tokenizer): hash lookup, multi-turn template, multi-EOS, utf8 stream`

---

### Task 2: AVX2 vec_dot + 构建开关 + matmul_quant_batch

**Files:**
- Modify: `include/cllm/core/quant.hpp`
- Modify: `src/core/quant.cpp`
- Modify: `tests/quant_test.cpp`
- Modify: `Makefile`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `matmul_quant_batch(...)`；AVX2 内核仅由 `vec_dot_*` 内部选择

- [ ] **Step 1: Write failing tests**

在 `quant_test.cpp` 增加 `test_matmul_quant_batch`：`n_rows=3`，逐行 `matmul_quant` 作参考，与 batch 结果逐元素比较（容差同现网）。

- [ ] **Step 2: Run to verify fail**（符号不存在）
- [ ] **Step 3: Implement**

`quant.hpp` 声明 `matmul_quant_batch`。  
`quant.cpp`：权重外环、激活行内环；`#if defined(__AVX2__)` 为 Q8_0/Q4_K/Q5_K/Q6_K 写 SIMD，公式与标量版相同。  
Makefile `CXXFLAGS` 改为 `-O3 -march=native`。CMake Release 加 `-O3`，`option(CLLM_NATIVE "Use -march=native" ON)`。

- [ ] **Step 4: `make test` / 直接跑 quant_test，全绿**
- [ ] **Step 5: Commit** `perf(quant): AVX2 vec_dot for Q8_0/Q4_K/Q5_K/Q6_K and batched matmul`

---

### Task 3: Engine Prefix Cache、n_ctx、batched prefill、skip logits

**Files:**
- Modify: `include/cllm/core/engine.hpp`
- Modify: `src/core/engine.cpp`
- Create: `tests/engine_kv_test.cpp`

**Interfaces:**
- Consumes: `matmul_quant_batch`
- Produces: `forward(int, bool compute_logits=true)`, `prefill(const vector<int>&)`, `seq_len()`

- [ ] **Step 1: Write failing `tests/engine_kv_test.cpp`**

若 `models/Qwen3-0.6B-Q8_0.gguf` 不存在则打印 skip 并 return 0。否则：

1. `prefill({t,t,t})` 后 `seq_len()==3`。
2. 再 `prefill({t,t,t,t})` 后 `seq_len()==4`（前缀 3 复用）。
3. `forward(t, false)` 返回 empty 且 `seq_len` +1。
4. 将引擎用很小 `n_ctx` 不可行（config 来自文件）；改为循环 `forward` 直到 `n_ctx` 抛 `"context length exceeded"` 不现实（n_ctx 很大）。改为：单测 `prefill` 传入长度为 `n_ctx+1` 的向量应抛错。可用短循环构造 `vector<int>(cfg.n_ctx+1, bos)`——若 n_ctx 为 32768，分配可以但 prefill 会极慢。

**替代：** 不跑满 n_ctx。在 `engine_kv_test` 只测 LCP 与 `compute_logits`。n_ctx 用一个可注入上限不在本轮加测试钩子；在 `forward`/`prefill` 里检查，另写纯函数测试不可行。**决定：** 若 `n_ctx > 4096`，构造长度为 1 的 prefill 后，不测溢出；另在测试中若我们无法廉价触发，则增加 `TransformerEngine` 不暴露 setter。改为在测试文件里只断言：`prefill` 空 vector 抛错。溢出路径用代码审查 + 在 `forward` 入口写死检查。额外：写 `tests/engine_kv_test.cpp` 测空 prefill 抛错、LCP、skip logits。

为覆盖 n_ctx：允许测试在 **没有** 改 config 的情况下跳过超长用例。实现必须包含检查。

- [ ] **Step 2: Compile/run，确认新 API 失败**
- [ ] **Step 3: Implement engine**

KV `reserve(n_ctx * kv_dim)`；`cached_tokens_`；`truncate_kv`；`prefill` LCP 逻辑；`T>1` 走 batch 层循环；`T==1` 走 `forward`。

- [ ] **Step 4: 跑 engine_kv_test + 原有测试**
- [ ] **Step 5: Commit** `feat(engine): prefix cache, n_ctx guard, batched prefill, skip logits`

---

### Task 4: HTTP 断连、线程、多轮、usage；CLI tok/s

**Files:**
- Modify: `src/server/http_server.cpp`
- Modify: `main.cpp`

**Interfaces:**
- Consumes: `prefill`, `is_eos`, `apply_chat_template(messages)`, `Utf8StreamDecoder`

- [ ] **Step 1: 无独立 HTTP 测试框架。将可测逻辑保持在已测模块。实现时：**
  - `send` + `MSG_NOSIGNAL`，失败 break
  - `detach` 连接线程
  - Content-Length 大小写不敏感，>4MB → 413
  - chat 走 messages 模板；`is_eos`；`prefill`
  - 非流式 usage；流式在 `[DONE]` 前发 usage 行
  - CLI 计时打印 tok/s；停止用 `is_eos`

- [ ] **Step 2: 编译 `make` 成功**
- [ ] **Step 3: Commit** `feat(server,cli): abort on disconnect, usage, multi-turn, tok/s`

---

### Task 5: 接入测试脚本并全量回归

**Files:**
- Modify: `tests/run_tests.sh`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: run_tests.sh 增加 tokenizer_test、engine_kv_test**
- [ ] **Step 2: `sh tests/run_tests.sh` 全绿**
- [ ] **Step 3: Commit** `test: wire tokenizer and engine kv into make test`

---

### Task 6: README 同步（仅事实）

**Files:**
- Modify: `README.md` 命令行/API/特性中与本轮行为相关的句子（prefix cache、usage、tok/s、AVX2、`-O3`）。不扩无关章节。

- [ ] **Step 1: 更新 README 对应表格与构建说明**
- [ ] **Step 2: Commit** `docs(readme): document 30B CPU runtime behavior`
