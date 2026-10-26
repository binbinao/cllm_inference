# CPU ≤30B 运行时设计

日期：2026-10-26  
项目：cllm_inference  
前提：在 CPU + 大内存上推理 ≤30B 的 GGUF 模型（主战场为 Linux x86_64，单机单用户对话）。

## 1. 目标

让本引擎在 30B 量级上「能聊」：多轮不必从头扫权重、x86 走整数点积、服务端断连立即停算、上下文溢出硬失败。

成功标准：

- 同一会话连续两轮请求若 prompt tokens 有公共前缀，只对新增后缀做 prefill。
- `seq_len` 达到 `n_ctx` 时抛出明确错误，KV 缓冲区在构造时按 `n_ctx` 预留，避免 GB 级 `vector` 扩容拷贝。
- Q4_K / Q5_K / Q6_K / Q8_0 在 `__AVX2__` 下走 SIMD `vec_dot`，数值与现有标量/反量化参考路径在既有容差内一致。
- Release 构建使用 `-O3 -march=native`。
- SSE 客户端断开后，当前生成循环立即结束并释放引擎锁；连接线程不无限堆积。
- Chat 请求按角色边界套原生模板；`<|im_end|>` / `<|eot_id|>` 等与 `eos_id` 一样能停；流式输出不出现半截 UTF-8。
- `/v1/chat/completions` 与 `/v1/completions` 返回 `usage`；CLI 打印 prefill/decode tok/s。

## 2. 非目标（本轮明确不做）

- Sampler `partial_sort` / `repeat_penalty` / `SampleParams.seed`
- Q4_1 / Q5_1 / Q8_1 / Q2_K / Q3_K 的 `vec_dot`，以及 FP16 LUT
- SentencePiece `token_scores` 重写
- Attention 按 head 并行、scratch 零分配、RoPE 全表预计算
- 多会话 map / 模型热切换 / 滑动窗口驱逐

## 3. 架构

```
HTTP/CLI
   │  ChatMessage[] + Utf8StreamDecoder
   ▼
Tokenizer.apply_chat_template(messages) / is_eos
   │
   ▼
TransformerEngine.prefill(tokens)     ← LCP + truncate + batched suffix
   │  cached_tokens_  ==  KV 长度
   ▼
decode: forward(token) + Sampler + is_eos + utf8 flush
```

单引擎、单把 `engine_mtx`。Prefix cache 只服务「上一串 tokens」。两人交替会 miss 并重算，可接受。

## 4. 引擎与 KV

### 4.1 状态

`TransformerEngine::Impl` 增加：

- `std::vector<int> cached_tokens_`：与 `seq_len` 等长，表示当前 KV 对应的 token 序列。
- 构造时对每层 `k_cache[l]` / `v_cache[l]` 执行 `reserve(n_ctx * n_head_kv * head_dim)`。

### 4.2 接口

```cpp
std::vector<float> forward(int token, bool compute_logits = true);
std::vector<float> prefill(const std::vector<int>& tokens);
void reset_kv_cache();          // 同时清空 cached_tokens_
int seq_len() const;
```

`forward`：

1. 若 `seq_len >= n_ctx`，抛 `std::runtime_error("context length exceeded")`。
2. 写入 embedding、跑各层；仅当 `compute_logits` 为 true 时计算 `output_norm` + `output.weight`。
3. 追加 KV 与 `cached_tokens_.push_back(token)`，`seq_len++`。
4. `compute_logits == false` 时返回空 vector。

`prefill(tokens)`：

1. `tokens` 为空则抛错。
2. `lcp =` 与 `cached_tokens_` 的最长公共前缀。
3. 若 `lcp == tokens.size()`：已全部在 cache 中，为得到最后位置 logits，将 KV/`cached_tokens_` 截到 `lcp-1`，再 `forward(tokens.back(), true)`。
4. 否则 `truncate_kv(lcp)`，对后缀 `tokens[lcp..]` 做 batched 前向；仅最后一枚 token 计算 logits。
5. 若 `tokens.size() > n_ctx` 抛同一条 context 错误。

`truncate_kv(n)`：每层 `k_cache`/`v_cache` `resize(n * kv_dim)`，`cached_tokens_.resize(n)`，`seq_len = n`。

CLI 与 HTTP 不再调用 `reset_kv_cache()` 后逐 token `forward`；改为 `prefill(ids)` 再循环 `forward(next)`。HTTP 仍持有 `engine_mtx` 覆盖整个请求。

### 4.3 Batched prefill

后缀长度 `T==1` 时走现有单步 `forward(..., compute_logits=true)`。

`T>1` 时权重静止：每个线性层把权重行读一次，与 T 行激活做点积。

新增：

```cpp
void matmul_quant_batch(const float* x, const void* w, GgmlType type,
                        const float* bias, int in_dim, int out_dim, int n_rows,
                        float* y, ThreadPool* pool);
```

- `x` 行主序 `[n_rows, in_dim]`，`y` 行主序 `[n_rows, out_dim]`。
- 先把每一行激活量化为 `QuantizedQ8_0`（有 `vec_dot` 时），再对每个输出行 `j`：读 `W[j]` 一次，对 `t=0..n_rows-1` 做 `vec_dot`。
- 无 `vec_dot` 的类型按行走现有兜底。
- Fp32/Fp16 模式同样按「外层 out_dim、内层 n_rows」访问权重，保持同一 cache 行为。

层内流程（仅新后缀，位置从当前 `seq_len` 起）：

1. 取出 T 行 embedding → `X[T, n_embd]`。
2. 每层：RMSNorm 逐行；Q/K/V 用 `matmul_quant_batch`；Qwen3 Q/K norm、RoPE 按绝对位置 `seq_len+t`；把 T 步 K/V 追加进 cache；对每个 t 做因果 attention（可见 `seq_len+t+1` 的 KV）；O 投影 batch；残差；FFN gate/up/down batch。
3. 只对最后一行做 output 投影，得到 logits。
4. 更新 `cached_tokens_` 与 `seq_len`。

Attention 本轮仍按 token 串行（非目标：head 并行）。收益来自 FFN/投影的权重复用，这是 30B 带宽账本的大头。

## 5. x86 AVX2 内核

仅扩展已有 `has_vec_dot` 类型中的主力四类：`Q8_0`、`Q4_K`、`Q5_K`、`Q6_K`。`Q4_0`/`Q5_0` 保持现有标量/NEON，不扩范围。

选择顺序（与现网一致）：

1. `__ARM_NEON && CLLM_HAS_SDOT` → 现有 NEON
2. `__AVX2__` → 新内核（可用 `_mm256_maddubs_epi16` + `_mm256_madd_epi16`）
3. 标量参考

AVX2 结果必须通过现有 `tests/quant_test.cpp` 的 `matmul_quant` 容差，并新增「同一组输入上 `vec_dot_row` 与标量公式一致」的断言（容差：非 K `1e-2/3%`，K `3e-2/6%`，与现网相同）。

构建：

- Makefile：`CXXFLAGS ?= -std=c++20 -O3 -march=native -Wall -Wextra`
- CMake：`Release` 默认 `-O3 -march=native`（可用 `CLLM_NATIVE=OFF` 关闭 `march=native` 以便可移植 CI）

## 6. Tokenizer

- 构造时建 `unordered_map<string,int> token_to_id_`，`find_token` 改为 O(1)。不重写 SentencePiece 切分。
- `struct ChatMessage { std::string role; std::string content; };`
- 保留 `apply_chat_template(user, system)`，内部转成 messages 后走新重载。
- `apply_chat_template(const vector<ChatMessage>&)`：按已识别格式逐条插入角色 token（ChatML / Llama3 / Llama2）；system 取第一条 system；其余按顺序渲染；最后打开 assistant 头、不写 assistant 正文。
- `is_eos(int id)`：`eos_id_` 以及词表中存在的 `<|im_end|>`、`<|eot_id|>`、`<|end_of_text|>`。
- `class Utf8StreamDecoder`：`feed(string_view bytes) -> string` 只吐出完整码点；`flush()` 在结束时倒出残余（非法尾字节原样给出，避免卡死）。

## 7. HTTP 与 CLI

- `send_all` 改 `send(fd, ..., MSG_NOSIGNAL)`；返回 false 则 SSE 循环 `break`。
- 连接处理：`detach` 工作线程，或在 `accept` 循环里 `join` 已结束线程。禁止 `threads` 只增不减。本轮采用：每连接 `std::thread` + `detach`，`Impl` 不再堆积 `vector<thread>`。
- Header 名大小写不敏感匹配 `Content-Length`；`content_length > 4*1024*1024` 回 413。
- 非流式 JSON 增加 `"usage":{"prompt_tokens":P,"completion_tokens":C,"total_tokens":P+C}`。流式结束前可发一行带 usage 的 chunk；若实现成本高，流式只保证正常 delta + `[DONE]`，usage 放在非流式。**本轮约定：非流式必带 usage；流式在 `[DONE]` 前多发一个带 usage 的 data 行（OpenAI 兼容可选，有则更好）。**
- CLI：prefill 与 decode 分别计时，结束后打印 `prefill: N tok, X tok/s` 与 `decode: M tok, Y tok/s`。

HTTP `handle_generate` 使用 `apply_chat_template(messages)`，不再把历史拼成 `user: ...\nassistant: ...`。

停止条件从 `next == eos_id()` 改为 `tokenizer->is_eos(next)`。

## 8. 测试

| 文件 | 覆盖 |
|------|------|
| `tests/tokenizer_test.cpp` | 哈希查找、多轮 ChatML 角色 token 顺序、`is_eos`、UTF-8 跨 token 缓冲 |
| `tests/quant_test.cpp` | 现有 matmul 在 AVX2 下仍过；可加 `vec_dot_row` 抽样 |
| `tests/engine_kv_test.cpp` | 用 `models/Qwen3-0.6B-Q8_0.gguf`（若存在）：`prefill` LCP 复用后 `seq_len` 正确、超 `n_ctx` 抛错、`compute_logits=false` 返回空且 `seq_len` 增加。模型缺失时 skip 并打印原因，退出码 0 |
| `tests/run_tests.sh` / CMake | 接入上述测试 |

不为本轮扩 `model_bench` 矩阵。

## 9. 错误处理

- 上下文溢出：异常，HTTP 层捕获后 400 + JSON `{"error":"..."}`。
- 空 prompt tokens：400 / CLI 报错。
- 客户端断开：静默结束生成，不写半截 JSON 伪成功。

## 10. 文件边界

- `include/cllm/core/engine.hpp` / `src/core/engine.cpp` — prefill、logits 开关、KV 纪律
- `include/cllm/core/quant.hpp` / `src/core/quant.cpp` — `matmul_quant_batch`、AVX2
- `include/cllm/core/tokenizer.hpp` / `src/core/tokenizer.cpp` — 模板、EOS、UTF-8、哈希
- `src/server/http_server.cpp` / `main.cpp` — 服务与 CLI
- `Makefile` / `CMakeLists.txt` — 优化级别与测试目标
