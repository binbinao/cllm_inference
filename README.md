# cllm_inference

自研 **C++20 GGUF 推理引擎**：从零实现，零第三方运行时依赖，支持加载 GGUF 量化模型并在 CPU 上完成推理，提供 OpenAI 兼容的 HTTP 服务。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Architecture diagram](https://gitdiagram.com/diagram-badge.svg)](https://gitdiagram.com/binbinao/cllm_inference?utm_source=readme&utm_medium=badge)

## 特性

- **纯 C++20 / 零依赖**：仅依赖标准库与 POSIX 系统调用，无任何第三方库
- **完整 GGUF 解析**：支持 GGUF v2/v3，mmap 零拷贝加载
- **量化类型全覆盖**：支持 F32 / F16 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q8_1 / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K 共 13 种 GGML 量化格式
- **零拷贝量化推理**：量化权重直接引用 mmap 数据，按输出行惰性反量化，无需加载期全量展开为 float32（0.5B Q4_K_M 常驻内存 ~430MB）
- **整数点积内核**：激活在线量化为 Q8_0，Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K 走 vec_dot；Apple Silicon 用 ARMv8.2 SDOT NEON，x86_64 上 Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K 走 AVX2；相对「反量化 + float matmul」**3-7x**
- **Prefix Cache + batched prefill**：多轮复用公共前缀 KV；新后缀按权重静止批量矩阵乘，仅最后一枚 token 算输出层
- **上下文硬限制**：`--ctx` 可把硬帽压到 `min(模型 n_ctx, N)`；超过明确报错。KV 最多按 4096 token 预留，避免按 32k–40k 空占数 GB
- **mmap 预热**：加载后 `madvise(MADV_WILLNEED)`，减轻 30B 首次 prefill 缺页
- **stop / seed**：CLI `--stop` / `--seed`，HTTP `stop`（字符串或数组）与 `seed`；生成文本后缀匹配即停
- **通用 Transformer 内核**：RMSNorm、NEOX RoPE、GQA Attention、SwiGLU FFN、KV Cache
- **CPU 多线程**：手写线程池，按行/分块并行矩阵乘
- **分词与采样**：BPE + SentencePiece，词表哈希 O(1) 查找；temperature / top-p / top-k
- **多轮对话模板**：ChatML / Llama 3 / Llama 2·Mistral 按 `messages` 角色边界渲染，不再拼接纯文本
- **多停止符 + UTF-8 流式**：`eos` / `<|im_end|>` / `<|eot_id|>` / `<|end_of_text|>` 均可停；流式凑齐完整码点再输出
- **双入口**：CLI（结束打印 prefill/decode tok/s）+ OpenAI 兼容 HTTP（SSE、`usage`、断连即停、包体 4MB 上限）
- **内置 Web UI**：单文件前端，流式对话、模型库、启动命令（Qwen/Llama ≤30B）
- **测试与基准**：量化 / 线程池 / 采样器 / 分词器 / Prefix Cache，含量化吞吐与模型级三模式对比基准

---

## 项目架构

系统由「模型加载 → 推理引擎 → 采样解码」三条主线与「HTTP 服务」旁路组成，全部为自研 C++ 模块，零第三方运行时依赖：

```
                        ┌─────────────────────────────────────────┐
                        │              main.cpp（双入口）           │
                        └───────┬──────────────────────┬──────────┘
                 --prompt      │ 加载模型               │ 无 --prompt 时
                        ┌───────▼─────────┐            │ 启动服务
                        │   GgufLoader    │      ┌─────▼──────────┐
                        │  GGUF 解析·mmap │      │   HttpServer   │
                        └───────┬─────────┘      │ :8080·SSE 流式 │
                                │ 张量/词表        └─────┬──────────┘
                        ┌───────▼────────────────────────▼──────────┐
                        │            TransformerEngine               │
                        │  RMSNorm · RoPE · GQA Attention · SwiGLU   │
                        │   KV Cache · Prefix Cache · batched prefill │
                        └───┬───────────────┬──────────────┬────────┘
                            │ 反量化权重     │ 并行矩阵乘    │ logits
                     ┌──────▼─────┐   ┌─────▼──────┐  ┌────▼─────┐
                     │   Quant    │   │ ThreadPool │  │  Sampler │──┐
                     │  反量化     │   │  并行分块   │  │ top-k/p  │  │ next token
                     └────────────┘   └────────────┘  └──────────┘◄─┘
```

| 模块 | 职责 | 关键文件 |
|------|------|---------|
| `GgufLoader` | 解析 GGUF 元数据/词表/张量，mmap 零拷贝加载 | `src/core/gguf.cpp` |
| `Quant` | 反量化 + vec_dot（NEON / AVX2 / 标量）+ `matmul_quant_batch` | `src/core/quant.cpp` |
| `TransformerEngine` | `prefill`（LCP 复用 + 批量后缀）/ `forward`；RMSNorm · RoPE · GQA · SwiGLU · KV | `src/core/engine.cpp` |
| `Tokenizer` | BPE / SentencePiece、哈希词表、多轮模板、多 EOS、`Utf8StreamDecoder` | `src/core/tokenizer.cpp` |
| `Sampler` | temperature / top-k / top-p 采样 | `src/core/sampler.cpp` |
| `ThreadPool` | 手写线程池，按行/分块并行矩阵乘 | `src/core/thread_pool.cpp` |
| `HttpServer` | OpenAI 兼容 HTTP、SSE、`usage`、断连即停 | `src/server/http_server.cpp` |

> 推理流程：`prefill(ids)` 对齐已缓存前缀并批量计算新后缀（只对最后一枚 token 求 logits），再循环「采样 → UTF-8 增量解码 → `forward` 下一枚」直至 `is_eos` 或 `max_tokens`。CLI 与 HTTP 复用同一套引擎；HTTP 单引擎串行，Prefix Cache 服务「上一串 tokens」。

---

## 依赖

| 依赖 | 要求 | 说明 |
|------|------|------|
| 编译器 | C++20 支持（clang++ 或 g++） | macOS 自带 clang++，Linux 用 g++/clang++ |
| 构建工具 | `make` 或 `cmake`（二选一） | 推荐 `make`，无需 cmake。Release 默认 `-O3 -march=native`（CMake 可用 `-DCLLM_NATIVE=OFF` 关闭） |

> macOS 无需额外安装；Linux 确保已安装 `build-essential`（g++/make）。

---

## 安装 Playbook

### 方式一：Makefile（推荐，无 cmake 依赖）

```bash
# 1. 编译 release 版可执行文件（产物为 ./cllm_inference）
make

# 2. 安装到系统（默认 /usr/local/bin）
sudo make install

# 可选：自定义安装前缀
make install PREFIX=$HOME/.local

# 可选：打包到临时目录（不污染系统，便于制作安装包）
make install DESTDIR=/tmp/pkg PREFIX=/usr/local

# 清理编译产物
make clean
```

### 方式二：CMake

```bash
# 1. 配置 + 编译
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2. 安装（默认 /usr/local）
sudo cmake --install build

# 可选：自定义前缀
cmake --install build --prefix $HOME/.local

# 可选：关闭 -march=native（便于可移植 CI）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCLLM_NATIVE=OFF
```

安装后，可执行文件位于 `<prefix>/bin/cllm_inference`，头文件位于 `<prefix>/include/cllm/`。

---

## 使用 Playbook

### 1. 准备模型

下载任意 GGUF 格式模型（推荐 Qwen2.5 小尺寸模型快速验证）：

```bash
# 示例：Qwen2.5-0.5B-Instruct（约 470MB，Q4_K_M 量化）
mkdir -p models
# 从 HuggingFace 等渠道下载 gguf 文件到 models/ 目录
```

> 已支持架构：`qwen2`、`qwen3`（含 head_dim 独立、Q/K 层级 RMSNorm）、`llama`。
> 模型量化类型需覆盖 13 种 GGML 格式（F32 / F16 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q8_1 / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K）。

### 2. 本地 CLI 推理

```bash
# 基础问答（greedy 采样）
cllm_inference --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "你好" --temperature 0 --max-tokens 40

# 创意生成（随机采样）
cllm_inference --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "写一首关于春天的短诗" --temperature 0.8 --top-p 0.9 --max-tokens 60

# 指定线程数（结束后打印 prefill / decode tok/s）
cllm_inference --model models/xxx.gguf --prompt "你好" --threads 8

# 限制上下文、固定种子、自定义停止后缀
cllm_inference --model models/xxx.gguf --prompt "你好" --ctx 4096 --seed 1 --stop $'\n\n'
```

### 3. HTTP 服务（OpenAI 兼容）

```bash
# 启动服务（默认端口 8080）
cllm_inference --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080 --threads 8
```

**健康检查**

```bash
curl http://127.0.0.1:8080/health
# => {"status":"ok"}
```

**文本补全（非流式）**

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"你好","temperature":0,"max_tokens":30}'
```

**对话补全（原生多轮模板，响应含 `usage`）**

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[
        {"role":"system","content":"用中文简短回答"},
        {"role":"user","content":"1+1等于几？"},
        {"role":"assistant","content":"2"},
        {"role":"user","content":"再加 1 呢？"}
      ],"temperature":0,"max_tokens":40}'
# 同一进程连续请求会复用公共前缀的 KV Cache
```

**流式响应（SSE）**

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"1+1等于几？","stream":true,"max_tokens":20}'
```

---

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model <path>` | GGUF 模型文件路径（必填） | — |
| `--prompt <text>` | 本地生成 prompt（提供则走 CLI，否则启动 HTTP） | 空 |
| `--port <n>` | HTTP 服务端口 | 8080 |
| `--threads <n>` | CPU 线程数 | 硬件并发数 |
| `--ctx <n>` | 上下文硬帽（同时限制 KV） | 模型 `n_ctx` |
| `--temperature <f>` | 采样温度（0 = greedy） | 0.8 |
| `--top-p <f>` | 核采样（nucleus） | 0.9 |
| `--top-k <n>` | Top-k 采样 | 40 |
| `--seed <n>` | 采样 RNG 种子 | 随机 |
| `--stop <str>` | 生成文本后缀停止（可重复） | 无 |
| `--max-tokens <n>` | 最大生成 token 数 | 64 |

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/v1/model/info` | 当前加载模型的元信息（架构、层数、上下文、模板格式） |
| GET | `/v1/models` | OpenAI 兼容的模型列表（当前模型包装为 data 数组） |
| POST | `/v1/completions` | 文本补全（`prompt` + 采样，支持 `stream`；返回 `usage`） |
| POST | `/v1/chat/completions` | 对话补全（原生多轮 `messages`；`stop` / `seed`；返回 `usage`；流式在 `[DONE]` 前带 usage） |
| OPTIONS | `/*` | CORS 预检，允许所有源 |
| GET | `/` 、`/web/*`、`/models.json` | 托管运行目录下 `web/` 前端静态资源（若存在） |

---

## 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `missing tensor: blk.N.xxx` | 模型张量命名与当前支持架构不匹配 | 换用 qwen2/llama 架构的 GGUF |
| `unsupported ggml type` | 模型含未实现的量化类型 | 已支持 13 种常用类型；如遇新类型需扩展 `quant.cpp` |
| `in_dim not divisible by block size` | 张量内维与量化 block 不整除 | 该类模型罕见，通常是转换异常 |
| 输出乱码 | 未走对话模板，或旧版按字节刷半截 UTF-8 | 确认 `tokenizer.chat_template`；当前流式会拼完整码点再输出 |
| `context length exceeded` | prompt 或生成超过硬帽（`--ctx` 或模型 `n_ctx`） | 缩短历史 / 增大 `--ctx` |
| 413 Payload Too Large | 请求体超过 4MB | 减小 `messages` / `prompt` |
| 内存占用 | 量化权重按需反量化；KV 最多按 4096 token 预留 | 0.5B Q4_K_M 权重约 430MB；用 `--ctx` 限制硬帽 |

> 吞吐说明：Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K 走 **激活量化 + 整数点积**。Apple Silicon 用 NEON SDOT；x86_64 上上述六类均有 AVX2。
> 大核数 / 多路机器建议进程外绑核：`taskset -c 0-15 ./cllm_inference ...` 或 `numactl --interleave=all`。AVX-VNNI 尚未单独实现，有对应 CPU 时下一刀再加。
> 相对「按行反量化 + float matmul」约 3-7x（见下文）。Q4_1/Q5_1/Q8_1/Q2_K/Q3_K 仍走 fallback。

---

## 测试与基准

```bash
# 运行全部单元测试（量化 / 线程池 / 采样器 / 分词器 / Prefix Cache）
make test

# 追加运行量化矩阵乘吞吐基准
make bench
```

- **覆盖范围**：13 种 GGML 反量化、`matmul_quant` / `matmul_quant_batch`、线程池、采样器、
  多轮模板 / 多 EOS / UTF-8 流式、Prefix Cache / `n_ctx` / skip-logits（模型文件缺失时 engine 用例 skip）。
- **基准**：`tests/bench.cpp` 报告各量化类型的 GFLOP/s 及相对纯 float matmul 的比值；
  `tests/model_bench.cpp` 对真实模型做「量化 vs 反量化 fp32/fp16」的 prefill/decode 对比（见下文）。
- CMake 构建下：`cmake --build build && ctest --test-dir build`。

### 性能基准（Apple M4，`in=1024 out=1024`）

| 类型 | 旧版（反量化 + float） | **新版（整数点积 + NEON）** | 内核加速 |
|------|----------------------|------------------------------|----------|
| Q8_0 | 5.40 GFLOP/s (0.62x) | **~38.2 GFLOP/s (~4.2x)** | **~7x** 🔥 |
| Q4_K | 4.74 GFLOP/s (0.54x) | **~41.5 GFLOP/s (~4.6x)** | **~8.8x** 🔥 |
| Q6_K | 4.01 GFLOP/s (0.46x) | **~34.7 GFLOP/s (~3.8x)** | **~8.7x** 🔥 |
| Q5_K | fallback | **~39.4 GFLOP/s (~4.3x)** | **~9x** 🔥 |
| Q5_0 | fallback | **~31.1 GFLOP/s (~3.4x)** | **~7x** 🔥 |
| Q4_0 | fallback | **~32.8 GFLOP/s (~3.6x)** | **~7x** 🔥 |

端到端（Qwen3-0.6B-Q8_0，10 tokens）：**10.3s → 2.5s**，约 **4.1x** 整体加速。

### 量化对齐关系（量化越小 → 越快）

`tests/bench.cpp` 支持 `bench [iters] [out_dim]`，报告每类的量化位数、每元素字节数（B/elem）
与读权重有效带宽（BW）。在统一维度下对齐全部六种已优化类型（M4，`in=1024`）：

| 类型 | bits | B/elem | out=1024 (cache 内) | out=4096 (带宽瓶颈) |
|------|------|--------|---------------------|---------------------|
| Q4_K | 4 | 0.56 | 4.57x | 6.67x |
| Q4_0 | 4 | 0.56 | 3.61x | 5.94x |
| Q5_K | 5 | 0.69 | 4.34x | 6.53x |
| Q5_0 | 5 | 0.69 | 3.42x | 5.13x |
| Q6_K | 6 | 0.82 | 3.82x | 5.94x |
| Q8_0 | 8 | 1.06 | 4.20x | **7.32x** |

**结论**：量化位数越少 → 权重字节越少 → 带宽需求越低（B/elem 严格单调：Q4 < Q5 < Q6 < Q8）。
六种类型均已实现 NEON vec_dot，相对 naive float 提速 3.4~7.3x。
加速比并非严格随位数单调，原因是解包与 scale 结构的差异：
- Q6_K 的 6-bit 跨字节解包开销高于 Q8_0 的零解包，故 Q6_K 略慢于 Q8_0；
- 同一位数下 K 系列（Q4_K/Q5_K）因块级 scale + 预计算 bsums 结构，通常略快于 0 系列（Q4_0/Q5_0）。

> 端到端长案例参考（256 tokens，仅供体感）：Qwen3-0.6B-Q8_0 ≈ 15.7 tok/s。
> 端到端因模型架构/参数量/FFN 宽度不同（如 Qwen2.5-0.5B 的 n_ff=4864 远大于 Qwen3-0.6B 的 3072），
> **不适合**用于量化位数的公平对齐，公平对齐应使用上表统一维度的 micro-benchmark。

### 模型级量化 vs 反量化 fp32/fp16 对比

`tests/model_bench.cpp` 对真实模型分别用三种推理路径（`TransformerEngine` 的 `WeightMode` 枚举控制，
默认 `WeightMode::Quant`）：

- **反量化 fp32**（`WeightMode::Fp32`）：量化权重 `dequantize()` 为 fp32 常驻，forward 走纯 float 矩阵乘
- **反量化 fp16**（`WeightMode::Fp16`）：量化权重 `dequantize_f16()` 为半精度位模式常驻，forward 时逐元素 half→float 累加
- **量化**（`WeightMode::Quant`，默认）：权重零拷贝引用 mmap，forward 走 vec_dot 整数点积快速路径

> 默认开启量化增强：CLI（`main.cpp`）与 HTTP 服务均以 `WeightMode::Quant` 构造 `TransformerEngine`，
> 走整数点积 + 零拷贝 mmap 的快速路径，性能最优。fp32/fp16 反量化路径仅用于公平性能对比与数值验证。

实测（Apple M4，10 线程，输入长度 16/32/64/128/256 token）：

| 模型 | 实际量化（参数量占比） | prefill 加速 | decode 加速 |
|------|----------------------|-------------|-------------|
| qwen2.5-0.5b-instruct-q4_k_m | Q5_0 61.6% + Q8_0 21.8% + Q4_K/Q6_K | 3.3~6.5x | 2.8~5.6x |
| Qwen3-0.6B-Q8_0 | Q8_0 100% | 3.7~5.1x | 3.8~8.1x |
| Llama-3.2-1B-Instruct-Q4_K_M | Q4_K 67.2% + Q6_K 32.8% | 5.7~6.7x | 3.0~13.7x |

**结论**：

- 量化相对 fp32 的 **prefill 加速稳定在 3~7x**，主要来自 FFN/投影矩阵乘的整数点积 + 更低内存带宽。
- **decode 加速比波动更大**（3~13x），因为 decode 的 attention 为 float 计算、不受量化影响，且随序列变长占比上升。
- ⚠️ **fp16 反量化路径通常比 fp32 更慢**：在缺乏硬件 fp16 加速的 x86 平台上，`half_to_float` 为纯软件位运算，
  每个乘加都要先转换，其 ALU 开销超过权重带宽减半的收益（见下方 Linux x86 实测）。
- ⚠️ **文件名可能与实际量化不符**：`qwen2.5-0.5b-instruct-q4_k_m.gguf` 的实际主体是 **Q5_0（61.6%）**，
  并非文件名标注的 Q4_K_M。`model_bench` 会打印每个模型的真实量化类型分布（按参数量加权），便于识别。

补充实测（Linux x86_64，8 核，`-O3 -march=native` + AVX2，Qwen3-0.6B-Q8_0）：

| 输入 | fp32 prefill | fp16 prefill | **quant prefill** | fp32 decode | **quant decode** | quant 加速（prefill / decode） |
|------|--------------|--------------|-------------------|-------------|------------------|--------------------------------|
| 16 tok | 3443 ms | 3716 ms | **520 ms** | 215 ms | **33.4 ms** | 6.63x / 6.43x |
| 32 tok | 6947 ms | 7117 ms | **1056 ms** | 215 ms | **35.1 ms** | 6.58x / 6.11x |
| 64 tok | 13939 ms | 14509 ms | **2222 ms** | 220 ms | **38.1 ms** | 6.27x / 5.78x |
| 128 tok | 28268 ms | 29628 ms | **4964 ms** | 231 ms | **45.9 ms** | 5.70x / 5.03x |

`model_bench` 的 prefill 段走 batched `prefill()`，decode 仍逐步 `forward`，用来对比三种 `WeightMode`。端到端 CLI 同模型上曾测 prefill 约 **75 tok/s**、decode 约 **27 tok/s**（35+24 token，temperature=0）。

用法：

```bash
# 对指定模型对比（或直接运行，默认测 models/ 下三个模型）
clang++ -std=c++20 -O3 -march=native -Iinclude tests/model_bench.cpp \
  src/core/engine.cpp src/core/gguf.cpp src/core/quant.cpp src/core/sampler.cpp \
  src/core/thread_pool.cpp src/core/tokenizer.cpp -o /tmp/model_bench -pthread
/tmp/model_bench models/Qwen3-0.6B-Q8_0.gguf
```

---

## Web 对话界面

内置一个**单文件前端**（`web/index.html`，纯 HTML/CSS/原生 JS，零构建、零依赖）。
启动后浏览器访问服务端根路径即可进入对话模式：

```bash
./cllm_inference --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080
# 打开 http://127.0.0.1:8080/
```

### 功能

- **多轮对话**：历史在前端维护，完整 `messages` 回传；后端按角色套原生模板，并复用公共前缀 KV
- **SSE 流式输出**：按完整 UTF-8 码点显示；前端中止后后端立即停算并释放引擎锁
- **当前模型展示**：从 `/v1/model/info` 读取架构、层数、上下文、模板格式
- **模型库**：内置 Qwen 2.5 / Qwen 3 / Llama 3.1 / Llama 3.2 的 GGUF 清单（均 ≤30B）
  —— 点击模型弹出**下载 + 启动命令**。
- **采样参数调节**：温度、top-p、top-k、最大 tokens 在输入区下方实时可调

### 说明

- 模型切换采用「重启」方式：前端仅展示启动命令，执行方需在终端 `Ctrl+C` 停掉当前进程后
  用新的 `--model` 重新启动。这是有意为之的设计——运行期热切换模型在单进程下易导致内存波动与卡顿。
- 模型文件本身**不**纳入版本控制，请按前端弹出的 `curl` 命令自行下载到 `models/` 目录。
- 如需自定义前端目录，设置环境变量 `CLLM_WEB_ROOT=/path/to/web` 后启动。

---

## 项目结构

```
cllm_inference/
├── CMakeLists.txt          # CMake 构建配置（含 install / test 目标）
├── Makefile                # Make 构建配置（含 make test / bench）
├── main.cpp                # CLI + HTTP 双入口
├── include/cllm/
│   ├── core/               # 引擎核心头文件（gguf/engine/quant/sampler/...）
│   └── server/             # HTTP 服务头文件
├── src/
│   ├── core/               # 引擎核心实现
│   └── server/             # HTTP 服务实现
├── tests/                  # 单元测试与基准
│   ├── quant_test.cpp      # 13 种量化反量化 + matmul_quant / batch
│   ├── tokenizer_test.cpp  # 哈希词表、多轮模板、多 EOS、UTF-8 流式
│   ├── engine_kv_test.cpp  # Prefix Cache / n_ctx / skip-logits
│   ├── threadpool_test.cpp # 线程池并行一致性
│   ├── sampler_test.cpp    # 采样器行为
│   ├── bench.cpp           # 量化内核吞吐基准
│   ├── model_bench.cpp     # 模型级「量化 vs fp32」对比基准
│   └── run_tests.sh        # 一键运行
└── web/                    # 单文件前端（HTTP 服务自动托管）
    ├── index.html          # 对话界面 + 模型库 + SSE 流式
    └── models.json         # Qwen / Llama ≤30B 的 GGUF 清单
```
