# cllm_inference

自研 **C++20 GGUF 推理引擎**：从零实现，零第三方运行时依赖，支持加载 GGUF 量化模型并在 CPU 上完成推理，提供 OpenAI 兼容的 HTTP 服务。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Architecture diagram](https://gitdiagram.com/diagram-badge.svg)](https://gitdiagram.com/binbinao/cllm_inference?utm_source=readme&utm_medium=badge)

## 特性

- **纯 C++20 / 零依赖**：仅依赖标准库与 POSIX 系统调用，无任何第三方库
- **完整 GGUF 解析**：支持 GGUF v2/v3，mmap 零拷贝加载
- **量化反量化**：支持 F32 / F16 / Q8_0 / Q5_0 / Q4_K / Q6_K 等 GGML 量化格式
- **通用 Transformer 内核**：RMSNorm、NEOX RoPE、GQA Attention、SwiGLU FFN、KV Cache
- **CPU 多线程**：手写线程池，按行/分块并行矩阵乘
- **分词与采样**：BPE（gpt2 类）+ SentencePiece 双模式，支持 temperature / top-p / top-k 采样
- **对话模板**：自动识别 Qwen 等 chat 模型并应用 `<|im_start|>` 对话模板
- **双入口**：本地 CLI 推理 + OpenAI 兼容 HTTP 服务（含 SSE 流式）

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
                        │              KV Cache（自回归）             │
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
| `Quant` | GGML 量化权重（Q4_K/Q5_0/Q6_K/Q8_0/F32/F16）反量化为 float32 | `src/core/quant.cpp` |
| `TransformerEngine` | 单步前向：RMSNorm + NEOX RoPE + GQA Attention + SwiGLU FFN，更新 KV Cache | `src/core/engine.cpp` |
| `Tokenizer` | BPE（gpt2 类）/ SentencePiece 双模式编解码 + Qwen chat template | `src/core/tokenizer.cpp` |
| `Sampler` | temperature / top-k / top-p 采样，从 logits 采样下一个 token | `src/core/sampler.cpp` |
| `ThreadPool` | 手写线程池，按行/分块并行矩阵乘 | `src/core/thread_pool.cpp` |
| `HttpServer` | 纯标准库实现的 OpenAI 兼容 HTTP 服务（含 SSE 流式） | `src/server/http_server.cpp` |

> 推理采用自回归流程：`main.cpp` 先 forward 整个 prompt，随后循环「采样 → decode → forward 下一个 token」直至 EOS 或达到 `max_tokens`。CLI 与 HTTP 两种入口复用同一套 `Tokenizer` 与 `TransformerEngine`。

---

## 依赖

| 依赖 | 要求 | 说明 |
|------|------|------|
| 编译器 | C++20 支持（clang++ 或 g++） | macOS 自带 clang++，Linux 用 g++/clang++ |
| 构建工具 | `make` 或 `cmake`（二选一） | 推荐 `make`，无需 cmake |

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

> 已支持架构：`qwen2`（含 GQA、bias 投影、NEOX RoPE）。模型量化类型需覆盖 Q5_0 / Q4_K / Q6_K / Q8_0 / F32 / F16。

### 2. 本地 CLI 推理

```bash
# 基础问答（greedy 采样）
cllm_inference --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "你好" --temperature 0 --max-tokens 40

# 创意生成（随机采样）
cllm_inference --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "写一首关于春天的短诗" --temperature 0.8 --top-p 0.9 --max-tokens 60

# 指定线程数
cllm_inference --model models/xxx.gguf --prompt "你好" --threads 8
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

**对话补全**

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"写一首关于春天的短诗"}],"temperature":0.8}'
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
| `--temperature <f>` | 采样温度（0 = greedy） | 0.8 |
| `--top-p <f>` | 核采样（nucleus） | 0.9 |
| `--top-k <n>` | Top-k 采样 | 40 |
| `--max-tokens <n>` | 最大生成 token 数 | 64 |

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/v1/completions` | 文本补全（`prompt` + 采样参数，支持 `stream`） |
| POST | `/v1/chat/completions` | 对话补全（`messages` 数组，取 content 拼接） |

---

## 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `missing tensor: blk.N.xxx` | 模型张量命名与当前支持架构不匹配 | 换用 qwen2/llama 架构的 GGUF |
| `unsupported ggml type` | 模型含未实现的量化类型 | 换用 Q4_K_M / Q5 等已支持量化 |
| 输出乱码 | 模型为 chat 模型但未走对话模板 | 确认模型含 `tokenizer.chat_template` |
| 加载缓慢 | 首次全量反量化权重到内存 | 属预期，Q4_K_M 约需数 GB 内存 |

---

## 项目结构

```
cllm_inference/
├── CMakeLists.txt          # CMake 构建配置（含 install 规则）
├── Makefile                # Make 构建配置（推荐）
├── main.cpp                # CLI + HTTP 双入口
├── include/cllm/
│   ├── core/               # 引擎核心头文件（gguf/engine/quant/sampler/...）
│   └── server/             # HTTP 服务头文件
└── src/
    ├── core/               # 引擎核心实现
    └── server/             # HTTP 服务实现
```
