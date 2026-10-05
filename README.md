# cllm_inference

自研 **C++20 GGUF 推理引擎**：从零实现，零第三方运行时依赖，支持加载 GGUF 量化模型并在 CPU 上完成推理，提供 OpenAI 兼容的 HTTP 服务。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Architecture diagram](https://gitdiagram.com/diagram-badge.svg)](https://gitdiagram.com/binbinao/cllm_inference?utm_source=readme&utm_medium=badge)

## 特性

- **纯 C++20 / 零依赖**：仅依赖标准库与 POSIX 系统调用，无任何第三方库
- **完整 GGUF 解析**：支持 GGUF v2/v3，mmap 零拷贝加载
- **量化类型全覆盖**：支持 F32 / F16 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q8_1 / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K 共 13 种 GGML 量化格式
- **零拷贝量化推理**：量化权重直接引用 mmap 数据，按输出行惰性反量化，无需加载期全量展开为 float32（0.5B Q4_K_M 常驻内存 ~430MB）
- **通用 Transformer 内核**：RMSNorm、NEOX RoPE、GQA Attention、SwiGLU FFN、KV Cache
- **CPU 多线程**：手写线程池，按行/分块并行矩阵乘
- **分词与采样**：BPE（gpt2 类）+ SentencePiece 双模式，支持 temperature / top-p / top-k 采样
- **多格式对话模板**：自动识别 ChatML（Qwen）/ Llama 3 / Llama 2·Mistral 并应用对应模板
- **双入口**：本地 CLI 推理 + OpenAI 兼容 HTTP 服务（含 SSE 流式）
- **内置 Web UI**：单文件 HTML 对话前端，支持流式输出、模型库浏览与启动命令生成（Qwen/Llama ≤30B）
- **自带测试与基准**：13 种量化反量化 + 矩阵乘内核 + 线程池 + 采样器共 135 项断言，含吞吐基准

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
| GET | `/v1/model/info` | 当前加载模型的元信息（架构、层数、上下文、模板格式） |
| GET | `/v1/models` | OpenAI 兼容的模型列表（当前模型包装为 data 数组） |
| POST | `/v1/completions` | 文本补全（`prompt` + 采样参数，支持 `stream`） |
| POST | `/v1/chat/completions` | 对话补全（`messages` 数组，返回 OpenAI chat.completion 结构） |
| OPTIONS | `/*` | CORS 预检，允许所有源 |
| GET | `/` 、`/web/*`、`/models.json` | 托管运行目录下 `web/` 前端静态资源（若存在） |

---

## 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `missing tensor: blk.N.xxx` | 模型张量命名与当前支持架构不匹配 | 换用 qwen2/llama 架构的 GGUF |
| `unsupported ggml type` | 模型含未实现的量化类型 | 已支持 13 种常用类型；如遇新类型需扩展 `quant.cpp` |
| `in_dim not divisible by block size` | 张量内维与量化 block 不整除 | 该类模型罕见，通常是转换异常 |
| 输出乱码 | 模型为 chat 模型但未走对话模板 | 确认模型含 `tokenizer.chat_template`；支持 ChatML / Llama3 / Llama2 |
| 内存占用 | 量化权重按需反量化 | 0.5B Q4_K_M 约 430MB（相比全量反量化缓存的 ~3GB 大幅下降） |

> 说明：当前量化内核为**标量惰性反量化**实现，主要收益是**内存**而非吞吐；
> 若需吞吐加速，后续可引入 SIMD 整数点积后端（见 `tests/bench.cpp` 基准数据）。

---

## 测试与基准

```bash
# 运行全部单元测试（量化 / 线程池 / 采样器）
make test

# 追加运行量化矩阵乘吞吐基准
make bench
```

- **覆盖范围**：13 种 GGML 量化类型反量化、`matmul_quant` 内核（与 float 参考逐元素比对）、
  线程池并行一致性、采样器 greedy/top-k/top-p/温度行为，共 **135 项断言**。
- **基准**：`tests/bench.cpp` 报告各量化类型的 GFLOP/s 及相对纯 float matmul 的比值。
- CMake 构建下：`cmake --build build && ctest --test-dir build`。

---

## Web 对话界面

内置一个**单文件前端**（`web/index.html`，纯 HTML/CSS/原生 JS，零构建、零依赖）。
启动后浏览器访问服务端根路径即可进入对话模式：

```bash
./cllm_inference --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080
# 打开 http://127.0.0.1:8080/
```

### 功能

- **多轮对话**：对话历史在前端维护，每次请求完整传回后端
- **SSE 流式输出**：字符逐 token 流式显示，支持中途中止
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
│   ├── quant_test.cpp      # 13 种量化反量化 + matmul_quant
│   ├── threadpool_test.cpp # 线程池并行一致性
│   ├── sampler_test.cpp    # 采样器行为
│   ├── bench.cpp           # 量化内核吞吐基准
│   └── run_tests.sh        # 一键运行
└── web/                    # 单文件前端（HTTP 服务自动托管）
    ├── index.html          # 对话界面 + 模型库 + SSE 流式
    └── models.json         # Qwen / Llama ≤30B 的 GGUF 清单
```
