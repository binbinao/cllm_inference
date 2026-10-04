# 实施计划

## 目录

- [实施计划](#实施计划)
  - [目录](#目录)
  - [用户前置任务](#用户前置任务)
  - [编码任务](#编码任务)
    - [批次1：基础设施层（构建+张量+量化+线程池）](#批次1基础设施层构建张量量化线程池)
    - [批次2：核心推理层（解析+引擎+分词+采样）](#批次2核心推理层解析引擎分词采样)
    - [批次3：服务入口层（HTTP+CLI）](#批次3服务入口层httpcli)

## 用户前置任务

无（Catch2 为 FetchContent 自动拉取的测试依赖，无需用户处理）

## 编码任务

> **注意**：编码任务仅包含本仓库可直接修改的内容。`design.md` 中的"非当前仓库依赖"应归入上方 `## 用户前置任务`。

**批次划分依据**：按依赖/层次 —— 基础设施层（构建/张量/量化/线程池）→ 核心推理层（解析/引擎/分词/采样）→ 服务入口层（HTTP/CLI）

### 批次1：基础设施层（构建+张量+量化+线程池）

**依据**：底层被依赖模块优先，后续批次依赖本批次的张量、量化内核与线程池

- [✓] 任务1：修改 - CMakeLists.txt 多模块目标组织
  - **对应design.md**：`D-1`
  - **对应需求**：`Requirement: 构建系统`
  - **改动文件**：`CMakeLists.txt`

- [✓] 任务2：新增 - 张量定义与 mmap 视图
  - **对应design.md**：`D-3`
  - **对应需求**：`Requirement: 张量权重加载与量化反量化`
  - **改动文件**：`include/cllm/core/tensor.hpp`

- [✓] 任务3：新增 - 量化类型枚举与反量化内核
  - **对应design.md**：`D-4`
  - **对应需求**：`Requirement: 张量权重加载与量化反量化 / int4-int8量化内核加速`
  - **改动文件**：`include/cllm/core/quant.hpp`、`src/core/quant.cpp`

- [✓] 任务4：新增 - 线程池
  - **对应design.md**：`D-5`
  - **对应需求**：`Requirement: CPU多线程并行加速`
  - **改动文件**：`include/cllm/core/thread_pool.hpp`、`src/core/thread_pool.cpp`

### 批次2：核心推理层（解析+引擎+分词+采样）

**依据**：依赖批次1的张量/量化/线程池，实现模型解析与推理核心

- [✓] 任务1：新增 - GGUF 解析器
  - **对应design.md**：`D-2`
  - **对应需求**：`Requirement: GGUF文件解析`
  - **改动文件**：`include/cllm/core/gguf.hpp`、`src/core/gguf.cpp`

- [✓] 任务2：新增 - 通用 Transformer 推理引擎
  - **对应design.md**：`D-6`
  - **对应需求**：`Requirement: 通用Transformer前向推理 / 通用架构适配层`
  - **改动文件**：`include/cllm/core/engine.hpp`、`src/core/engine.cpp`

- [✓] 任务3：新增 - Tokenizer 编解码
  - **对应design.md**：`D-7`
  - **对应需求**：`Requirement: Tokenizer编解码`
  - **改动文件**：`include/cllm/core/tokenizer.hpp`、`src/core/tokenizer.cpp`

- [✓] 任务4：新增 - 采样器
  - **对应design.md**：`D-8`
  - **对应需求**：`Requirement: 采样与生成控制`
  - **改动文件**：`include/cllm/core/sampler.hpp`、`src/core/sampler.cpp`

### 批次3：服务入口层（HTTP+CLI）

**依据**：依赖批次2的引擎/分词/采样，提供 HTTP 服务与 CLI 入口

- [✓] 任务1：新增 - HTTP 推理服务器（OpenAI 兼容 + SSE）
  - **对应design.md**：`D-9`
  - **对应需求**：`Requirement: HTTP推理服务（OpenAI兼容+SSE）`
  - **改动文件**：`include/cllm/server/http_server.hpp`、`src/server/http_server.cpp`

- [✓] 任务2：修改 - CLI / 服务入口 main.cpp
  - **对应design.md**：`D-10`
  - **对应需求**：`Requirement: 本地CLI推理入口`
  - **改动文件**：`src/main.cpp`
