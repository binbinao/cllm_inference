# GGUF 推理服务引擎 需求规范

> **参考链接**：无（用户口述需求，从 0 自研，不依赖 llama.cpp / ggml）

## TAPD评论补充信息

> 无 TAPD 输入。补充的需求点来自用户澄清：
> - 需要 CPU 多线程并行（线程池 + matmul 分块）
> - 需要 int4 / int8 CPU 量化内核加速
> - HTTP 服务需支持 OpenAI 兼容协议
> - HTTP 服务需支持流式 SSE 响应

## Purpose

从 0 构建基于 C++20 的 GGUF 模型推理服务引擎，支持本地命令行推理与内嵌 HTTP 推理服务（OpenAI 兼容协议 + SSE 流式响应），并通过 metadata 驱动做到对 Transformer 类模型的尽量通用支持；核心计算需具备 CPU 多线程并行与 int4/int8 量化内核加速能力。

## Requirements

### 模型解析与加载

#### REQ-1: GGUF 文件解析

系统 SHALL 解析 GGUF 格式模型文件，支持 v2 / v3 版本，提取头部（magic、version）、metadata KV 字典与 tensor info 表。

##### Scenario 1.1: 解析合法 GGUF 文件

- **WHEN** 用户载入一个 magic 为 `GGUF`、版本为 2 或 3 的模型文件
- **THEN** the system SHALL 正确读取 metadata KV（含架构名、层数、隐藏维、注意力变体等超参）
- **AND** 构建完整的 tensor info 表（name、维度、类型、文件偏移）

##### Scenario 1.2: 解析非法文件

- **WHEN** 文件 magic 不匹配或版本不支持
- **THEN** the system SHALL 返回明确错误信息并中止加载
- **AND** 不加载任何权重

**需求依据**：用户口述需求"基于 C 的推理服务引擎支持 gguf 模型推理"，从 0 自研
**实现优先级**：必须
**依赖关系**：无

#### REQ-2: 张量权重加载与量化反量化

系统 SHALL 通过 mmap 按 tensor info 从文件加载各张量权重，并支持常见 GGUF 数据类型（F32、F16、Q8_0、Q4_0、Q4_K、Q6_K 等）的解码，输出统一的 float 张量供计算使用。

##### Scenario 2.1: 加载非量化权重

- **WHEN** 张量类型为 F32 / F16
- **THEN** the system SHALL 将其转为内部 float 表示

##### Scenario 2.2: 加载量化权重

- **WHEN** 张量类型为 Q4_0 / Q4_K / Q8_0 / Q6_K 等量化格式
- **THEN** the system SHALL 调用对应反量化内核还原为 float 张量
- **AND** 反量化结果数值误差在可接受范围内（与参考实现一致）

**需求依据**：用户澄清"需要 int4 int8 CPU 量化内核加速"
**实现优先级**：必须
**依赖关系**：依赖 REQ-1

### 推理内核

#### REQ-3: 通用 Transformer 前向推理

系统 SHALL 提供尽量通用的 Transformer 前向计算，抽象出 RMSNorm、RoPE、Multi-Head / Grouped-Query Attention、SwiGLU FFN，并维护 KV Cache 以支持自回归生成。

##### Scenario 3.1: 单步前向

- **WHEN** 输入一个 token 及历史 KV Cache
- **THEN** the system SHALL 计算并输出该位置的 logits，并更新 KV Cache

##### Scenario 3.2: 架构参数差异

- **WHEN** 不同模型在 metadata 中声明不同的层数 / 隐藏维 / 注意力头数 / GQA 分组数 / RoPE 参数
- **THEN** the system SHALL 依据 metadata 参数化构建计算图，而非硬编码单一架构

**需求依据**：用户口述"尽量通用"
**实现优先级**：必须
**依赖关系**：依赖 REQ-2

#### REQ-4: CPU 多线程并行加速

系统 SHALL 提供线程池，对矩阵乘 / 大张量运算按行或分块进行多线程并行，充分利用多核 CPU。

##### Scenario 4.1: 大矩阵乘并行

- **WHEN** 执行线性层 matmul（权重矩阵较大）
- **THEN** the system SHALL 将计算拆分到线程池多个worker并行执行
- **AND** 结果与非并行实现一致

##### Scenario 4.2: 线程数配置

- **WHEN** 用户通过配置/参数指定线程数
- **THEN** the system SHALL 使用该线程数初始化线程池（默认取硬件并发数）

**需求依据**：用户澄清"需要多线程并行"
**实现优先级**：必须
**依赖关系**：依赖 REQ-3

#### REQ-5: int4 / int8 量化内核加速

系统 SHALL 在 CPU 上实现 int4 / int8 量化矩阵乘内核，对量化权重直接进行计算（或先反量化再乘），相比 naive float 实现获得明显加速。

##### Scenario 5.1: int4 权重矩阵乘

- **WHEN** 线性层权重为 Q4 系列量化类型
- **THEN** the system SHALL 使用 int4 内核（或反量化后加速路径）完成 x·Wᵀ
- **AND** 输出与 float 参考结果误差在采样可接受范围内

##### Scenario 5.2: int8 权重矩阵乘

- **WHEN** 线性层权重为 Q8_0 等 int8 量化类型
- **THEN** the system SHALL 使用 int8 内核完成计算

**需求依据**：用户澄清"需要 int4 int8 CPU 量化内核加速"
**实现优先级**：必须
**依赖关系**：依赖 REQ-2、REQ-4

#### REQ-6: Tokenizer 编解码

系统 SHALL 基于 GGUF 内置 vocab（token 表、scores、特殊 token、合并规则）实现文本编码与解码。

##### Scenario 6.1: 编码

- **WHEN** 输入 prompt 文本
- **THEN** the system SHALL 输出对应的 token id 序列（含 BOS 处理）

##### Scenario 6.2: 解码

- **WHEN** 模型生成一个 token id
- **THEN** the system SHALL 转换为文本片段（含合并规则还原）

**需求依据**：用户口述推理服务基本能力
**实现优先级**：必须
**依赖关系**：依赖 REQ-1

#### REQ-7: 采样与生成控制

系统 SHALL 支持 greedy / temperature / top-p / top-k 采样，并支持 max_tokens、stop 序列等生成终止条件。

##### Scenario 7.1: 温度采样

- **WHEN** 用户提供 temperature>0
- **THEN** the system SHALL 对 logits 施加温度并做概率采样

##### Scenario 7.2: 终止条件

- **WHEN** 生成达到 max_tokens 或命中 stop 序列或遇到 EOS
- **THEN** the system SHALL 停止生成并返回结果

**需求依据**：推理服务基本能力
**实现优先级**：必须
**依赖关系**：依赖 REQ-3、REQ-6

### 服务入口

#### REQ-8: 本地 CLI 推理入口

系统 SHALL 提供命令行入口，加载模型文件、接收 prompt、并以流式/非流式方式输出生成文本。

##### Scenario 8.1: 本地生成

- **WHEN** 用户执行 `cllm_inference --model <path> --prompt "..."`
- **THEN** the system SHALL 加载模型并输出生成文本到 stdout

**需求依据**：用户口述"本地+http"
**实现优先级**：必须
**依赖关系**：依赖 REQ-1~REQ-7

#### REQ-9: HTTP 推理服务（OpenAI 兼容 + SSE）

系统 SHALL 内嵌自研 HTTP 服务器（不引入第三方框架），提供 OpenAI 兼容接口，并支持 SSE 流式响应。

##### Scenario 9.1: 兼容接口

- **WHEN** 客户端请求 `/v1/completions` 或 `/v1/chat/completions`
- **THEN** the system SHALL 解析符合 OpenAI 格式的请求体（model、messages/prompt、max_tokens、temperature、stream 等）
- **AND** 返回符合 OpenAI 格式的响应体

##### Scenario 9.2: 健康检查

- **WHEN** 客户端请求 `/health`
- **THEN** the system SHALL 返回服务健康状态（如 `{"status":"ok"}`）

##### Scenario 9.3: SSE 流式响应

- **WHEN** 请求中 `stream=true`
- **THEN** the system SHALL 以 `text/event-stream` 逐 token 推送 `data: {...}` 事件
- **AND** 结束时推送 `data: [DONE]`

##### Scenario 9.4: 并发请求

- **WHEN** 多个客户端同时发起请求
- **THEN** the system SHALL 使用请求处理线程池并发处理，互不阻塞

**需求依据**：用户澄清"需要支持 OpenAI 兼容协议"+"需要 SSE 响应"+"本地+http"
**实现优先级**：必须
**依赖关系**：依赖 REQ-1~REQ-8

### 通用架构适配

#### REQ-10: 通用架构适配层

系统 SHALL 通过 metadata 字段（架构名、层数、隐藏维、注意力变体等）注册/选择对应计算路径，使引擎尽量通用地支持不同 Transformer 类模型。

##### Scenario 10.1: 新架构接入

- **WHEN** 模型 metadata 声明一个已注册架构
- **THEN** the system SHALL 自动选择对应计算路径完成加载与推理

##### Scenario 10.2: 未知架构

- **WHEN** metadata 声明未注册架构
- **THEN** the system SHALL 给出明确提示，仍可尝试按通用 Transformer 路径加载（如适用）

**需求依据**：用户口述"尽量通用"
**实现优先级**：必须
**依赖关系**：依赖 REQ-3

## 待技术设计阶段确认的问题

以下问题将在技术设计阶段基于代码分析确认：

1. **GGUF 数据类型覆盖范围**：首版需完整支持哪些量化类型（Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 等），是否分阶段实现。
2. **线程池与量化内核的执行粒度**：matmul 分块策略与 int4/int8 内核的向量化指令（如是否需要 NEON / AVX2 内在函数）。
3. **HTTP 服务器并发模型**：采用每请求一线程、线程池还是基于 select/poll 的事件循环（纯 C++ 标准库实现，不引入第三方库）。
4. **SSE 与流式生成的回压/取消**：客户端断开连接时如何及时终止生成。
5. **Tokenizer 合并规则**：是否支持 BPE / SentencePiece 两种合并方式，首版范围。
6. **OpenAI 接口字段完整性**：首版需对齐的字段集合（是否包含 logprobs、usage 等）。
