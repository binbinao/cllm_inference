# inference-service Specification

## Purpose

提供本地命令行推理与内嵌 HTTP 推理服务两种入口（OpenAI 兼容协议 + SSE 流式响应），解决推理引擎缺乏对外服务能力的问题。

## Requirements

### Requirement: 本地 CLI 推理入口

系统 SHALL 提供命令行入口，加载模型文件、接收 prompt、并以流式/非流式方式输出生成文本。

#### Scenario: 本地生成
- **WHEN** 用户执行 `cllm_inference --model <path> --prompt "..."`
- **THEN** 系统 SHALL 加载模型并输出生成文本到 stdout

### Requirement: HTTP 推理服务（OpenAI 兼容 + SSE）

系统 SHALL 内嵌自研 HTTP 服务器（不引入第三方框架），提供 OpenAI 兼容接口，并支持 SSE 流式响应。

#### Scenario: 兼容接口
- **WHEN** 客户端请求 `/v1/completions` 或 `/v1/chat/completions`
- **THEN** 系统 SHALL 解析符合 OpenAI 格式的请求体（model、messages/prompt、max_tokens、temperature、stream 等）
- **AND** 返回符合 OpenAI 格式的响应体

#### Scenario: 健康检查
- **WHEN** 客户端请求 `/health`
- **THEN** 系统 SHALL 返回服务健康状态

#### Scenario: SSE 流式响应
- **WHEN** 请求中 `stream=true`
- **THEN** 系统 SHALL 以 `text/event-stream` 逐 token 推送 `data: {...}` 事件
- **AND** 结束时推送 `data: [DONE]`

#### Scenario: 并发请求
- **WHEN** 多个客户端同时发起请求
- **THEN** 系统 SHALL 并发处理，互不阻塞
