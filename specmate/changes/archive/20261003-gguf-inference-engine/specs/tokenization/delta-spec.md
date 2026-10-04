<!-- Delta Spec: 新增分词与采样能力 -->
<!-- Change ID: 20261003-gguf-inference-engine -->
<!-- Module Status: NEW -->

## Purpose

基于 GGUF 内置词表实现文本与 token 的编解码，并对模型输出 logits 施加温度、top-k、top-p 采样，解决推理引擎缺少文本预处理与生成控制的问题。

## ADDED Requirements

### Requirement: Tokenizer 编解码

系统 SHALL 基于 GGUF 内置 vocab（token 表、scores、特殊 token、合并规则）实现文本编码与解码。

#### Scenario: 编码
- **WHEN** 输入 prompt 文本
- **THEN** 系统 SHALL 输出对应的 token id 序列（含 BOS 处理）

#### Scenario: 解码
- **WHEN** 模型生成一个 token id
- **THEN** 系统 SHALL 转换为文本片段（含合并规则还原）

#### Scenario: 双合并方式
- **WHEN** 词表采用 SentencePiece（llama 类）或 BPE（gpt2）合并方式
- **THEN** 系统 SHALL 依据 `tokenizer.ggml.model` 选择对应编码路径

### Requirement: 采样与生成控制

系统 SHALL 支持 greedy / temperature / top-p / top-k 采样，并支持 max_tokens、stop 序列等生成终止条件。

#### Scenario: 温度采样
- **WHEN** 用户提供 temperature > 0
- **THEN** 系统 SHALL 对 logits 施加温度并做概率采样

#### Scenario: greedy 退化
- **WHEN** temperature = 0
- **THEN** 系统 SHALL 退化为取最大概率 token（greedy）

#### Scenario: 终止条件
- **WHEN** 生成达到 max_tokens 或命中 stop 序列或遇到 EOS
- **THEN** 系统 SHALL 停止生成并返回结果
