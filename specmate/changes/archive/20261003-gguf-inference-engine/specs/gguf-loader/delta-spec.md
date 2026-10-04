<!-- Delta Spec: 新增 GGUF 模型解析与加载能力 -->
<!-- Change ID: 20261003-gguf-inference-engine -->
<!-- Module Status: NEW -->

## Purpose

解析 GGUF 格式的量化模型文件并加载权重，为推理引擎提供统一的模型数据（超参 + 张量 + 词表），解决从零自研推理引擎无法读取主流 GGUF 模型的问题。

## ADDED Requirements

### Requirement: GGUF 文件解析

系统 SHALL 解析 GGUF 格式模型文件，支持 v2 / v3 版本，提取头部（magic、version）、metadata KV 字典与 tensor info 表。

#### Scenario: 解析合法 GGUF 文件
- **WHEN** 用户载入一个 magic 为 `GGUF`、版本为 2 或 3 的模型文件
- **THEN** 系统 SHALL 正确读取 metadata KV（架构名、层数、隐藏维、注意力变体等超参）
- **AND** 构建完整的 tensor info 表（name、维度、类型、文件偏移）

#### Scenario: 解析非法文件
- **WHEN** 文件 magic 不匹配或版本不支持
- **THEN** 系统 SHALL 返回明确错误信息并中止加载
- **AND** 不加载任何权重

### Requirement: 张量权重加载与量化反量化

系统 SHALL 通过 mmap 按 tensor info 从文件加载各张量权重，并支持常见 GGUF 数据类型（F32、F16、Q8_0、Q4_0、Q4_K 等）的解码，输出统一的 float 张量供计算使用。

#### Scenario: 加载非量化权重
- **WHEN** 张量类型为 F32 / F16
- **THEN** 系统 SHALL 将其转为内部 float 表示

#### Scenario: 加载量化权重
- **WHEN** 张量类型为 Q4_0 / Q4_K / Q8_0 等量化格式
- **THEN** 系统 SHALL 调用对应反量化内核还原为 float 张量
- **AND** 反量化结果数值误差在可接受范围内

#### Scenario: 词表加载
- **WHEN** GGUF metadata 中包含 tokenizer.ggml.tokens / scores / token_type 等词表字段
- **THEN** 系统 SHALL 将其提取为可供分词器使用的词表结构
