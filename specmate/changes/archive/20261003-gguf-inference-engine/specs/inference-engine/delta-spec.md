<!-- Delta Spec: 新增通用 Transformer 推理内核 -->
<!-- Change ID: 20261003-gguf-inference-engine -->
<!-- Module Status: NEW -->

## Purpose

提供 metadata 驱动的通用 Transformer 前向推理能力（RMSNorm、RoPE、GQA Attention、SwiGLU FFN、KV Cache），并通过多线程与 int4/int8 量化内核在 CPU 上加速，解决推理引擎核心计算缺失的问题。

## ADDED Requirements

### Requirement: 通用 Transformer 前向推理

系统 SHALL 提供尽量通用的 Transformer 前向计算，抽象出 RMSNorm、RoPE、Multi-Head / Grouped-Query Attention、SwiGLU FFN，并维护 KV Cache 以支持自回归生成。

#### Scenario: 单步前向
- **WHEN** 输入一个 token 及历史 KV Cache
- **THEN** 系统 SHALL 计算并输出该位置的 logits，并更新 KV Cache

#### Scenario: 架构参数差异
- **WHEN** 不同模型在 metadata 中声明不同的层数 / 隐藏维 / 注意力头数 / GQA 分组数 / RoPE 参数
- **THEN** 系统 SHALL 依据 metadata 参数化构建计算，而非硬编码单一架构

### Requirement: CPU 多线程并行加速

系统 SHALL 提供线程池，对矩阵乘 / 大张量运算按行或分块进行多线程并行，充分利用多核 CPU。

#### Scenario: 大矩阵乘并行
- **WHEN** 执行线性层 matmul（权重矩阵较大）
- **THEN** 系统 SHALL 将计算拆分到线程池多个 worker 并行执行
- **AND** 结果与非并行实现一致

#### Scenario: 线程数配置
- **WHEN** 用户通过配置/参数指定线程数
- **THEN** 系统 SHALL 使用该线程数初始化线程池（默认取硬件并发数）

### Requirement: int4 / int8 量化内核加速

系统 SHALL 在 CPU 上实现 int4 / int8 量化矩阵乘内核，对量化权重直接进行计算（或先反量化再乘），相比 naive float 实现获得明显加速。

#### Scenario: int4 权重矩阵乘
- **WHEN** 线性层权重为 Q4 系列量化类型
- **THEN** 系统 SHALL 使用 int4 内核完成 x·Wᵀ
- **AND** 输出与 float 参考结果误差在采样可接受范围内

#### Scenario: int8 权重矩阵乘
- **WHEN** 线性层权重为 Q8_0 等 int8 量化类型
- **THEN** 系统 SHALL 使用 int8 内核完成计算

### Requirement: 通用架构适配层

系统 SHALL 通过 metadata 字段（架构名、层数、隐藏维、注意力变体等）选择对应计算路径，使引擎尽量通用地支持不同 Transformer 类模型。

#### Scenario: 新架构接入
- **WHEN** 模型 metadata 声明一个已支持的架构
- **THEN** 系统 SHALL 自动选择对应计算路径完成加载与推理

#### Scenario: 未知架构
- **WHEN** metadata 声明未支持的架构
- **THEN** 系统 SHALL 给出明确提示，仍可尝试按通用 Transformer 路径加载（如适用）
