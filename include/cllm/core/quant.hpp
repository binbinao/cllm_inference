#pragma once

#include <cstdint>
#include <cstddef>

namespace cllm {

class ThreadPool;

// GGML 量化类型枚举（与 gguf 格式的 tensor type 对齐）
enum class GgmlType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
};

// 每种类型的块大小（每个 block 包含的元素数）
int block_size(GgmlType type);

// 每种类型每个 block 占用的字节数
size_t type_size(GgmlType type);

// 将一段量化数据反量化为 float。
// src: 指向量化数据起始位置；dst: 输出 float 缓冲（长度 >= n）
// n: 元素总数（会按 block 对齐处理）
// 当前为标量实现，覆盖 GGUF 常用全部类型，入口统一以便后续替换 SIMD 内核。
void dequantize_block(const void* src, float* dst, GgmlType type, int64_t n);

// 量化权重矩阵乘：y[out_dim] = x[in_dim] · W[out_dim, in_dim]ᵀ (+ bias)
// 对量化权重按输出行惰性反量化，仅保留行级 float 缓冲，避免整块权重常驻内存。
// w: 指向量化权重数据（如 mmap 区域）；type: 权重量化类型；
// bias: 可选偏置（长度 out_dim）；pool: 可选线程池，按输出行分块并行；
// 要求 in_dim 能被该类型的 block_size 整除。
void matmul_quant(const float* x, const void* w, GgmlType type,
                  const float* bias, int in_dim, int out_dim, float* y,
                  ThreadPool* pool = nullptr);

}  // namespace cllm
