#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

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

// IEEE 754 半精度（fp16）位模式 <-> float 转换（用于 fp16 反量化推理路径）
// inline 定义以便在矩阵乘热循环中内联，避免逐元素函数调用开销。
inline float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FF;
            f = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(uint32_t));

    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x007FFFFFu;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;

    // NaN / Inf
    if (((x >> 23) & 0xFFu) == 0xFFu) {
        if (mant) return (uint16_t)(sign | 0x7E00u);  // NaN -> quiet NaN
        return (uint16_t)(sign | 0x7C00u);            // Inf
    }

    if (exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u);  // overflow -> Inf
    }

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;  // underflow -> signed zero
        mant |= 0x00800000u;                     // implicit leading 1
        uint32_t t = (uint32_t)(14 - exp);       // shift amount
        uint32_t a = (1u << (t - 1)) - 1u;       // rounding bias
        uint32_t b = (mant >> t) & 1u;           // round bit
        mant = (mant + a + b) >> t;
        return (uint16_t)(sign | mant);
    }

    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

// 每种类型的块大小（每个 block 包含的元素数）
int block_size(GgmlType type);

// 每种类型每个 block 占用的字节数
size_t type_size(GgmlType type);

// 将一段量化数据反量化为 float。
// src: 指向量化数据起始位置；dst: 输出 float 缓冲（长度 >= n）
// n: 元素总数（会按 block 对齐处理）
// 当前为标量实现，覆盖 GGUF 常用全部类型，入口统一以便后续替换 SIMD 内核。
void dequantize_block(const void* src, float* dst, GgmlType type, int64_t n);

// 将一段量化数据反量化为 fp16（半精度位模式）。
// src: 指向量化数据起始位置；dst: 输出 fp16 位模式缓冲（长度 >= n）
// n: 元素总数（会按 block 对齐处理）
// 用于「反量化 fp16」推理路径：权重以 fp16 精度常驻，矩阵乘时 half -> float 累加。
void dequantize_block_f16(const void* src, uint16_t* dst, GgmlType type, int64_t n);

// =========================================================================
// 量化矩阵乘加速：激活在线量化 + 整数点积
// =========================================================================
//
// 旧路径（兜底）：权重按行反量化 -> float 点积
// 新路径（更快）：激活在线量化为 Q8 -> 整数点积 -> float 累加
//
// 新路径对已注册 vec_dot 内核的权重类型生效，其他类型自动走旧路径，向后兼容。

// 激活量化格式：Q8_0（block=32），每块一个 f32 scale + 32 x int8
// 对于 Q4_K/Q5_K/Q6_K/Q2_K/Q3_K 走 Q8_K（block=256，带 sum 预计算），后续按需扩展。
struct QuantizedQ8_0 {
    std::vector<float> scales;   // 每块 scale，长度 = n / 32
    std::vector<int8_t> qs;      // 量化值，长度 = n（n 需为 32 的倍数）
    // 每块 qs 之和（int32），用于 Q4_K/Q5_K 等 "dmin*m*sum(q_x)" 项的高效计算
    std::vector<int32_t> bsums;
    int64_t n = 0;               // 元素总数
};

// 将 float 向量量化为 Q8_0 格式（block=32，absmax scale）
void quantize_row_q8_0(const float* x, QuantizedQ8_0& out, int64_t n);

// 查询某种权重量化类型是否有 vec_dot 快速路径（当前支持 Q4_0 / Q5_0 / Q8_0 / Q4_K / Q5_K / Q6_K）
bool has_vec_dot(GgmlType type);

// 对单行权重做 "量化激活 · 量化权重" 的整数点积，返回 float 累加结果。
// 要求 n 能被权重类型的 block_size 整除。
// 用于 matmul_quant 内部，一般不直接调用。
float vec_dot_row(const void* w_row, GgmlType w_type,
                  const QuantizedQ8_0& qx, int64_t n);

// 量化权重矩阵乘：y[out_dim] = x[in_dim] · W[out_dim, in_dim]ᵀ (+ bias)
// 优化策略：
//   - 若 w_type 支持 vec_dot：先把 x 量化为 Q8_0（一次性），然后每行走整数点积
//   - 否则：旧路径，按行反量化为 float 后做 float 点积
// 要求 in_dim 能被该类型的 block_size 整除。
void matmul_quant(const float* x, const void* w, GgmlType type,
                  const float* bias, int in_dim, int out_dim, float* y,
                  ThreadPool* pool = nullptr);

// 权重静止的批量矩阵乘：X[n_rows, in_dim] · W[out_dim, in_dim]ᵀ
// y 行主序 [n_rows, out_dim]。每个权重行只读一次，供 batched prefill 使用。
void matmul_quant_batch(const float* x, const void* w, GgmlType type,
                        const float* bias, int in_dim, int out_dim, int n_rows,
                        float* y, ThreadPool* pool = nullptr);

}  // namespace cllm
