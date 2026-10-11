#include "cllm/core/quant.hpp"
#include "cllm/core/tensor.hpp"
#include "cllm/core/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(__ARM_NEON)
#include <arm_neon.h>
// ARMv8.2+ 提供 vdotq_s32（SDOT 指令），一条指令完成 16 对 int8 乘加
#if defined(__ARM_FEATURE_DOTPROD)
#define CLLM_HAS_SDOT 1
#else
#define CLLM_HAS_SDOT 0
#endif
#endif

namespace cllm {

namespace {

// IEEE 754 half precision -> float
float half_to_float(uint16_t h) {
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

// ---- F32 ----
void deq_f32(const void* src, float* dst, int64_t n) {
    std::memcpy(dst, src, n * sizeof(float));
}

// ---- F16 ----
void deq_f16(const void* src, float* dst, int64_t n) {
    const uint16_t* s = static_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < n; ++i) dst[i] = half_to_float(s[i]);
}

// ---- Q8_0 : block = [d:f16][qs:32 x int8]，value = d * q ----
void deq_q8_0(const void* src, float* dst, int64_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    const int bs = 32;
    for (int64_t i = 0; i < n; i += bs) {
        float d = half_to_float(*reinterpret_cast<const uint16_t*>(s));
        s += 2;
        for (int j = 0; j < bs && i + j < n; ++j) {
            dst[i + j] = d * static_cast<int8_t>(s[j]);
        }
        s += bs;
    }
}

// ---- Q4_0 : block = [d:f16][qs:16B 打包 32 个 nibble]，value = d * (q - 8) ----
// qs[j] 低 nibble = 元素 j（0-15），高 nibble = 元素 j+16（16-31）
void deq_q4_0(const void* src, float* dst, int64_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    const int bs = 32;
    for (int64_t i = 0; i < n; i += bs) {
        float d = half_to_float(*reinterpret_cast<const uint16_t*>(s));
        s += 2;
        for (int j = 0; j < bs / 2; ++j) {
            uint8_t byte = s[j];
            if (i + j < n)      dst[i + j]      = d * ((byte & 0xF) - 8);
            if (i + j + 16 < n) dst[i + j + 16] = d * ((byte >> 4) - 8);
        }
        s += bs / 2;
    }
}

// ---- Q4_K : super block(256) = [d:f16][dmin:f16][scales:12B][qs:128B] ----
// scales 用 get_scale_min_k4 交错解包（见 llama.cpp ggml-quants.c）
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

void deq_q4_k(const void* src, float* dst, int64_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    const int QK = 256;
    for (int64_t i = 0; i < n; i += QK) {
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(s)); s += 2;
        const float dmin = half_to_float(*reinterpret_cast<const uint16_t*>(s)); s += 2;
        const uint8_t* scales = s; s += 12;
        const uint8_t* q = s;      s += QK / 2;

        int is = 0;
        for (int j = 0; j < QK; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, sc, m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) {
                if (i + j + l < n) dst[i + j + l] = d1 * (q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                if (i + j + 32 + l < n) dst[i + j + 32 + l] = d2 * (q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

// ---- Q5_0 : block = [d:f16][qh:4B 高1位][qs:16B 低4位]，value = d * (q - 16) ----
// qs 布局：qs[j] 低 nibble = 元素 j（0-15），高 nibble = 元素 j+16（16-31）
// qh 布局：32 bit，第 j 位 = 元素 j 的高 1 位（小端 uint32）
void deq_q5_0(const void* src, float* dst, int64_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    const int bs = 32;
    for (int64_t i = 0; i < n; i += bs) {
        float d = half_to_float(*reinterpret_cast<const uint16_t*>(s));
        s += 2;
        const uint8_t* qh = s; s += 4;
        const uint8_t* qs = s; s += 16;
        uint32_t qh_bits = (uint32_t)qh[0] | ((uint32_t)qh[1] << 8) |
                           ((uint32_t)qh[2] << 16) | ((uint32_t)qh[3] << 24);
        for (int j = 0; j < 16; ++j) {
            int hi0 = (qh_bits >> j) & 1;
            int hi1 = (qh_bits >> (j + 16)) & 1;
            int q0 = (qs[j] & 0xF) | (hi0 << 4);
            int q1 = (qs[j] >> 4) | (hi1 << 4);
            if (i + j < n)      dst[i + j]      = d * (q0 - 16);
            if (i + j + 16 < n) dst[i + j + 16] = d * (q1 - 16);
        }
    }
}

// ---- Q6_K : super block(256) = [ql:128B][qh:64B][scales:16B][d:f16]，value = d * sc * (q - 32) ----
void deq_q6_k(const void* src, float* dst, int64_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    const int QK = 256;
    for (int64_t i = 0; i < n; i += QK) {
        const uint8_t* ql = s; s += QK / 2;    // 128 字节
        const uint8_t* qh = s; s += QK / 4;    // 64 字节
        const int8_t* sc = reinterpret_cast<const int8_t*>(s); s += QK / 16;  // 16 字节
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(s)); s += 2;

        for (int nblk = 0; nblk < QK; nblk += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = (ql[l] & 0xF)      | (((qh[l] >> 0) & 3) << 4);
                int q2 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                int q3 = (ql[l] >> 4)       | (((qh[l] >> 4) & 3) << 4);
                int q4 = (ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4);
                if (i + nblk + l < n)      dst[i + nblk + l]      = d * sc[is + 0] * (q1 - 32);
                if (i + nblk + l + 32 < n) dst[i + nblk + l + 32] = d * sc[is + 2] * (q2 - 32);
                if (i + nblk + l + 64 < n) dst[i + nblk + l + 64] = d * sc[is + 4] * (q3 - 32);
                if (i + nblk + l + 96 < n) dst[i + nblk + l + 96] = d * sc[is + 6] * (q4 - 32);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ---- Q4_1 : block = [d:f16][m:f16][qs:16B 打包 32 个 nibble]，value = d * q + m ----
void deq_q4_1(const void* src, float* dst, int64_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    const int bs = 32;
    for (int64_t i = 0; i < n; i += bs) {
        float d = half_to_float(*reinterpret_cast<const uint16_t*>(s)); s += 2;
        float m = half_to_float(*reinterpret_cast<const uint16_t*>(s)); s += 2;
        for (int j = 0; j < bs / 2; ++j) {
            uint8_t byte = s[j];
            if (i + j < n)      dst[i + j]      = d * (byte & 0xF) + m;
            if (i + j + 16 < n) dst[i + j + 16] = d * (byte >> 4) + m;
        }
        s += bs / 2;
    }
}

// ---- Q5_1 : block = [d:f16][m:f16][qh:4B 高1位][qs:16B 低4位]，value = d * q + m ----
// qs[j] 低 nibble = 元素 j，高 nibble = 元素 j+16；qh 第 j 位 = 元素 j 的高 1 位
void deq_q5_1(const void* src, float* dst, int64_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    const int bs = 32;
    for (int64_t i = 0; i < n; i += bs) {
        float d = half_to_float(*reinterpret_cast<const uint16_t*>(s)); s += 2;
        float m = half_to_float(*reinterpret_cast<const uint16_t*>(s)); s += 2;
        const uint8_t* qh = s; s += 4;
        const uint8_t* qs = s; s += 16;
        uint32_t qh_bits = (uint32_t)qh[0] | ((uint32_t)qh[1] << 8) |
                           ((uint32_t)qh[2] << 16) | ((uint32_t)qh[3] << 24);
        for (int j = 0; j < 16; ++j) {
            int hi0 = (qh_bits >> j) & 1;
            int hi1 = (qh_bits >> (j + 16)) & 1;
            int q0 = (qs[j] & 0xF) | (hi0 << 4);
            int q1 = (qs[j] >> 4) | (hi1 << 4);
            if (i + j < n)      dst[i + j]      = d * q0 + m;
            if (i + j + 16 < n) dst[i + j + 16] = d * q1 + m;
        }
    }
}

// ---- Q8_1 : block = [d:f16][s:f16][qs:32 x int8]，value = d * q（s 仅用于点积）----
void deq_q8_1(const void* src, float* dst, int64_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    const int bs = 32;
    for (int64_t i = 0; i < n; i += bs) {
        float d = half_to_float(*reinterpret_cast<const uint16_t*>(s));
        s += 4;  // 跳过 d 与 s 两个 f16
        for (int j = 0; j < bs && i + j < n; ++j) {
            dst[i + j] = d * static_cast<int8_t>(s[j]);
        }
        s += bs;
    }
}

// ---- Q2_K : super block(256) = [scales:16B][qs:64B][d:f16][dmin:f16] ----
// 每字节 scale = [sc:低4位][m:高4位]，value = d*sc*q - dmin*m，q 为 2 bit
void deq_q2_k(const void* src, float* dst, int64_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    const int QK = 256;
    for (int64_t i = 0; i < n; i += QK) {
        const uint8_t* scales = s; s += QK / 16;  // 16 字节
        const uint8_t* q = s;      s += QK / 4;   // 64 字节
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(s)); s += 2;
        const float dmin = half_to_float(*reinterpret_cast<const uint16_t*>(s)); s += 2;

        int is = 0;
        for (int nb = 0; nb < QK; nb += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l) {
                    if (i + nb + j * 32 + l < n)
                        dst[i + nb + j * 32 + l] = dl * ((q[l] >> shift) & 3) - ml;
                }
                sc = scales[is++];
                dl = d * (sc & 0xF);
                ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l) {
                    if (i + nb + j * 32 + 16 + l < n)
                        dst[i + nb + j * 32 + 16 + l] = dl * ((q[l + 16] >> shift) & 3) - ml;
                }
                shift += 2;
            }
            q += 32;
        }
    }
}

// ---- Q3_K : super block(256) = [hmask:32B][qs:64B][scales:12B][d:f16] ----
// 6 bit scale(交错打包) - 32；q 为 2 bit 减去 hmask 提供的高位
void deq_q3_k(const void* src, float* dst, int64_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    const int QK = 256;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    for (int64_t i = 0; i < n; i += QK) {
        const uint8_t* hm = s; s += QK / 8;   // 32 字节
        const uint8_t* q  = s; s += QK / 4;   // 64 字节
        uint32_t aux[4];
        std::memcpy(aux, s, 12);
        s += 12;
        const float d_all = half_to_float(*reinterpret_cast<const uint16_t*>(s)); s += 2;

        // 12 字节交错打包的 6 bit scales 解包（见 llama.cpp dequantize_row_q3_K）
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

        int is = 0;
        uint8_t m = 1;
        for (int nb = 0; nb < QK; nb += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int idx = i + nb + j * 32 + l;
                    if (idx < n)
                        dst[idx] = dl * (((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int idx = i + nb + j * 32 + 16 + l;
                    if (idx < n)
                        dst[idx] = dl * (((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

// ---- Q5_K : super block(256) = [d:f16][dmin:f16][scales:12B][qh:32B][qs:128B] ----
// 低 4 位在 qs，高 1 位在 qh（每 64 元素换一个 bit 位），value = d*sc*q - dmin*m
void deq_q5_k(const void* src, float* dst, int64_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    const int QK = 256;
    for (int64_t i = 0; i < n; i += QK) {
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(s)); s += 2;
        const float dmin = half_to_float(*reinterpret_cast<const uint16_t*>(s)); s += 2;
        const uint8_t* scales = s; s += 12;
        const uint8_t* qh = s; s += QK / 8;   // 32 字节
        const uint8_t* ql = s; s += QK / 2;   // 128 字节

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) {
                int idx = i + j + l;
                if (idx < n)
                    dst[idx] = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                int idx = i + j + 32 + l;
                if (idx < n)
                    dst[idx] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

}  // namespace

int block_size(GgmlType type) {
    switch (type) {
        case GgmlType::F32: case GgmlType::F16: return 1;
        case GgmlType::Q4_0: case GgmlType::Q4_1:
        case GgmlType::Q5_0: case GgmlType::Q5_1:
        case GgmlType::Q8_0: case GgmlType::Q8_1: return 32;
        case GgmlType::Q2_K: case GgmlType::Q3_K: case GgmlType::Q4_K:
        case GgmlType::Q5_K: case GgmlType::Q6_K: return 256;
    }
    return 1;
}

size_t type_size(GgmlType type) {
    switch (type) {
        case GgmlType::F32: return 4;
        case GgmlType::F16: return 2;
        case GgmlType::Q4_0: return 18;
        case GgmlType::Q4_1: return 20;
        case GgmlType::Q5_0: return 22;
        case GgmlType::Q5_1: return 24;
        case GgmlType::Q8_0: return 34;
        case GgmlType::Q8_1: return 36;
        case GgmlType::Q2_K: return 84;
        case GgmlType::Q3_K: return 110;
        case GgmlType::Q4_K: return 144;
        case GgmlType::Q5_K: return 176;
        case GgmlType::Q6_K: return 210;
    }
    return 0;
}

void dequantize_block(const void* src, float* dst, GgmlType type, int64_t n) {
    switch (type) {
        case GgmlType::F32:  deq_f32(src, dst, n); break;
        case GgmlType::F16:  deq_f16(src, dst, n); break;
        case GgmlType::Q8_0: deq_q8_0(src, dst, n); break;
        case GgmlType::Q8_1: deq_q8_1(src, dst, n); break;
        case GgmlType::Q4_0: deq_q4_0(src, dst, n); break;
        case GgmlType::Q4_1: deq_q4_1(src, dst, n); break;
        case GgmlType::Q5_0: deq_q5_0(src, dst, n); break;
        case GgmlType::Q5_1: deq_q5_1(src, dst, n); break;
        case GgmlType::Q2_K: deq_q2_k(src, dst, n); break;
        case GgmlType::Q3_K: deq_q3_k(src, dst, n); break;
        case GgmlType::Q4_K: deq_q4_k(src, dst, n); break;
        case GgmlType::Q5_K: deq_q5_k(src, dst, n); break;
        case GgmlType::Q6_K: deq_q6_k(src, dst, n); break;
        default:
            throw std::runtime_error("unsupported ggml type for dequantize");
    }
}

std::vector<float> Tensor::dequantize() const {
    std::vector<float> out(numel());
    dequantize_block(data, out.data(), type, static_cast<int64_t>(numel()));
    return out;
}

// =========================================================================
// 激活量化 + 整数点积快路径
// =========================================================================

// 将 float 向量量化为 Q8_0（block=32，absmax scale），并预计算每块 q 的 int32 之和
void quantize_row_q8_0(const float* x, QuantizedQ8_0& out, int64_t n) {
    constexpr int QK8 = 32;
    if (n % QK8 != 0) {
        throw std::runtime_error("quantize_row_q8_0: n not divisible by 32");
    }
    const int64_t nb = n / QK8;
    out.n = n;
    out.scales.resize(nb);
    out.qs.resize(n);
    out.bsums.resize(nb);
    for (int64_t b = 0; b < nb; ++b) {
        const float* xb = x + b * QK8;
        float amax = 0.0f;
        for (int j = 0; j < QK8; ++j) {
            float a = std::fabs(xb[j]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        out.scales[b] = d;
        int8_t* qb = out.qs.data() + b * QK8;
        int32_t bsum = 0;
        for (int j = 0; j < QK8; ++j) {
            float v = xb[j] * id;
            int q = (int)std::nearbyint(v);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            qb[j] = (int8_t)q;
            bsum += q;
        }
        out.bsums[b] = bsum;
    }
}

// Q8_0 × Q8_0 vec_dot（标量版，用作 fallback 与数值参考）
// 权重块布局：[d_w:f16][qs_w:32 x int8] -> block_size=32, type_size=34
// 激活块布局：Q8_0_act = [d_x:f32] + [qs_x:32 x int8]（存储于 QuantizedQ8_0 的两个独立数组）
// 单块贡献：d_w * d_x * dot_int32(qs_w, qs_x)
[[maybe_unused]] static float vec_dot_q8_0_q8_0_scalar(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
    constexpr int QK = 32;
    const int64_t nb = n / QK;
    const int8_t* qx_qs = qx.qs.data();
    const float* qx_sc = qx.scales.data();
    float acc = 0.0f;
    for (int64_t b = 0; b < nb; ++b) {
        const float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w));
        const int8_t* qw = reinterpret_cast<const int8_t*>(w + 2);
        const int8_t* qxb = qx_qs + b * QK;
        int32_t sum = 0;
        for (int j = 0; j < QK; ++j) sum += (int32_t)qw[j] * (int32_t)qxb[j];
        acc += d_w * qx_sc[b] * (float)sum;
        w += 34;  // d:f16(2B) + qs:int8(32B)
    }
    return acc;
}

#if defined(__ARM_NEON) && CLLM_HAS_SDOT
// Q8_0 × Q8_0 vec_dot NEON SDOT 版本
// 每块 32 元素 = 2 条 vdotq_s32，一条处理 16 对 int8（4 个 int32 累加通道）
static float vec_dot_q8_0_q8_0_neon(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
    constexpr int QK = 32;
    const int64_t nb = n / QK;
    const int8_t* qx_qs = qx.qs.data();
    const float* qx_sc = qx.scales.data();
    float acc = 0.0f;
    for (int64_t b = 0; b < nb; ++b) {
        const float d_w = half_to_float(*reinterpret_cast<const uint16_t*>(w));
        const int8_t* qw = reinterpret_cast<const int8_t*>(w + 2);
        const int8_t* qxb = qx_qs + b * QK;

        int8x16_t qw_lo = vld1q_s8(qw);
        int8x16_t qw_hi = vld1q_s8(qw + 16);
        int8x16_t qx_lo = vld1q_s8(qxb);
        int8x16_t qx_hi = vld1q_s8(qxb + 16);

        int32x4_t sum_vec = vdupq_n_s32(0);
        sum_vec = vdotq_s32(sum_vec, qw_lo, qx_lo);
        sum_vec = vdotq_s32(sum_vec, qw_hi, qx_hi);
        int32_t sum = vaddvq_s32(sum_vec);

        acc += d_w * qx_sc[b] * (float)sum;
        w += 34;
    }
    return acc;
}
#endif

static inline float vec_dot_q8_0_q8_0(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
#if defined(__ARM_NEON) && CLLM_HAS_SDOT
    return vec_dot_q8_0_q8_0_neon(w, qx, n);
#else
    return vec_dot_q8_0_q8_0_scalar(w, qx, n);
#endif
}

// Q4_0 × Q8_0 vec_dot（标量版，用作 fallback 与数值参考）
// 权重块 = 32 元素 = [d:f16][qs:16B]，value = d*(q-8)，q 为 0-15 的 nibble
// qs[j] 低 nibble = 元素 j（0-15），高 nibble = 元素 j+16（16-31）
// 单块贡献：sub_dot = sum_j (d*(q4[j]-8) * d_x*q_x[j])
//           = d*d_x*sum(q4*q_x) - 8*d*d_x*sum(q_x)
// sum(q_x) 已在 quantize_row_q8_0 预计算 bsums[b]
[[maybe_unused]] static float vec_dot_q4_0_q8_scalar(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
    constexpr int QK = 32;
    const int64_t nb = n / QK;
    const int8_t* qx_qs = qx.qs.data();
    const float* qx_sc = qx.scales.data();
    const int32_t* qx_bs = qx.bsums.data();
    float acc = 0.0f;
    for (int64_t b = 0; b < nb; ++b) {
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;
        const uint8_t* qs = w; w += 16;
        const int8_t* qxb = qx_qs + (size_t)b * QK;
        int32_t dot = 0;
        for (int j = 0; j < 16; ++j) {
            dot += (int32_t)(qs[j] & 0xF) * (int32_t)qxb[j];
            dot += (int32_t)(qs[j] >> 4) * (int32_t)qxb[j + 16];
        }
        acc += d * qx_sc[b] * ((float)dot - 8.0f * (float)qx_bs[b]);
    }
    return acc;
}

#if defined(__ARM_NEON) && CLLM_HAS_SDOT
// Q4_0 × Q8_0 vec_dot NEON SDOT 版本
// 每块 16 字节 qs 拆成低 nibble（元素 0..15）与高 nibble（元素 16..31）
static float vec_dot_q4_0_q8_neon(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
    constexpr int QK = 32;
    const int64_t nb = n / QK;
    const int8_t* qx_qs = qx.qs.data();
    const float* qx_sc = qx.scales.data();
    const int32_t* qx_bs = qx.bsums.data();
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    float acc = 0.0f;
    for (int64_t b = 0; b < nb; ++b) {
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;
        const uint8_t* qs = w; w += 16;
        const uint8x16_t raw = vld1q_u8(qs);
        const int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(raw, mask_lo));  // 元素 0..15
        const int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));      // 元素 16..31
        const int8_t* qxb = qx_qs + (size_t)b * QK;
        const int8x16_t qx0 = vld1q_s8(qxb);
        const int8x16_t qx1 = vld1q_s8(qxb + 16);
        int32x4_t sum = vdupq_n_s32(0);
        sum = vdotq_s32(sum, q4_lo, qx0);
        sum = vdotq_s32(sum, q4_hi, qx1);
        const int32_t dot = vaddvq_s32(sum);
        acc += d * qx_sc[b] * ((float)dot - 8.0f * (float)qx_bs[b]);
    }
    return acc;
}
#endif

static inline float vec_dot_q4_0_q8(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
#if defined(__ARM_NEON) && CLLM_HAS_SDOT
    return vec_dot_q4_0_q8_neon(w, qx, n);
#else
    return vec_dot_q4_0_q8_scalar(w, qx, n);
#endif
}

// Q5_0 × Q8_0 vec_dot（标量版，用作 fallback 与数值参考）
// 权重块 = 32 元素 = [d:f16][qh:4B][qs:16B]，value = d*(q-16)
// q = low4 | (highbit<<4)，qh 为 uint32 小端，第 j 位 = 元素 j 的高 1 位
[[maybe_unused]] static float vec_dot_q5_0_q8_scalar(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
    constexpr int QK = 32;
    const int64_t nb = n / QK;
    const int8_t* qx_qs = qx.qs.data();
    const float* qx_sc = qx.scales.data();
    const int32_t* qx_bs = qx.bsums.data();
    float acc = 0.0f;
    for (int64_t b = 0; b < nb; ++b) {
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;
        const uint8_t* qh = w; w += 4;
        const uint8_t* qs = w; w += 16;
        const uint32_t qh_bits = (uint32_t)qh[0] | ((uint32_t)qh[1] << 8) |
                                 ((uint32_t)qh[2] << 16) | ((uint32_t)qh[3] << 24);
        const int8_t* qxb = qx_qs + (size_t)b * QK;
        int32_t dot = 0;
        for (int j = 0; j < 16; ++j) {
            const int q0 = (qs[j] & 0xF) | (((qh_bits >> j) & 1) << 4);
            const int q1 = (qs[j] >> 4) | (((qh_bits >> (j + 16)) & 1) << 4);
            dot += q0 * (int32_t)qxb[j];
            dot += q1 * (int32_t)qxb[j + 16];
        }
        acc += d * qx_sc[b] * ((float)dot - 16.0f * (float)qx_bs[b]);
    }
    return acc;
}

#if defined(__ARM_NEON) && CLLM_HAS_SDOT
// Q5_0 × Q8_0 vec_dot NEON SDOT 版本
// qh 的 32 个高位 bit 用标量展开为 2 个 16-byte mask（0 或 16），与低 4 位合并
static float vec_dot_q5_0_q8_neon(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
    constexpr int QK = 32;
    const int64_t nb = n / QK;
    const int8_t* qx_qs = qx.qs.data();
    const float* qx_sc = qx.scales.data();
    const int32_t* qx_bs = qx.bsums.data();
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    float acc = 0.0f;
    for (int64_t b = 0; b < nb; ++b) {
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;
        const uint8_t* qh = w; w += 4;
        const uint8_t* qs = w; w += 16;
        const uint32_t qh_bits = (uint32_t)qh[0] | ((uint32_t)qh[1] << 8) |
                                 ((uint32_t)qh[2] << 16) | ((uint32_t)qh[3] << 24);

        const uint8x16_t raw = vld1q_u8(qs);
        const uint8x16_t q_lo = vandq_u8(raw, mask_lo);  // 元素 0..15 低 4 位
        const uint8x16_t q_hi = vshrq_n_u8(raw, 4);      // 元素 16..31 低 4 位

        // 展开 32 个高位 bit -> 2 个 16-byte mask（每个 byte 为 0 或 16）
        uint8_t hi_lo[16], hi_hi[16];
        for (int j = 0; j < 16; ++j) {
            hi_lo[j] = (uint8_t)(((qh_bits >> j) & 1) << 4);
            hi_hi[j] = (uint8_t)(((qh_bits >> (j + 16)) & 1) << 4);
        }
        const int8x16_t q5_lo = vreinterpretq_s8_u8(vorrq_u8(q_lo, vld1q_u8(hi_lo)));
        const int8x16_t q5_hi = vreinterpretq_s8_u8(vorrq_u8(q_hi, vld1q_u8(hi_hi)));

        const int8_t* qxb = qx_qs + (size_t)b * QK;
        const int8x16_t qx0 = vld1q_s8(qxb);
        const int8x16_t qx1 = vld1q_s8(qxb + 16);
        int32x4_t sum = vdupq_n_s32(0);
        sum = vdotq_s32(sum, q5_lo, qx0);
        sum = vdotq_s32(sum, q5_hi, qx1);
        const int32_t dot = vaddvq_s32(sum);
        acc += d * qx_sc[b] * ((float)dot - 16.0f * (float)qx_bs[b]);
    }
    return acc;
}
#endif

static inline float vec_dot_q5_0_q8(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
#if defined(__ARM_NEON) && CLLM_HAS_SDOT
    return vec_dot_q5_0_q8_neon(w, qx, n);
#else
    return vec_dot_q5_0_q8_scalar(w, qx, n);
#endif
}

// Q4_K × Q8_0 vec_dot（标量版，用作 fallback 与数值参考）
// 权重 super-block = 256 元素 = 8 个 32-元素子块：
//   [d:f16][dmin:f16][scales:12B][qs:128B]
// 单元素恢复：w[i] = d*sc_k * q4[i] - dmin*m_k，其中 (sc_k, m_k) 由 get_scale_min_k4 解码
//
// 32-元素子块对激活点积：
//   sub_dot = sum_j (w[j] * x[j])
//           = sum_j ((d*sc*q4[j] - dmin*m) * d_x * q_x[j])
//           = d*sc*d_x * sum(q4*q_x)  -  dmin*m*d_x * sum(q_x)
// sum(q_x) 已在 quantize_row_q8_0 预计算 bsums[b]
[[maybe_unused]] static float vec_dot_q4_k_q8_scalar(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
    constexpr int QK = 256;
    const int64_t n_super = n / QK;
    const int8_t* qx_qs = qx.qs.data();
    const float* qx_sc = qx.scales.data();
    const int32_t* qx_bs = qx.bsums.data();
    float acc = 0.0f;
    for (int64_t s = 0; s < n_super; ++s) {
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;
        const float dmin = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;
        const uint8_t* scales = w; w += 12;
        const uint8_t* q = w;      w += 128;

        // 8 个 32-元素子块，激活块 base 索引 = s*8
        const int64_t x_base = s * 8;
        int is = 0;
        for (int j = 0; j < QK; j += 64) {
            // 子块 is+0：对应 x_base + is + 0，使用 q4 的低 nibble
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, sc, m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            const float d2 = d * sc, m2 = dmin * m;

            // 低 nibble 子块：32 元素
            {
                const int b_lo = (int)(x_base + is + 0);
                const int8_t* qxb = qx_qs + (size_t)b_lo * 32;
                int32_t dot = 0;
                for (int l = 0; l < 32; ++l) {
                    dot += (int32_t)(q[l] & 0xF) * (int32_t)qxb[l];
                }
                acc += d1 * qx_sc[b_lo] * (float)dot;
                acc -= m1 * qx_sc[b_lo] * (float)qx_bs[b_lo];
            }
            // 高 nibble 子块：32 元素
            {
                const int b_hi = (int)(x_base + is + 1);
                const int8_t* qxb = qx_qs + (size_t)b_hi * 32;
                int32_t dot = 0;
                for (int l = 0; l < 32; ++l) {
                    dot += (int32_t)(q[l] >> 4) * (int32_t)qxb[l];
                }
                acc += d2 * qx_sc[b_hi] * (float)dot;
                acc -= m2 * qx_sc[b_hi] * (float)qx_bs[b_hi];
            }
            q += 32;
            is += 2;
        }
    }
    return acc;
}

// --- 导出接口 ---

#if defined(__ARM_NEON) && CLLM_HAS_SDOT
// Q4_K × Q8_0 vec_dot NEON SDOT 版本
// 每子块（32 元素）：从 128B qs 中拆出低 nibble 和高 nibble，各 16 字节
// SDOT 处理：qs_low (16B) · qxb (16B) -> int32x4，再按另一半聚合
static float vec_dot_q4_k_q8_neon(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
    constexpr int QK = 256;
    const int64_t n_super = n / QK;
    const int8_t* qx_qs = qx.qs.data();
    const float* qx_sc = qx.scales.data();
    const int32_t* qx_bs = qx.bsums.data();
    const int8x16_t mask_lo = vdupq_n_s8(0x0F);
    float acc = 0.0f;
    for (int64_t s = 0; s < n_super; ++s) {
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;
        const float dmin = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;
        const uint8_t* scales = w; w += 12;
        const uint8_t* q = w;      w += 128;

        const int64_t x_base = s * 8;
        int is = 0;
        for (int j = 0; j < QK; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, sc, m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            const float d2 = d * sc, m2 = dmin * m;

            // 读取 32 字节 qs，拆出低 nibble (q4 低半块) 和高 nibble (q4 高半块)
            // q4 值域 0-15，最高位 bit 一定为 0，故可直接当 int8 用于 SDOT
            uint8x16_t raw0 = vld1q_u8(q);          // 元素 0..15 的 packed
            uint8x16_t raw1 = vld1q_u8(q + 16);     // 元素 16..31 的 packed
            int8x16_t q4_lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, vreinterpretq_u8_s8(mask_lo)));
            int8x16_t q4_lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, vreinterpretq_u8_s8(mask_lo)));
            int8x16_t q4_hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));  // 无符号右移保留 0-15
            int8x16_t q4_hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));

            // 低 nibble 子块
            {
                const int b_lo = (int)(x_base + is + 0);
                const int8_t* qxb = qx_qs + (size_t)b_lo * 32;
                int8x16_t qx0 = vld1q_s8(qxb);
                int8x16_t qx1 = vld1q_s8(qxb + 16);
                int32x4_t sum = vdupq_n_s32(0);
                sum = vdotq_s32(sum, q4_lo0, qx0);
                sum = vdotq_s32(sum, q4_lo1, qx1);
                int32_t dot = vaddvq_s32(sum);
                acc += d1 * qx_sc[b_lo] * (float)dot;
                acc -= m1 * qx_sc[b_lo] * (float)qx_bs[b_lo];
            }
            // 高 nibble 子块
            {
                const int b_hi = (int)(x_base + is + 1);
                const int8_t* qxb = qx_qs + (size_t)b_hi * 32;
                int8x16_t qx0 = vld1q_s8(qxb);
                int8x16_t qx1 = vld1q_s8(qxb + 16);
                int32x4_t sum = vdupq_n_s32(0);
                sum = vdotq_s32(sum, q4_hi0, qx0);
                sum = vdotq_s32(sum, q4_hi1, qx1);
                int32_t dot = vaddvq_s32(sum);
                acc += d2 * qx_sc[b_hi] * (float)dot;
                acc -= m2 * qx_sc[b_hi] * (float)qx_bs[b_hi];
            }
            q += 32;
            is += 2;
        }
    }
    return acc;
}
#endif

static inline float vec_dot_q4_k_q8(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
#if defined(__ARM_NEON) && CLLM_HAS_SDOT
    return vec_dot_q4_k_q8_neon(w, qx, n);
#else
    return vec_dot_q4_k_q8_scalar(w, qx, n);
#endif
}

// Q5_K × Q8_0 vec_dot（标量版，用作 fallback 与数值参考）
// 权重 super-block = 256 元素 = [d:f16][dmin:f16][scales:12B][qh:32B][qs:128B]
// 低 4 位在 qs（128B = 256 nibble），高 1 位在 qh（32B = 256 bit，交错布局）
// 子块 k（k=0..7，32 元素）：
//   低 4 位来自 qs 段 [(k/2)*32 .. (k/2)*32+31]，k 偶数取低 nibble、k 奇数取高 nibble
//   高 1 位来自 qh[l] 的 bit k（l=0..31）
//   scale/m 由 get_scale_min_k4(k, scales, sc, m) 解码
// value = d*sc*q - dmin*m，q = 0-31
// sub_dot = d*sc*d_x*sum(q5*q_x) - dmin*m*d_x*sum(q_x)
[[maybe_unused]] static float vec_dot_q5_k_q8_scalar(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
    constexpr int QK = 256;
    const int64_t n_super = n / QK;
    const int8_t* qx_qs = qx.qs.data();
    const float* qx_sc = qx.scales.data();
    const int32_t* qx_bs = qx.bsums.data();
    float acc = 0.0f;
    for (int64_t s = 0; s < n_super; ++s) {
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;
        const float dmin = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;
        const uint8_t* scales = w; w += 12;
        const uint8_t* qh = w; w += 32;
        const uint8_t* ql = w; w += 128;

        const int64_t x_base = s * 8;
        for (int k = 0; k < 8; ++k) {
            uint8_t sc, m;
            get_scale_min_k4(k, scales, sc, m);
            const float dk = d * sc, mk = dmin * m;
            const uint8_t* qlg = ql + (size_t)(k / 2) * 32;
            const bool use_hi = (k & 1);

            const int b = (int)(x_base + k);
            const int8_t* qxb = qx_qs + (size_t)b * 32;
            int32_t dot = 0;
            for (int l = 0; l < 32; ++l) {
                const int low4 = use_hi ? (qlg[l] >> 4) : (qlg[l] & 0xF);
                const int q = low4 | (((qh[l] >> k) & 1) << 4);
                dot += q * (int32_t)qxb[l];
            }
            acc += dk * qx_sc[b] * (float)dot;
            acc -= mk * qx_sc[b] * (float)qx_bs[b];
        }
    }
    return acc;
}

#if defined(__ARM_NEON) && CLLM_HAS_SDOT
// Q5_K × Q8_0 vec_dot NEON SDOT 版本
// 8 个子块用编译期常量 K 手动展开，避免 vshrq_n_u8 的立即数限制
static float vec_dot_q5_k_q8_neon(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
    constexpr int QK = 256;
    const int64_t n_super = n / QK;
    const int8_t* qx_qs = qx.qs.data();
    const float* qx_sc = qx.scales.data();
    const int32_t* qx_bs = qx.bsums.data();
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    const uint8x16_t vone = vdupq_n_u8(1);
    float acc = 0.0f;
    for (int64_t s = 0; s < n_super; ++s) {
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;
        const float dmin = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;
        const uint8_t* scales = w; w += 12;
        const uint8_t* qh = w; w += 32;
        const uint8_t* ql = w; w += 128;

        const int64_t x_base = s * 8;
        // qh 32 字节一次读入：子块 k 用 qh[l] 的 bit k
        const uint8x16_t qh0 = vld1q_u8(qh);        // qh[0..15]
        const uint8x16_t qh1 = vld1q_u8(qh + 16);   // qh[16..31]

        // 处理单个子块 k（编译期常量）
        auto sub = [&]<int K>(const uint8x16_t& qh0, const uint8x16_t& qh1) {
            uint8_t sc, m;
            get_scale_min_k4(K, scales, sc, m);
            const float dk = d * sc, mk = dmin * m;

            const uint8_t* qlg = ql + (size_t)(K / 2) * 32;
            const uint8x16_t raw0 = vld1q_u8(qlg);
            const uint8x16_t raw1 = vld1q_u8(qlg + 16);

            // 低 4 位：K 偶数取低 nibble，K 奇数取高 nibble
            int8x16_t lo0, lo1;
            if constexpr (K & 1) {
                lo0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
                lo1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
            } else {
                lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_lo));
                lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_lo));
            }

            // 高 1 位：qh 的 bit K
            uint8x16_t h0, h1;
            if constexpr (K == 0) {
                h0 = qh0;
                h1 = qh1;
            } else {
                h0 = vshrq_n_u8(qh0, K);
                h1 = vshrq_n_u8(qh1, K);
            }
            h0 = vshlq_n_u8(vandq_u8(h0, vone), 4);
            h1 = vshlq_n_u8(vandq_u8(h1, vone), 4);

            const int8x16_t q5_0 = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(lo0), h0));
            const int8x16_t q5_1 = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(lo1), h1));

            const int b = (int)(x_base + K);
            const int8_t* qxb = qx_qs + (size_t)b * 32;
            const int8x16_t qx0 = vld1q_s8(qxb);
            const int8x16_t qx1 = vld1q_s8(qxb + 16);
            int32x4_t sum = vdupq_n_s32(0);
            sum = vdotq_s32(sum, q5_0, qx0);
            sum = vdotq_s32(sum, q5_1, qx1);
            const int32_t dot = vaddvq_s32(sum);
            acc += dk * qx_sc[b] * (float)dot;
            acc -= mk * qx_sc[b] * (float)qx_bs[b];
        };

        sub.template operator()<0>(qh0, qh1);
        sub.template operator()<1>(qh0, qh1);
        sub.template operator()<2>(qh0, qh1);
        sub.template operator()<3>(qh0, qh1);
        sub.template operator()<4>(qh0, qh1);
        sub.template operator()<5>(qh0, qh1);
        sub.template operator()<6>(qh0, qh1);
        sub.template operator()<7>(qh0, qh1);
    }
    return acc;
}
#endif

static inline float vec_dot_q5_k_q8(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
#if defined(__ARM_NEON) && CLLM_HAS_SDOT
    return vec_dot_q5_k_q8_neon(w, qx, n);
#else
    return vec_dot_q5_k_q8_scalar(w, qx, n);
#endif
}

// Q6_K × Q8_0 vec_dot（标量版，用作 fallback 与数值参考）
// 权重 super-block = 256 元素 = [ql:128B][qh:64B][scales:int8[16]B][d:f16]
// 每元素 6-bit：ql 存低 4 位，qh 存高 2 位
// 每 16 元素共用一个 int8 scale（有符号）：w[i] = d * sc[i/16] * (q6[i] - 32)
//
// 激活用 Q8_0（block=32）。一个激活块覆盖 2 个 scale 组（元素 i..i+15 和 i+16..i+31）。
// 32-元素子块点积：
//   sub_dot = sum_j (w[j] * x[j])
//           = d*d_x * ( sc_a*sum_{j=0..15}(q6*q_x) + sc_b*sum_{j=16..31}(q6*q_x) )
//            - 32*d*d_x * (sc_a*sum_{j=0..15}(q_x) + sc_b*sum_{j=16..31}(q_x))
// 后一项需要激活 "每 16 元素" 的 sum（bsums 是 32 粒度），需在循环内即时累加。
[[maybe_unused]] static float vec_dot_q6_k_q8_scalar(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
    constexpr int QK = 256;
    const int64_t n_super = n / QK;
    const int8_t* qx_qs = qx.qs.data();
    const float* qx_sc = qx.scales.data();
    float acc = 0.0f;
    for (int64_t s = 0; s < n_super; ++s) {
        const uint8_t* ql = w; w += 128;
        const uint8_t* qh = w; w += 64;
        const int8_t* sc = reinterpret_cast<const int8_t*>(w); w += 16;
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;

        const int64_t x_base = s * 8;  // 该 super-block 的第一个激活块索引

        // 按 128 元素一组迭代（共 2 组），每组内按 32 元素子块索引 l
        for (int grp = 0; grp < 2; ++grp) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                // 每 l 对应 4 个元素，分别位于 4 个 32-元素子块（相对 base）：
                //   子块 0: 元素 l           (16 元素 scale sc[is])
                //   子块 1: 元素 l+32        (sc[is+2])
                //   子块 2: 元素 l+64 (-->l) (sc[is+4])
                //   子块 3: 元素 l+96        (sc[is+6])
                int q1 = (ql[l] & 0xF)      | (((qh[l] >> 0) & 3) << 4);
                int q2 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                int q3 = (ql[l] >> 4)       | (((qh[l] >> 4) & 3) << 4);
                int q4 = (ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4);

                // 子块 0..3 分别对应激活块 x_base + grp*4 + 0..3
                // 子块内第 l 元素（当 l < 16 时在前半，否则在后半）
                int block_within_32 = l;   // 0..31
                int b0 = (int)(x_base + grp * 4 + 0);
                int b1 = (int)(x_base + grp * 4 + 1);
                int b2 = (int)(x_base + grp * 4 + 2);
                int b3 = (int)(x_base + grp * 4 + 3);
                const int8_t qx0 = qx_qs[b0 * 32 + block_within_32];
                const int8_t qx1 = qx_qs[b1 * 32 + block_within_32];
                const int8_t qx2 = qx_qs[b2 * 32 + block_within_32];
                const int8_t qx3 = qx_qs[b3 * 32 + block_within_32];

                // 单元素贡献 = d * sc * d_x * ((q6 - 32) * q_x)
                acc += d * sc[is + 0] * qx_sc[b0] * (float)((q1 - 32) * qx0);
                acc += d * sc[is + 2] * qx_sc[b1] * (float)((q2 - 32) * qx1);
                acc += d * sc[is + 4] * qx_sc[b2] * (float)((q3 - 32) * qx2);
                acc += d * sc[is + 6] * qx_sc[b3] * (float)((q4 - 32) * qx3);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return acc;
}

#if defined(__ARM_NEON) && CLLM_HAS_SDOT
// Q6_K × Q8_0 vec_dot NEON SDOT 版本
// 布局回顾（每 super-block 256 元素）：
//   ql:128B（低4位） qh:64B（高2位） scales:int8[16]（每16元素一个） d:f16
// 每 128 元素一组，组内 4 个 32-元素子块：
//   子块0: 元素 0..31   低4位=ql[0..31] 低nibble, 高2位=qh[0..31] bit0-1
//   子块1: 元素 32..63  低4位=ql[32..63]低nibble, 高2位=qh[0..31] bit2-3
//   子块2: 元素 64..95  低4位=ql[0..31] 高nibble, 高2位=qh[0..31] bit4-5
//   子块3: 元素 96..127 低4位=ql[32..63]高nibble, 高2位=qh[0..31] bit6-7
// q6 = low4 | (high2 << 4)（0..63），减 32 得 -32..31 的 int8
// 点积：sub = d * d_x * (sc_a * dot(q6[0..15], qx[0..15]) + sc_b * dot(q6[16..31], qx[16..31]))
static float vec_dot_q6_k_q8_neon(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
    constexpr int QK = 256;
    const int64_t n_super = n / QK;
    const int8_t* qx_qs = qx.qs.data();
    const float* qx_sc = qx.scales.data();
    const int8x16_t v32 = vdupq_n_s8(32);
    const uint8x16_t v0x0F = vdupq_n_u8(0x0F);
    const uint8x16_t v0x03 = vdupq_n_u8(0x03);

    float acc = 0.0f;
    for (int64_t s = 0; s < n_super; ++s) {
        const uint8_t* ql = w; w += 128;
        const uint8_t* qh = w; w += 64;
        const int8_t* sc = reinterpret_cast<const int8_t*>(w); w += 16;
        const float d = half_to_float(*reinterpret_cast<const uint16_t*>(w)); w += 2;

        const int64_t x_base = s * 8;

        for (int grp = 0; grp < 2; ++grp) {
            const uint8_t* ql_g = ql + (size_t)grp * 64;
            const uint8_t* qh_g = qh + (size_t)grp * 32;
            const int8_t* sc_g = sc + grp * 8;

            const uint8x16_t ql_lo0 = vld1q_u8(ql_g);       // ql[0..15]
            const uint8x16_t ql_lo1 = vld1q_u8(ql_g + 16);  // ql[16..31]
            const uint8x16_t ql_hi0 = vld1q_u8(ql_g + 32);  // ql[32..47]
            const uint8x16_t ql_hi1 = vld1q_u8(ql_g + 48);  // ql[48..63]
            const uint8x16_t qh0 = vld1q_u8(qh_g);          // qh[0..15]
            const uint8x16_t qh1 = vld1q_u8(qh_g + 16);     // qh[16..31]

            // 解包一个 32 元素子块并累加点积（SHIFT 必须为编译期常量）
            auto sub = [&]<int SHIFT>(const uint8x16_t& ql0, const uint8x16_t& ql1,
                                      bool low_in_hi, int sub_idx) {
                const uint8x16_t lo0 = low_in_hi ? vshrq_n_u8(ql0, 4) : vandq_u8(ql0, v0x0F);
                const uint8x16_t lo1 = low_in_hi ? vshrq_n_u8(ql1, 4) : vandq_u8(ql1, v0x0F);
                uint8x16_t qh_s0, qh_s1;
                if constexpr (SHIFT == 0) {
                    qh_s0 = qh0;
                    qh_s1 = qh1;
                } else {
                    qh_s0 = vshrq_n_u8(qh0, SHIFT);
                    qh_s1 = vshrq_n_u8(qh1, SHIFT);
                }
                const uint8x16_t hi0 = vshlq_n_u8(vandq_u8(qh_s0, v0x03), 4);
                const uint8x16_t hi1 = vshlq_n_u8(vandq_u8(qh_s1, v0x03), 4);
                const int8x16_t q6_0 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo0, hi0)), v32);
                const int8x16_t q6_1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo1, hi1)), v32);

                const int b = (int)(x_base + grp * 4 + sub_idx);
                const int8_t* qxb = qx_qs + (size_t)b * 32;
                const int8x16_t qx0 = vld1q_s8(qxb);
                const int8x16_t qx1 = vld1q_s8(qxb + 16);
                const int32_t dot_a = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q6_0, qx0));
                const int32_t dot_b = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q6_1, qx1));

                const float sc_a = (float)sc_g[sub_idx * 2 + 0];
                const float sc_b = (float)sc_g[sub_idx * 2 + 1];
                acc += d * qx_sc[b] * (sc_a * (float)dot_a + sc_b * (float)dot_b);
            };

            sub.template operator()<0>(ql_lo0, ql_lo1, false, 0);
            sub.template operator()<2>(ql_hi0, ql_hi1, false, 1);
            sub.template operator()<4>(ql_lo0, ql_lo1, true,  2);
            sub.template operator()<6>(ql_hi0, ql_hi1, true,  3);
        }
    }
    return acc;
}
#endif

static inline float vec_dot_q6_k_q8(const uint8_t* w, const QuantizedQ8_0& qx, int64_t n) {
#if defined(__ARM_NEON) && CLLM_HAS_SDOT
    return vec_dot_q6_k_q8_neon(w, qx, n);
#else
    return vec_dot_q6_k_q8_scalar(w, qx, n);
#endif
}

bool has_vec_dot(GgmlType type) {
    switch (type) {
        case GgmlType::Q4_0: return true;
        case GgmlType::Q5_0: return true;
        case GgmlType::Q8_0: return true;
        case GgmlType::Q4_K: return true;
        case GgmlType::Q5_K: return true;
        case GgmlType::Q6_K: return true;
        default: return false;
    }
}

float vec_dot_row(const void* w_row, GgmlType w_type,
                  const QuantizedQ8_0& qx, int64_t n) {
    const uint8_t* w = static_cast<const uint8_t*>(w_row);
    switch (w_type) {
        case GgmlType::Q4_0: return vec_dot_q4_0_q8(w, qx, n);
        case GgmlType::Q5_0: return vec_dot_q5_0_q8(w, qx, n);
        case GgmlType::Q8_0: return vec_dot_q8_0_q8_0(w, qx, n);
        case GgmlType::Q4_K: return vec_dot_q4_k_q8(w, qx, n);
        case GgmlType::Q5_K: return vec_dot_q5_k_q8(w, qx, n);
        case GgmlType::Q6_K: return vec_dot_q6_k_q8(w, qx, n);
        default:
            throw std::runtime_error("vec_dot_row: no kernel for this type");
    }
}

void matmul_quant(const float* x, const void* w, GgmlType type,
                  const float* bias, int in_dim, int out_dim, float* y,
                  ThreadPool* pool) {
    const int bs = block_size(type);
    if (bs <= 0 || in_dim % bs != 0) {
        throw std::runtime_error("matmul_quant: in_dim not divisible by block size");
    }
    const size_t row_bytes = (size_t)(in_dim / bs) * type_size(type);
    const uint8_t* base = static_cast<const uint8_t*>(w);

    // ---- 快路径：激活量化 + 整数点积（针对已注册 vec_dot 的权重类型）----
    if (has_vec_dot(type)) {
        QuantizedQ8_0 qx;
        quantize_row_q8_0(x, qx, in_dim);
        auto row_fast = [&](int j) {
            const void* rp = base + (size_t)j * row_bytes;
            float acc = vec_dot_row(rp, type, qx, in_dim);
            if (bias) acc += bias[j];
            y[j] = acc;
        };
        if (pool && out_dim >= 256) {
            pool->parallel_for(0, (size_t)out_dim, [&](size_t b, size_t e) {
                for (size_t j = b; j < e; ++j) row_fast((int)j);
            });
        } else {
            for (int j = 0; j < out_dim; ++j) row_fast(j);
        }
        return;
    }

    // ---- 兜底路径：按行反量化为 float 后做 float 点积 ----
    // 分块融合：按 cache 友好的粒度反量化并立即累加，避免整行 float 缓冲的完整往返。
    // 块长对齐到该类型的 block size。
    const int CHUNK = 512;
    const int chunk_len = (CHUNK / bs) * bs;
    auto row_dot = [&](std::vector<float>& buf, int j) {
        const uint8_t* rp = base + (size_t)j * row_bytes;
        float acc = bias ? bias[j] : 0.0f;
        for (int off = 0; off < in_dim; off += chunk_len) {
            const int len = std::min(chunk_len, in_dim - off);
            dequantize_block(rp + (size_t)(off / bs) * type_size(type), buf.data(), type, len);
            const float* xp = x + off;
            for (int i = 0; i < len; ++i) acc += xp[i] * buf[i];
        }
        y[j] = acc;
    };

    // 大矩阵走线程池并行（每个任务块复用自己的行缓冲）
    if (pool && out_dim >= 256) {
        pool->parallel_for(0, (size_t)out_dim, [&](size_t b, size_t e) {
            std::vector<float> buf(chunk_len);
            for (size_t j = b; j < e; ++j) row_dot(buf, (int)j);
        });
    } else {
        std::vector<float> buf(chunk_len);
        for (int j = 0; j < out_dim; ++j) row_dot(buf, j);
    }
}

}  // namespace cllm
