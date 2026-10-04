#include "cllm/core/quant.hpp"
#include "cllm/core/tensor.hpp"

#include <cstring>
#include <stdexcept>

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
void deq_q4_0(const void* src, float* dst, int64_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    const int bs = 32;
    for (int64_t i = 0; i < n; i += bs) {
        float d = half_to_float(*reinterpret_cast<const uint16_t*>(s));
        s += 2;
        for (int j = 0; j < bs / 2 && i + j * 2 < n; ++j) {
            uint8_t byte = s[j];
            int q0 = byte & 0xF;
            int q1 = byte >> 4;
            if (i + j * 2 < n)     dst[i + j * 2]     = d * (q0 - 8);
            if (i + j * 2 + 1 < n) dst[i + j * 2 + 1] = d * (q1 - 8);
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
        case GgmlType::Q4_0: deq_q4_0(src, dst, n); break;
        case GgmlType::Q5_0: deq_q5_0(src, dst, n); break;
        case GgmlType::Q4_K: deq_q4_k(src, dst, n); break;
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

}  // namespace cllm
