// 量化反量化与量化矩阵乘内核的正确性测试
// 构造已知量化块（d=1.0 等）并比对手算期望值，覆盖全部 13 种 GGML 类型。
#include "cllm/core/quant.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cllm;

namespace {

int g_fail = 0;
int g_pass = 0;

#define CHECK_NEAR(actual, expected, msg)                                        \
    do {                                                                         \
        double a_ = (actual), e_ = (expected);                                   \
        if (std::fabs(a_ - e_) > 1e-4) {                                         \
            std::printf("  FAIL %s: got %.6f, want %.6f\n", msg, a_, e_);        \
            ++g_fail;                                                            \
        } else {                                                                 \
            ++g_pass;                                                            \
        }                                                                        \
    } while (0)

// f16 常量（IEEE754 half）
constexpr uint16_t H0  = 0x0000;  // 0.0
constexpr uint16_t H1  = 0x3C00;  // 1.0
constexpr uint16_t H05 = 0x3800;  // 0.5

inline void put_f16(uint8_t* p, uint16_t h) { p[0] = (uint8_t)(h & 0xFF); p[1] = (uint8_t)(h >> 8); }

std::vector<float> deq(GgmlType t, const std::vector<uint8_t>& bytes, int64_t n) {
    std::vector<float> out(n);
    dequantize_block(bytes.data(), out.data(), t, n);
    return out;
}

// ---- F32 / F16 ----
void test_f32_f16() {
    std::printf("[F32/F16]\n");
    float f[4] = {1.5f, -2.25f, 0.0f, 3.0f};
    auto y = deq(GgmlType::F32, std::vector<uint8_t>((uint8_t*)f, (uint8_t*)f + sizeof(f)), 4);
    CHECK_NEAR(y[0], 1.5, "F32[0]"); CHECK_NEAR(y[1], -2.25, "F32[1]");

    std::vector<uint8_t> b(4 * 2);
    put_f16(&b[0], H1); put_f16(&b[2], H05);
    put_f16(&b[4], H0); put_f16(&b[6], 0xC000);  // -2.0
    auto y2 = deq(GgmlType::F16, b, 4);
    CHECK_NEAR(y2[0], 1.0, "F16[0]"); CHECK_NEAR(y2[1], 0.5, "F16[1]");
    CHECK_NEAR(y2[2], 0.0, "F16[2]"); CHECK_NEAR(y2[3], -2.0, "F16[3]");
}

// ---- Q4_0 : value = d*(q-8) ----
void test_q4_0() {
    std::printf("[Q4_0]\n");
    std::vector<uint8_t> b(18, 0);
    put_f16(&b[0], H1);
    b[2] = 0x9A;  // low nibble=0xA -> elem0=2 ; high nibble=0x9 -> elem16=1
    auto y = deq(GgmlType::Q4_0, b, 32);
    CHECK_NEAR(y[0], 2.0, "Q4_0[0]");
    CHECK_NEAR(y[16], 1.0, "Q4_0[16]");
    CHECK_NEAR(y[1], -8.0, "Q4_0[1]");
}

// ---- Q4_1 : value = d*q + m ----
void test_q4_1() {
    std::printf("[Q4_1]\n");
    std::vector<uint8_t> b(20, 0);
    put_f16(&b[0], H1);   // d = 1.0
    put_f16(&b[2], H05);  // m = 0.5
    b[4] = 0x21;          // low=1 -> elem0=1.5 ; high=2 -> elem16=2.5
    auto y = deq(GgmlType::Q4_1, b, 32);
    CHECK_NEAR(y[0], 1.5, "Q4_1[0]");
    CHECK_NEAR(y[16], 2.5, "Q4_1[16]");
    CHECK_NEAR(y[1], 0.5, "Q4_1[1]");
}

// ---- Q5_0 : value = d*(q-16)，q = low4 | (highbit<<4) ----
void test_q5_0() {
    std::printf("[Q5_0]\n");
    std::vector<uint8_t> b(22, 0);
    put_f16(&b[0], H1);
    b[2] = 0x01;  // qh 第 0 位置位 -> elem0 的高位 = 1
    b[6] = 0x21;  // qs[0]: low=1 -> elem0 ; high=2 -> elem16
    auto y = deq(GgmlType::Q5_0, b, 32);
    CHECK_NEAR(y[0], 1.0, "Q5_0[0]");    // (1|16)-16 = 1
    CHECK_NEAR(y[16], -14.0, "Q5_0[16]"); // 2-16 = -14
}

// ---- Q5_1 : value = d*q + m，q = low4 | (highbit<<4) ----
void test_q5_1() {
    std::printf("[Q5_1]\n");
    std::vector<uint8_t> b(24, 0);
    put_f16(&b[0], H1);   // d
    put_f16(&b[2], H0);   // m = 0
    b[4] = 0x01;          // qh 第 0 位置位
    b[8] = 0x21;          // qs[0]
    auto y = deq(GgmlType::Q5_1, b, 32);
    CHECK_NEAR(y[0], 17.0, "Q5_1[0]");  // (1|16)
    CHECK_NEAR(y[16], 2.0, "Q5_1[16]"); // 2
}

// ---- Q8_0 : value = d*q ----
void test_q8_0() {
    std::printf("[Q8_0]\n");
    std::vector<uint8_t> b(34, 0);
    put_f16(&b[0], H1);
    b[2] = 0xFF;  // -1
    b[3] = 0x02;  // 2
    auto y = deq(GgmlType::Q8_0, b, 32);
    CHECK_NEAR(y[0], -1.0, "Q8_0[0]");
    CHECK_NEAR(y[1], 2.0, "Q8_0[1]");
}

// ---- Q8_1 : value = d*q（s 仅用于点积，不参与反量化）----
void test_q8_1() {
    std::printf("[Q8_1]\n");
    std::vector<uint8_t> b(36, 0);
    put_f16(&b[0], H1);   // d
    put_f16(&b[2], H0);   // s（忽略）
    b[4] = 0xFF;          // -1
    b[5] = 0x02;          // 2
    auto y = deq(GgmlType::Q8_1, b, 32);
    CHECK_NEAR(y[0], -1.0, "Q8_1[0]");
    CHECK_NEAR(y[1], 2.0, "Q8_1[1]");
}

// ---- Q2_K : value = d*(sc&0xF)*q - dmin*(sc>>4) ----
void test_q2_k() {
    std::printf("[Q2_K]\n");
    std::vector<uint8_t> b(84, 0);
    // scales[0..15] 全设 0x01 -> sc=1, m=0
    for (int i = 0; i < 16; ++i) b[i] = 0x01;
    // qs[0] = 0x03 -> elem0 = (0x03>>0)&3 = 3 ；elem32(shift2) = (0x03>>2)&3 = 0
    b[16 + 0] = 0x03;
    // qs[16] 用于 +16 段（j=0, shift=0）-> elem16 = 3
    b[16 + 16] = 0x03;
    put_f16(&b[80], H1);  // d = 1.0
    put_f16(&b[82], H0);  // dmin = 0
    auto y = deq(GgmlType::Q2_K, b, 256);
    CHECK_NEAR(y[0], 3.0, "Q2_K[0]");
    CHECK_NEAR(y[16], 3.0, "Q2_K[16]");
    CHECK_NEAR(y[32], 0.0, "Q2_K[32]");
    CHECK_NEAR(y[128], 0.0, "Q2_K[128]");
}

// ---- Q3_K : value = d*(scale-32)*q'，q' = q2bit - (hmask 高位缺失则 -4) ----
void test_q3_k() {
    std::printf("[Q3_K]\n");
    // 构造 12 字节 scale，使其解包后 16 个 scale 均为 33（(33-32)=1）
    // 由 llama.cpp 解包公式反推：输入字节 = 0x11 x8 + 0xAA x4
    std::vector<uint8_t> b(110, 0);
    for (int i = 0; i < 8; ++i) b[96 + i] = 0x11;   // scales[0..7] 位置（hmask32+qs64=96 起）
    for (int i = 8; i < 12; ++i) b[96 + i] = 0xAA;  // scales[8..11]
    // hmask = 0 -> 高位缺失 -> q' = 0 - 4 = -4 ；qs = 0
    put_f16(&b[108], H1);  // d = 1.0
    auto y = deq(GgmlType::Q3_K, b, 256);
    CHECK_NEAR(y[0], -4.0, "Q3_K[0]");
    CHECK_NEAR(y[255], -4.0, "Q3_K[255]");

    // hmask 全 0xFF -> 高位存在 -> q' = 0 -> 输出全 0
    for (int i = 0; i < 32; ++i) b[i] = 0xFF;
    auto y2 = deq(GgmlType::Q3_K, b, 256);
    CHECK_NEAR(y2[0], 0.0, "Q3_K(h0)[0]");
    CHECK_NEAR(y2[255], 0.0, "Q3_K(h0)[255]");
}

// ---- Q4_K : value = d*sc*q - dmin*m ----
void test_q4_k() {
    std::printf("[Q4_K]\n");
    std::vector<uint8_t> b(144, 0);
    // scales 使 8 组 sc=1, m=0：{0x01 x4, 0x00 x4, 0x01 x4}
    uint8_t sc[12] = {1,1,1,1, 0,0,0,0, 1,1,1,1};
    for (int i = 0; i < 12; ++i) b[4 + i] = sc[i];
    b[16 + 0] = 0x21;  // qs[0]: low=1 -> elem0 ; high=2 -> elem32
    put_f16(&b[0], H1);   // d
    put_f16(&b[2], H0);   // dmin
    auto y = deq(GgmlType::Q4_K, b, 256);
    CHECK_NEAR(y[0], 1.0, "Q4_K[0]");
    CHECK_NEAR(y[32], 2.0, "Q4_K[32]");
    CHECK_NEAR(y[1], 0.0, "Q4_K[1]");
}

// ---- Q5_K : value = d*sc*(low4|highbit) - dmin*m ----
void test_q5_k() {
    std::printf("[Q5_K]\n");
    std::vector<uint8_t> b(176, 0);
    uint8_t sc[12] = {1,1,1,1, 0,0,0,0, 1,1,1,1};
    for (int i = 0; i < 12; ++i) b[4 + i] = sc[i];
    // qh = 0 -> 高位 0 ；qs[0]=0x21 -> elem0=1, elem32=2
    b[16 + 32 + 0] = 0x21;  // qs 区（d+dmin(4) + scales(12) + qh(32) = 48 起）
    put_f16(&b[0], H1);     // d
    put_f16(&b[2], H0);     // dmin
    auto y = deq(GgmlType::Q5_K, b, 256);
    CHECK_NEAR(y[0], 1.0, "Q5_K[0]");
    CHECK_NEAR(y[32], 2.0, "Q5_K[32]");

    // 置 qh 第 0 位（u1=1 作用于 elem0）-> elem0 = 1|16 = 17
    b[16 + 0] = 0x01;
    auto y2 = deq(GgmlType::Q5_K, b, 256);
    CHECK_NEAR(y2[0], 17.0, "Q5_K(qh)[0]");
}

// ---- Q6_K : value = d*sc*(q-32) ----
void test_q6_k() {
    std::printf("[Q6_K]\n");
    std::vector<uint8_t> b(210, 0);
    for (int i = 0; i < 16; ++i) b[128 + 64 + i] = 0x01;  // 全部 scale = 1
    b[0] = 0x21;             // ql[0]: low=1 -> elem0(q1) ; high=2 -> elem64(q3)
    put_f16(&b[208], H1);    // d = 1.0
    auto y = deq(GgmlType::Q6_K, b, 256);
    CHECK_NEAR(y[0], -31.0, "Q6_K[0]");    // 1*(1-32)
    CHECK_NEAR(y[64], -30.0, "Q6_K[64]");  // 1*(2-32)
}

// ---- 未支持类型应抛异常 ----
void test_unsupported() {
    std::printf("[unsupported]\n");
    // 目前 13 种全部实现；用一个超出枚举的值验证仍会抛异常
    std::vector<uint8_t> b(64, 0);
    std::vector<float> out(32);
    bool threw = false;
    try {
        dequantize_block(b.data(), out.data(), static_cast<GgmlType>(999), 32);
    } catch (const std::exception&) {
        threw = true;
    }
    if (threw) ++g_pass; else { std::printf("  FAIL expected throw for unknown type\n"); ++g_fail; }
}

// ---- matmul_quant：与「反量化 + float 点积」参考实现逐元素比对 ----
void test_matmul_quant() {
    std::printf("[matmul_quant]\n");
    const int out_dim = 40, in_dim = 64;  // Q8_0 block=32 -> 64/32=2 块
    // 构造量化权重：d=1.0，qs 用可预测的值
    const int bs = 32;
    const size_t row_bytes = (in_dim / bs) * type_size(GgmlType::Q8_0);
    std::vector<uint8_t> w(row_bytes * out_dim, 0);
    for (int j = 0; j < out_dim; ++j) {
        uint8_t* row = &w[(size_t)j * row_bytes];
        for (int blk = 0; blk < in_dim / bs; ++blk) {
            uint8_t* p = row + (size_t)blk * type_size(GgmlType::Q8_0);
            put_f16(p, H1);
            for (int i = 0; i < bs; ++i) p[2 + i] = (uint8_t)((i + j) % 7 - 3);  // 有正有负
        }
    }
    std::vector<float> x(in_dim);
    for (int i = 0; i < in_dim; ++i) x[i] = (float)((i % 5) - 2) * 0.5f;

    // 内核结果
    std::vector<float> y(out_dim);
    matmul_quant(x.data(), w.data(), GgmlType::Q8_0, nullptr, in_dim, out_dim, y.data(), nullptr);

    // 参考：逐行反量化后点积
    int mism = 0;
    for (int j = 0; j < out_dim; ++j) {
        std::vector<float> row(in_dim);
        dequantize_block(&w[(size_t)j * row_bytes], row.data(), GgmlType::Q8_0, in_dim);
        double ref = 0;
        for (int i = 0; i < in_dim; ++i) ref += (double)x[i] * row[i];
        if (std::fabs(ref - y[j]) > 1e-3) ++mism;
    }
    if (mism == 0) { ++g_pass; std::printf("  OK %d rows matched\n", out_dim); }
    else { g_fail += mism; std::printf("  FAIL %d/%d rows mismatched\n", mism, out_dim); }

    // bias 生效验证
    std::vector<float> bias(out_dim);
    for (int j = 0; j < out_dim; ++j) bias[j] = (float)j;
    std::vector<float> y2(out_dim);
    matmul_quant(x.data(), w.data(), GgmlType::Q8_0, bias.data(), in_dim, out_dim, y2.data(), nullptr);
    CHECK_NEAR(y2[3] - y[3], 3.0, "matmul_quant bias");
}

}  // namespace

int main() {
    test_f32_f16();
    test_q4_0();  test_q4_1();
    test_q5_0();  test_q5_1();
    test_q8_0();  test_q8_1();
    test_q2_k();  test_q3_k();
    test_q4_k();  test_q5_k();  test_q6_k();
    test_unsupported();
    test_matmul_quant();

    std::printf("\n==== quant_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
