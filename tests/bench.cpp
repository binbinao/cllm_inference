// 量化矩阵乘内核基准：报告不同量化类型下的吞吐，量化"相比 naive float"的加速比。
// 用法：bench [iters]；不带参数时用默认迭代次数。
#include "cllm/core/quant.hpp"
#include "cllm/core/thread_pool.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace cllm;

namespace {

inline void put_f16(uint8_t* p, uint16_t h) { memcpy(p, &h, 2); }

// 构造一份量化权重（值不重要，只用于测吞吐）
std::vector<uint8_t> make_quant_weight(GgmlType t, int in_dim, int out_dim) {
    const int bs = block_size(t);
    const size_t row_bytes = (size_t)(in_dim / bs) * type_size(t);
    std::vector<uint8_t> w(row_bytes * out_dim, 0);
    for (int j = 0; j < out_dim; ++j) {
        uint8_t* row = &w[(size_t)j * row_bytes];
        for (int blk = 0; blk < in_dim / bs; ++blk) {
            uint8_t* p = row + (size_t)blk * type_size(t);
            // 第一个 f16 位置写 1.0（大多数类型的 d）
            put_f16(p, 0x3C00);
            // 填充后续字节为可预测的小值
            for (size_t k = 2; k < type_size(t); ++k) p[k] = (uint8_t)((k + blk) % 5);
        }
    }
    return w;
}

// naive float 参考：x · Wᵀ（W 已是 float），与量化内核一样走线程池并行
void matmul_float_par(const std::vector<float>& x, const std::vector<float>& W,
                      int in_dim, int out_dim, std::vector<float>& y, ThreadPool& pool) {
    y.assign(out_dim, 0.0f);
    pool.parallel_for(0, (size_t)out_dim, [&](size_t b, size_t e) {
        for (size_t j = b; j < e; ++j) {
            const float* wr = W.data() + j * in_dim;
            float s = 0.0f;
            for (int i = 0; i < in_dim; ++i) s += x[i] * wr[i];
            y[j] = s;
        }
    });
}

double time_ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Case { const char* name; GgmlType type; int bits; };

// 每种类型每个元素占用的字节数（用于计算读权重的有效内存带宽）
double bytes_per_elem(GgmlType t) {
    switch (t) {
        case GgmlType::F32: return 4.0;
        case GgmlType::F16: return 2.0;
        default: return (double)type_size(t) / (double)block_size(t);
    }
}

// 每种类型的名义量化位数（用于展示"量化越小"的单调性）
int type_bits(GgmlType t) {
    switch (t) {
        case GgmlType::F32: return 32;
        case GgmlType::F16: return 16;
        case GgmlType::Q8_0: case GgmlType::Q8_1: return 8;
        case GgmlType::Q6_K: return 6;
        case GgmlType::Q5_0: case GgmlType::Q5_1: case GgmlType::Q5_K: return 5;
        case GgmlType::Q4_0: case GgmlType::Q4_1: case GgmlType::Q4_K: return 4;
        case GgmlType::Q3_K: return 3;
        case GgmlType::Q2_K: return 2;
        default: return 0;
    }
}

}  // namespace

int main(int argc, char** argv) {
    // 用法：bench [iters] [out_dim]
    //   iters   迭代次数（默认 50）
    //   out_dim 输出维度（默认 1024）；调大可让内存带宽成为瓶颈，
    //           从而观察"量化越小 → 权重字节越少 → 越快"的对齐关系
    int iters = (argc > 1) ? std::atoi(argv[1]) : 50;
    int out_dim = (argc > 2) ? std::atoi(argv[2]) : 1024;

    // in_dim 固定 1024（典型 hidden 维度，且能整除 32 与 256 block）
    const int in_dim = 1024;
    std::vector<float> x(in_dim);
    for (int i = 0; i < in_dim; ++i) x[i] = (float)((i % 7) - 3) * 0.1f;

    ThreadPool pool(0);
    std::printf("bench: in=%d out=%d iters=%d threads=%zu\n\n",
                in_dim, out_dim, iters, pool.size());

    // --- naive float 基线（同样多线程，保证公平对比）---
    std::vector<float> Wf((size_t)in_dim * out_dim);
    for (size_t i = 0; i < Wf.size(); ++i) Wf[i] = (float)((i % 11) - 5) * 0.01f;
    std::vector<float> y_float;
    double t_float = 0;
    for (int r = 0; r < iters; ++r) {
        auto a = std::chrono::steady_clock::now();
        matmul_float_par(x, Wf, in_dim, out_dim, y_float, pool);
        auto b = std::chrono::steady_clock::now();
        t_float += time_ms(a, b);
    }
    t_float /= iters;

    Case cases[] = {
        {"F32",  GgmlType::F32,  32},
        {"F16",  GgmlType::F16,  16},
        {"Q8_0", GgmlType::Q8_0,  8},
        {"Q6_K", GgmlType::Q6_K,  6},
        {"Q5_K", GgmlType::Q5_K,  5},
        {"Q5_0", GgmlType::Q5_0,  5},
        {"Q4_K", GgmlType::Q4_K,  4},
        {"Q4_0", GgmlType::Q4_0,  4},
    };

    std::printf("%-6s %4s %8s %10s %10s %10s %10s\n",
                "type", "bits", "B/elem", "ms/op", "GFLOP/s", "BW(GB/s)", "vs float");
    std::printf("%-6s %4s %8.2f %10.3f %10.2f %10.2f %10s\n",
                "float", "32", 4.0, t_float,
                2.0 * in_dim * out_dim / (t_float * 1e6),
                (double)out_dim * in_dim * 4.0 / (t_float * 1e6), "1.00x");

    for (const auto& c : cases) {
        std::vector<uint8_t> wq = make_quant_weight(c.type, in_dim, out_dim);
        std::vector<float> y(out_dim);
        double t = 0;
        for (int r = 0; r < iters; ++r) {
            auto a = std::chrono::steady_clock::now();
            matmul_quant(x.data(), wq.data(), c.type, nullptr, in_dim, out_dim, y.data(), &pool);
            auto b = std::chrono::steady_clock::now();
            t += time_ms(a, b);
        }
        t /= iters;
        double gflops = 2.0 * in_dim * out_dim / (t * 1e6);
        double bw = (double)out_dim * in_dim * bytes_per_elem(c.type) / (t * 1e6);
        std::printf("%-6s %4d %8.2f %10.3f %10.2f %10.2f %9.2fx\n",
                    c.name, c.bits, bytes_per_elem(c.type), t, gflops, bw, t_float / t);
    }

    std::printf("\n注：量化内核按需反量化，省内存；此处对比其与纯 float matmul 的吞吐。\n");
    std::printf("    BW(GB/s) = 读权重有效带宽，量化越小该值越低，带宽瓶颈下推理越快。\n");
    return 0;
}
