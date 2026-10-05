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

struct Case { const char* name; GgmlType type; };

}  // namespace

int main(int argc, char** argv) {
    int iters = (argc > 1) ? std::atoi(argv[1]) : 50;

    // 维度取 1024（可被 K 量化的 256 block 整除），模拟典型隐藏层规模
    const int in_dim = 1024, out_dim = 1024;
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
        {"F32",  GgmlType::F32},
        {"F16",  GgmlType::F16},
        {"Q8_0", GgmlType::Q8_0},
        {"Q4_0", GgmlType::Q4_0},
        {"Q5_0", GgmlType::Q5_0},
        {"Q6_K", GgmlType::Q6_K},
        {"Q4_K", GgmlType::Q4_K},
    };

    std::printf("%-6s %10s %12s %14s\n", "type", "ms/op", "GFLOP/s", "vs float");
    std::printf("%-6s %10.3f %12.2f %14s\n", "float", t_float,
                2.0 * in_dim * out_dim / (t_float * 1e6), "1.00x");

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
        std::printf("%-6s %10.3f %12.2f %13.2fx\n", c.name, t, gflops, t_float / t);
    }

    std::printf("\n注：量化内核按需反量化，省内存；此处对比其与纯 float matmul 的吞吐。\n");
    return 0;
}
