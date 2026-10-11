// 模型级量化对比基准：
//   对 models/ 下的模型，分别用「反量化 fp32」与「文件名上的量化方法」两种推理路径，
//   测试不同输入长度（prompt token 数）下的 prefill 与 decode 耗时，并计算量化加速比。
//
//   - 反量化 fp32（use_quant=false）：所有量化权重 dequantize 为 float 常驻，forward 走纯 float 矩阵乘
//   - 量化方法（use_quant=true）：权重零拷贝引用 mmap，forward 走 vec_dot 整数点积快速路径
//
// 用法：model_bench [model1 model2 ...]
//   不带参数时默认测试 models/ 下三个模型（按规模从小到大）。
#include "cllm/core/engine.hpp"
#include "cllm/core/gguf.hpp"
#include "cllm/core/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace cllm;

namespace {

double time_ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

const char* type_name(GgmlType t) {
    switch (t) {
        case GgmlType::F32:  return "F32";
        case GgmlType::F16:  return "F16";
        case GgmlType::Q4_0: return "Q4_0";
        case GgmlType::Q4_1: return "Q4_1";
        case GgmlType::Q5_0: return "Q5_0";
        case GgmlType::Q5_1: return "Q5_1";
        case GgmlType::Q8_0: return "Q8_0";
        case GgmlType::Q8_1: return "Q8_1";
        case GgmlType::Q2_K: return "Q2_K";
        case GgmlType::Q3_K: return "Q3_K";
        case GgmlType::Q4_K: return "Q4_K";
        case GgmlType::Q5_K: return "Q5_K";
        case GgmlType::Q6_K: return "Q6_K";
        default: return "?";
    }
}

struct ModeResult {
    std::vector<double> prefill_ms;    // 每个长度的 prefill 总耗时
    std::vector<double> prefill_tok_s; // 每个长度的 prefill 吞吐（token/s）
    std::vector<double> decode_ms;     // 每个长度 prefill 之后的单步 decode 耗时
};

// 对单个模型做一种推理模式的基准
//   lens           输入长度序列（token 数）
//   token_id       用于构造序列的 token（重复该 token，内容不影响计算量）
//   prefill_rounds prefill 测量轮数（取最小值，排除冷启动）
//   decode_steps   prefill 后连续 decode 的步数（取平均）
ModeResult bench_mode(const GgufModel& model, ThreadPool& pool, bool use_quant,
                      const std::vector<int>& lens, int token_id,
                      int prefill_rounds, int decode_steps) {
    TransformerEngine engine(model, pool, use_quant);
    ModeResult r;
    r.prefill_ms.resize(lens.size());
    r.prefill_tok_s.resize(lens.size());
    r.decode_ms.resize(lens.size());

    for (size_t li = 0; li < lens.size(); ++li) {
        const int L = lens[li];

        // ---- prefill：reset + 连续 forward L 个 token ----
        // 首轮作为 warmup（冷缓存），不计入；后续取最小耗时
        double best_prefill = 1e18;
        for (int rnd = 0; rnd < prefill_rounds + 1; ++rnd) {
            engine.reset_kv_cache();
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < L; ++i) engine.forward(token_id);
            auto t1 = std::chrono::steady_clock::now();
            double ms = time_ms(t0, t1);
            if (rnd > 0 && ms < best_prefill) best_prefill = ms;
        }
        r.prefill_ms[li] = best_prefill;
        r.prefill_tok_s[li] = L / (best_prefill / 1000.0);

        // ---- decode：prefill 结束后（seq_len = L）连续 decode 若干步 ----
        // 先 warmup 1 步，再测 decode_steps 步取平均（平均对应 seq ≈ L + decode_steps/2）
        engine.forward(token_id);  // warmup，seq -> L+1
        double sum = 0.0;
        for (int s = 0; s < decode_steps; ++s) {
            auto t0 = std::chrono::steady_clock::now();
            engine.forward(token_id);
            auto t1 = std::chrono::steady_clock::now();
            sum += time_ms(t0, t1);
        }
        r.decode_ms[li] = sum / decode_steps;
    }
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> models;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) models.emplace_back(argv[i]);
    } else {
        models = {
            "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "models/Qwen3-0.6B-Q8_0.gguf",
            "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        };
    }

    // 不同输入长度（prompt token 数），覆盖短/中/长上下文
    const std::vector<int> lens = {16, 32, 64, 128, 256};
    const int prefill_rounds = 2;
    const int decode_steps = 3;

    ThreadPool pool(0);
    std::printf("cllm 模型级量化对比：反量化 fp32  vs  量化（文件名上的量化方法）\n");
    std::printf("threads=%zu   prefill 取 %d 轮最小，decode 取 %d 步平均\n\n",
                pool.size(), prefill_rounds, decode_steps);

    for (const auto& mp : models) {
        std::printf("============================================================\n");
        std::printf("模型: %s\n", mp.c_str());
        std::fflush(stdout);

        GgufModel model = GgufLoader::load(mp);
        std::printf("  arch=%s  layers=%u  hidden=%u  n_ff=%u  vocab=%u\n",
                    model.config.arch.c_str(), model.config.n_layers, model.config.n_embd,
                    model.config.n_ff, model.config.vocab_size);

        // 实际量化类型分布（按参数量加权），用于识别「主体量化」而非仅看文件名
        {
            std::map<GgmlType, uint64_t> qcnt;
            uint64_t total = 0;
            for (const auto& t : model.tensor_list) {
                if (t.type != GgmlType::F32 && t.type != GgmlType::F16) {
                    qcnt[t.type] += t.numel();
                    total += t.numel();
                }
            }
            std::vector<std::pair<GgmlType, uint64_t>> v(qcnt.begin(), qcnt.end());
            std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            std::printf("  实际量化(参数量占比): ");
            for (size_t i = 0; i < v.size(); ++i) {
                std::printf("%s%s %.1f%%", i ? ", " : "", type_name(v[i].first),
                            100.0 * (double)v[i].second / (double)total);
            }
            std::printf("\n");
        }

        const int token_id = model.bos_id;

        // 两种模式（先 fp32 后量化；fp32 常驻内存较大，测完即释放）
        auto fp32  = bench_mode(model, pool, false, lens, token_id, prefill_rounds, decode_steps);
        auto quant = bench_mode(model, pool, true,  lens, token_id, prefill_rounds, decode_steps);

        std::printf("  %-6s | %12s %12s %8s | %12s %12s %8s\n",
                    "输入", "fp32 prefill", "quant prefill", "prefill", "fp32 decode", "quant decode", "decode");
        std::printf("  %-6s | %12s %12s %8s | %12s %12s %8s\n",
                    "(tok)", "(ms)", "(ms)", "加速x", "(ms)", "(ms)", "加速x");
        std::printf("  ------+-----------------------------------------+--------------------------------------\n");
        for (size_t i = 0; i < lens.size(); ++i) {
            double ps = fp32.prefill_ms[i] / quant.prefill_ms[i];
            double ds = fp32.decode_ms[i] / quant.decode_ms[i];
            std::printf("  %-6d | %12.3f %12.3f %8.2f | %12.4f %12.4f %8.2f\n",
                        lens[i], fp32.prefill_ms[i], quant.prefill_ms[i], ps,
                        fp32.decode_ms[i], quant.decode_ms[i], ds);
        }
        // prefill 吞吐（token/s）
        std::printf("  prefill 吞吐 fp32→quant (tok/s): ");
        for (size_t i = 0; i < lens.size(); ++i) {
            std::printf("%s%d:%.0f→%.0f", i ? ", " : "", lens[i],
                        fp32.prefill_tok_s[i], quant.prefill_tok_s[i]);
        }
        std::printf("\n\n");
        std::fflush(stdout);
    }

    std::printf("注：量化加速比 >1 表示量化快于 fp32。prefill 加速来自 FFN/投影矩阵乘；\n");
    std::printf("    decode 的 attention 为 float 计算，随序列变长占比上升，故 decode 加速比通常低于 prefill。\n");
    return 0;
}
