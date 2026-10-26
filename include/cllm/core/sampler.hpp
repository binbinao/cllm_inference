#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace cllm {

// 采样参数
struct SampleParams {
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 40;
};

// 若 text 以任一 stop 为后缀，写入去掉该后缀的结果并返回 true。
inline bool consume_stop_suffix(const std::string& text,
                                const std::vector<std::string>& stops,
                                std::string& stripped) {
    for (const auto& s : stops) {
        if (s.empty() || text.size() < s.size()) continue;
        if (text.compare(text.size() - s.size(), s.size(), s) == 0) {
            stripped = text.substr(0, text.size() - s.size());
            return true;
        }
    }
    return false;
}

// 采样器：对 logits 施加温度、top-k、top-p，返回采样出的 token id
class Sampler {
public:
    static int sample(const std::vector<float>& logits,
                      const SampleParams& p,
                      std::mt19937& rng);
};

}  // namespace cllm
