#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace cllm {

// 采样参数
struct SampleParams {
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 40;
};

// 采样器：对 logits 施加温度、top-k、top-p，返回采样出的 token id
class Sampler {
public:
    static int sample(const std::vector<float>& logits,
                      const SampleParams& p,
                      std::mt19937& rng);
};

}  // namespace cllm
