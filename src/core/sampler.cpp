#include "cllm/core/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cllm {

namespace {
// 温度缩放 + softmax
std::vector<float> softmax(const std::vector<float>& logits, float temp) {
    std::vector<float> x = logits;
    if (temp <= 0.0f) temp = 1.0f;  // 保护，避免除零
    if (temp != 1.0f) {
        for (auto& v : x) v /= temp;
    }
    float max_v = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (auto& v : x) { v = std::exp(v - max_v); sum += v; }
    if (sum > 0) for (auto& v : x) v /= sum;
    return x;
}
}  // namespace

int Sampler::sample(const std::vector<float>& logits,
                    const SampleParams& p,
                    std::mt19937& rng) {
    int n = (int)logits.size();
    if (n == 0) return 0;

    // temperature == 0 -> greedy（取 argmax）
    if (p.temperature <= 0.0f) {
        return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    std::vector<float> probs = softmax(logits, p.temperature);

    // 构建 (index, prob) 并按 prob 降序
    std::vector<std::pair<int, float>> items(n);
    for (int i = 0; i < n; ++i) items[i] = {i, probs[i]};
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // top-k 过滤
    int k = (p.top_k > 0 && p.top_k < n) ? p.top_k : n;
    float sum = 0.0f;
    std::vector<std::pair<int, float>> pool;
    pool.reserve(k);
    for (int i = 0; i < k; ++i) {
        // top-p：累积概率超过 top_p 则截断
        if (p.top_p > 0.0f && p.top_p < 1.0f && sum >= p.top_p) break;
        pool.push_back(items[i]);
        sum += items[i].second;
    }
    if (pool.empty()) pool.push_back(items[0]);

    // 归一化 pool 并采样
    float total = 0.0f;
    for (auto& it : pool) total += it.second;
    std::uniform_real_distribution<float> dist(0.0f, total);
    float r = dist(rng);
    float acc = 0.0f;
    for (auto& it : pool) {
        acc += it.second;
        if (r <= acc) return it.first;
    }
    return pool.back().first;
}

}  // namespace cllm
