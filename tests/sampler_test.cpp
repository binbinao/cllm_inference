// Sampler 采样器测试：验证 greedy 退化、温度、top-k/top-p 过滤与分布合理性。
#include "cllm/core/sampler.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace cllm;

namespace {
int g_fail = 0, g_pass = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (cond) { ++g_pass; }                                              \
        else { std::printf("  FAIL %s\n", msg); ++g_fail; }                  \
    } while (0)

// temperature = 0 -> 必须取 argmax（greedy）
void test_greedy() {
    std::printf("[greedy]\n");
    std::vector<float> logits = {0.1f, 5.0f, -3.0f, 2.0f, 4.9f};
    SampleParams p; p.temperature = 0.0f;
    std::mt19937 rng(123);
    for (int i = 0; i < 20; ++i) {
        int t = Sampler::sample(logits, p, rng);
        CHECK(t == 1, "greedy should pick argmax=1");
    }
}

// top-k = 1 时无论温度多高都只能取 argmax
void test_topk_one() {
    std::printf("[top-k=1]\n");
    std::vector<float> logits = {1.0f, 3.0f, 2.0f, 0.5f};
    SampleParams p; p.temperature = 1.0f; p.top_k = 1; p.top_p = 1.0f;
    std::mt19937 rng(7);
    for (int i = 0; i < 50; ++i)
        CHECK(Sampler::sample(logits, p, rng) == 1, "top-k=1 -> argmax");
}

// top-k 限制：只允许 top-2 集合内的 token 出现
void test_topk_scope() {
    std::printf("[top-k scope]\n");
    std::vector<float> logits = {0.0f, 0.1f, 5.0f, 4.0f, -10.0f};
    SampleParams p; p.temperature = 1.0f; p.top_k = 2; p.top_p = 1.0f;
    std::mt19937 rng(99);
    bool ok = true;
    for (int i = 0; i < 200; ++i) {
        int t = Sampler::sample(logits, p, rng);
        if (t != 2 && t != 3) ok = false;  // top-2 = {2,3}
    }
    CHECK(ok, "top-k=2 must stay within {2,3}");
}

// top-p 极端：top_p 很小时只保留最高概率 token
void test_topp_tight() {
    std::printf("[top-p tight]\n");
    std::vector<float> logits = {0.0f, 10.0f, 0.0f, 0.0f};
    SampleParams p; p.temperature = 1.0f; p.top_k = 0; p.top_p = 0.01f;
    std::mt19937 rng(3);
    bool ok = true;
    for (int i = 0; i < 100; ++i)
        if (Sampler::sample(logits, p, rng) != 1) ok = false;
    CHECK(ok, "tight top-p -> argmax");
}

// 温度影响：高温下低概率 token 也应有机会出现
void test_temperature_spread() {
    std::printf("[temperature]\n");
    std::vector<float> logits = {3.0f, 0.0f};  // 概率约 [0.95, 0.05]
    SampleParams p; p.temperature = 1.0f; p.top_k = 0; p.top_p = 1.0f;
    std::mt19937 rng(2024);
    int cnt0 = 0, cnt1 = 0;
    for (int i = 0; i < 2000; ++i) {
        (Sampler::sample(logits, p, rng) == 1) ? ++cnt1 : ++cnt0;
    }
    // 期望 cnt1 约占 4.7% → 2000 次里应落在合理区间（宽松判定）
    CHECK(cnt1 > 40 && cnt1 < 200, "temp=1 low-prob token should appear ~5%");
}

}  // namespace

int main() {
    test_greedy();
    test_topk_one();
    test_topk_scope();
    test_topp_tight();
    test_temperature_spread();

    std::printf("==== sampler_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
