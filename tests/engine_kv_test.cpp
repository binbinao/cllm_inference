// 引擎 Prefix Cache / n_ctx / skip-logits。模型缺失时 skip（退出 0）。
#include "cllm/core/engine.hpp"
#include "cllm/core/gguf.hpp"
#include "cllm/core/thread_pool.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cllm;

namespace {
int g_fail = 0, g_pass = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (cond) { ++g_pass; }                                              \
        else { std::printf("  FAIL %s\n", msg); ++g_fail; }                  \
    } while (0)

bool file_ok(const char* p) {
    std::ifstream f(p);
    return static_cast<bool>(f);
}

}  // namespace

int main() {
    const char* path = "models/Qwen3-0.6B-Q8_0.gguf";
    if (!file_ok(path)) {
        std::printf("SKIP engine_kv_test: %s not found\n", path);
        return 0;
    }

    GgufModel model = GgufLoader::load(path);
    ThreadPool pool(2);
    TransformerEngine eng(model, pool);
    const int t = model.bos_id;
    const auto vocab = model.config.vocab_size;

    std::printf("[prefill empty]\n");
    bool threw_empty = false;
    try { eng.prefill({}); } catch (const std::exception&) { threw_empty = true; }
    CHECK(threw_empty, "empty prefill throws");

    std::printf("[prefill + prefix reuse]\n");
    auto logits = eng.prefill({t, t, t});
    CHECK(eng.seq_len() == 3, "prefill 3 -> seq_len 3");
    CHECK(logits.size() == vocab, "prefill returns vocab logits");

    auto logits2 = eng.prefill({t, t, t, t});
    CHECK(eng.seq_len() == 4, "prefix-3 reuse then +1 -> seq_len 4");
    CHECK(logits2.size() == vocab, "extended prefill returns logits");

    std::printf("[skip logits]\n");
    auto empty_logits = eng.forward(t, false);
    CHECK(empty_logits.empty(), "compute_logits=false returns empty");
    CHECK(eng.seq_len() == 5, "skip-logits still advances seq");

    std::printf("[n_ctx overflow]\n");
    bool threw_ctx = false;
    try {
        std::vector<int> too_big(static_cast<size_t>(model.config.n_ctx) + 1, t);
        eng.prefill(too_big);
    } catch (const std::exception& e) {
        threw_ctx = std::string(e.what()).find("context length exceeded") != std::string::npos;
    }
    CHECK(threw_ctx, "prefill n_ctx+1 throws context length exceeded");

    std::printf("[reset]\n");
    eng.reset_kv_cache();
    CHECK(eng.seq_len() == 0, "reset clears seq_len");

    std::printf("==== engine_kv_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
