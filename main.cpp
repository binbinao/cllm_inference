#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cllm/core/engine.hpp"
#include "cllm/core/gguf.hpp"
#include "cllm/core/sampler.hpp"
#include "cllm/core/thread_pool.hpp"
#include "cllm/core/tokenizer.hpp"
#include "cllm/server/http_server.hpp"

namespace {

void print_usage() {
    std::cout <<
        "Usage: cllm_inference --model <path> [options]\n"
        "Options:\n"
        "  --model <path>       GGUF model file (required)\n"
        "  --prompt <text>      Run local generation with this prompt\n"
        "  --port <n>           HTTP server port (default 8080)\n"
        "  --threads <n>        Number of CPU threads (default: hardware concurrency)\n"
        "  --ctx <n>            Cap context / KV hard limit (default: model n_ctx)\n"
        "  --temperature <f>    Sampling temperature (default 0.8)\n"
        "  --top-p <f>          Top-p (nucleus) sampling (default 0.9)\n"
        "  --top-k <n>          Top-k sampling (default 40)\n"
        "  --seed <n>           RNG seed (default: random)\n"
        "  --stop <str>         Stop suffix (repeatable)\n"
        "  --max-tokens <n>     Max tokens to generate (default 64)\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt;
    uint16_t port = 8080;
    size_t threads = 0;
    cllm::SampleParams sp;
    int max_tokens = 64;
    int ctx_limit = 0;
    uint32_t seed = 0;
    bool has_seed = false;
    std::vector<std::string> stops;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << name << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (arg == "--model") model_path = next("--model");
        else if (arg == "--prompt") prompt = next("--prompt");
        else if (arg == "--port") port = (uint16_t)std::stoi(next("--port"));
        else if (arg == "--threads") threads = (size_t)std::stoi(next("--threads"));
        else if (arg == "--ctx") ctx_limit = std::stoi(next("--ctx"));
        else if (arg == "--seed") { seed = (uint32_t)std::stoul(next("--seed")); has_seed = true; }
        else if (arg == "--stop") stops.push_back(next("--stop"));
        else if (arg == "--temperature") sp.temperature = std::stof(next("--temperature"));
        else if (arg == "--top-p") sp.top_p = std::stof(next("--top-p"));
        else if (arg == "--top-k") sp.top_k = std::stoi(next("--top-k"));
        else if (arg == "--max-tokens") max_tokens = std::stoi(next("--max-tokens"));
        else if (arg == "--help" || arg == "-h") { print_usage(); return 0; }
        else { std::cerr << "unknown argument: " << arg << "\n"; print_usage(); return 1; }
    }

    if (model_path.empty()) {
        std::cerr << "error: --model is required\n";
        print_usage();
        return 1;
    }

    try {
        // 加载模型
        auto model = cllm::GgufLoader::load(model_path);
        const auto& cfg = model.config;

        const uint32_t model_ctx = cfg.n_ctx;
        cllm::ThreadPool pool(threads);
        cllm::TransformerEngine engine(model, pool, cllm::WeightMode::Quant, ctx_limit);
        std::cout << "Loaded model: " << model_path << "\n"
                  << "  architecture: " << cfg.arch << "\n"
                  << "  layers: " << cfg.n_layers
                  << ", hidden: " << cfg.n_embd
                  << ", heads: " << cfg.n_head
                  << " (kv: " << cfg.n_head_kv << ")\n"
                  << "  vocab: " << cfg.vocab_size
                  << ", ctx: " << engine.config().n_ctx << "/" << model_ctx << "\n"
                  << std::flush;
        cllm::Tokenizer tokenizer(model);

        if (!prompt.empty()) {
            // 本地生成
            std::vector<int> ids;
            if (!model.chat_template.empty()) {
                ids = tokenizer.apply_chat_template(prompt);
            } else {
                ids = tokenizer.encode(prompt, model.add_bos_token);
            }
            std::mt19937 rng;
            if (has_seed) rng.seed(seed);
            else rng.seed(std::random_device{}());

            using clock = std::chrono::steady_clock;
            const auto t0 = clock::now();
            std::vector<float> logits = engine.prefill(ids);
            const auto t1 = clock::now();

            cllm::Utf8StreamDecoder dec;
            std::cout << "\n" << prompt;
            int n_gen = 0;
            std::string generated;
            for (int i = 0; i < max_tokens; ++i) {
                int next = cllm::Sampler::sample(logits, sp, rng);
                if (tokenizer.is_eos(next)) break;
                ids.push_back(next);
                ++n_gen;
                std::string piece = dec.feed(tokenizer.decode(next));
                generated += piece;
                std::string stripped;
                if (cllm::consume_stop_suffix(generated, stops, stripped)) {
                    const size_t extra = generated.size() - stripped.size();
                    if (extra <= piece.size()) std::cout << piece.substr(0, piece.size() - extra);
                    else std::cout << piece;
                    std::cout << std::flush;
                    break;
                }
                std::cout << piece << std::flush;
                logits = engine.forward(next);
            }
            std::string tail = dec.flush();
            generated += tail;
            std::string stripped;
            if (cllm::consume_stop_suffix(generated, stops, stripped)) {
                const size_t extra = generated.size() - stripped.size();
                if (extra <= tail.size()) std::cout << tail.substr(0, tail.size() - extra);
            } else {
                std::cout << tail;
            }
            std::cout << "\n";
            const auto t2 = clock::now();

            auto ms = [](clock::time_point a, clock::time_point b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            const double prefill_ms = ms(t0, t1);
            const double decode_ms = ms(t1, t2);
            const int n_prompt = (int)ids.size() - n_gen;
            if (prefill_ms > 0 && n_prompt > 0) {
                std::cout << "prefill: " << n_prompt << " tok, "
                          << (n_prompt / (prefill_ms / 1000.0)) << " tok/s\n";
            }
            if (decode_ms > 0 && n_gen > 0) {
                std::cout << "decode: " << n_gen << " tok, "
                          << (n_gen / (decode_ms / 1000.0)) << " tok/s\n";
            }
        } else {
            // HTTP 服务
            std::cout << "Starting HTTP server on port " << port << " ...\n";
            cllm::HttpServer server(engine, tokenizer, port, model_path);
            server.start();
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}