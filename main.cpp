#include <cstring>
#include <iostream>
#include <string>

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
        "  --temperature <f>    Sampling temperature (default 0.8)\n"
        "  --top-p <f>          Top-p (nucleus) sampling (default 0.9)\n"
        "  --top-k <n>          Top-k sampling (default 40)\n"
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

        std::cout << "Loaded model: " << model_path << "\n"
                  << "  architecture: " << cfg.arch << "\n"
                  << "  layers: " << cfg.n_layers
                  << ", hidden: " << cfg.n_embd
                  << ", heads: " << cfg.n_head
                  << " (kv: " << cfg.n_head_kv << ")\n"
                  << "  vocab: " << cfg.vocab_size
                  << ", ctx: " << cfg.n_ctx << "\n"
                  << std::flush;

        cllm::ThreadPool pool(threads);
        cllm::TransformerEngine engine(model, pool);
        cllm::Tokenizer tokenizer(model);

        if (!prompt.empty()) {
            // 本地生成
            std::vector<int> ids;
            if (!model.chat_template.empty()) {
                ids = tokenizer.apply_chat_template(prompt);
            } else {
                ids = tokenizer.encode(prompt, model.add_bos_token);
            }
            std::mt19937 rng(std::random_device{}());
            engine.reset_kv_cache();

            // 先 forward 整个 prompt
            std::vector<float> logits;
            for (int id : ids) logits = engine.forward(id);

            std::cout << "\n" << prompt;
            for (int i = 0; i < max_tokens; ++i) {
                int next = cllm::Sampler::sample(logits, sp, rng);
                if (next == tokenizer.eos_id()) break;
                ids.push_back(next);
                std::cout << tokenizer.decode(next) << std::flush;
                logits = engine.forward(next);
            }
            std::cout << "\n";
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