#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "cllm/core/tensor.hpp"

namespace cllm {

// 从 GGUF metadata 抽取出的模型超参数
struct ModelConfig {
    std::string arch;
    uint32_t n_layers = 0;    // block count
    uint32_t n_embd = 0;      // embedding length / hidden dim
    uint32_t n_head = 0;      // attention head count
    uint32_t n_head_kv = 0;   // GQA 的 KV 头数
    uint32_t n_ctx = 0;       // context length
    uint32_t n_ff = 0;        // feed forward length
    uint32_t vocab_size = 0;
    float rope_theta = 10000.0f;  // rope freq base
    float norm_eps = 1e-5f;       // rms norm epsilon

    // 派生量
    uint32_t head_dim() const { return n_embd / n_head; }
};

// GGUF metadata 的通用值容器（仅保存我们关心的标量/字符串/数组）
struct GgufValue {
    enum class Kind { None, U32, I32, F32, Bool, Str, ArrStr, ArrF32, ArrI32 };
    Kind kind = Kind::None;
    uint32_t u32 = 0;
    int32_t i32 = 0;
    float f32 = 0;
    bool b = false;
    std::string str;
    std::vector<std::string> arr_str;
    std::vector<float> arr_f32;
    std::vector<int32_t> arr_i32;
};

// 加载完成的模型：超参 + 张量 + 词表 + mmap 管理
struct GgufModel {
    ModelConfig config;
    std::unordered_map<std::string, Tensor> tensors;   // 按 name 索引
    std::vector<Tensor> tensor_list;                   // 保持文件顺序

    // 词表
    std::vector<std::string> tokens;
    std::vector<float> token_scores;
    std::vector<int32_t> token_types;
    std::vector<std::string> merges;   // BPE 合并规则
    std::string tokenizer_model;       // "llama" / "gpt2"
    int bos_id = 1;
    int eos_id = 2;
    bool add_bos_token = true;         // 编码时是否自动加 BOS
    std::string chat_template;         // 对话模板（Jinja，当前仅用于识别是否 chat 模型）

    // mmap 管理（析构时 unmap）
    void* mmap_base = nullptr;
    size_t mmap_size = 0;

    ~GgufModel();
    GgufModel() = default;
    GgufModel(GgufModel&&) noexcept;
    GgufModel& operator=(GgufModel&&) noexcept;
    GgufModel(const GgufModel&) = delete;
    GgufModel& operator=(const GgufModel&) = delete;
};

// GGUF 解析器
class GgufLoader {
public:
    static GgufModel load(const std::string& path);
};

}  // namespace cllm
