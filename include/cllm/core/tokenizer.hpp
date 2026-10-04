#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "cllm/core/gguf.hpp"

namespace cllm {

// 基于 GGUF 内置词表的编解码器（支持 llama 类 SentencePiece 与 gpt2 BPE）
class Tokenizer {
public:
    explicit Tokenizer(const GgufModel& model);

    // 文本 -> token ids（含 BOS 处理）
    std::vector<int> encode(const std::string& text, bool add_bos = true) const;
    // token id -> 文本片段
    std::string decode(int token) const;
    // token ids -> 完整文本
    std::string decode(const std::vector<int>& ids) const;

    // 应用 Qwen 风格 chat template（<|im_start|>system/user/assistant<|im_end|>），
    // 返回不带 BOS 的 token ids。若模型无特殊 token 或非 chat 模型，回退为普通 encode。
    std::vector<int> apply_chat_template(const std::string& user_message,
                                         const std::string& system_message = "") const;

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    bool add_bos_token() const { return add_bos_token_; }
    bool has_chat_template() const { return has_chat_template_; }

private:
    std::vector<std::string> tokens_;
    int bos_id_ = 1;
    int eos_id_ = 2;
    bool is_gpt2_ = false;   // true: BPE；false: SentencePiece(llama 类)
    bool add_bos_token_ = true;
    bool has_chat_template_ = false;

    // BPE 合并排名
    std::unordered_map<std::string, int> merges_rank_;

    int find_token(const std::string& s) const;
    std::vector<int> encode_bpe(const std::string& text) const;
    std::vector<int> encode_sp(const std::string& text) const;
};

}  // namespace cllm
