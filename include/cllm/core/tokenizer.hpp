#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cllm/core/gguf.hpp"

namespace cllm {

struct ChatMessage {
    std::string role;
    std::string content;
};

// 流式拼出完整 UTF-8 码点，避免 BPE 按字节吐出半截中文。
class Utf8StreamDecoder {
public:
    std::string feed(std::string_view bytes);
    std::string flush();

private:
    std::string buf_;
};

// 基于 GGUF 内置词表的编解码器（支持 llama 类 SentencePiece 与 gpt2 BPE）
class Tokenizer {
public:
    explicit Tokenizer(const GgufModel& model);

    std::vector<int> encode(const std::string& text, bool add_bos = true) const;
    std::string decode(int token) const;
    std::string decode(const std::vector<int>& ids) const;

    std::vector<int> apply_chat_template(const std::string& user_message,
                                         const std::string& system_message = "") const;
    std::vector<int> apply_chat_template(const std::vector<ChatMessage>& messages) const;

    bool is_eos(int token_id) const;

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    bool add_bos_token() const { return add_bos_token_; }
    bool has_chat_template() const { return has_chat_template_; }
    std::string chat_template_format() const { return chat_template_format_; }

private:
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_set<int> stop_ids_;
    int bos_id_ = 1;
    int eos_id_ = 2;
    bool is_gpt2_ = false;
    bool add_bos_token_ = true;
    bool has_chat_template_ = false;
    std::string chat_template_;
    std::string chat_template_format_;
    std::unordered_map<std::string, int> merges_rank_;

    int find_token(const std::string& s) const;
    std::vector<int> encode_bpe(const std::string& text) const;
    std::vector<int> encode_sp(const std::string& text) const;
};

}  // namespace cllm
