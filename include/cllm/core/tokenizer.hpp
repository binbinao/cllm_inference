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

    // 应用对话模板（支持 ChatML/Qwen、Llama3、Llama2/Mistral 等主流格式）。
    // 依据词表中实际存在的特殊 token 分派格式，返回不带 BOS 的 token ids。
    // 若均不匹配则回退为普通 encode（不静默丢模板，而是明确退化）。
    std::vector<int> apply_chat_template(const std::string& user_message,
                                         const std::string& system_message = "") const;

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    bool add_bos_token() const { return add_bos_token_; }
    bool has_chat_template() const { return has_chat_template_; }
    // 当前识别到的模板格式名（用于诊断/日志）
    std::string chat_template_format() const { return chat_template_format_; }

private:
    std::vector<std::string> tokens_;
    int bos_id_ = 1;
    int eos_id_ = 2;
    bool is_gpt2_ = false;   // true: BPE；false: SentencePiece(llama 类)
    bool add_bos_token_ = true;
    bool has_chat_template_ = false;
    std::string chat_template_;         // 原始 chat_template 字符串（Jinja）
    std::string chat_template_format_;  // 识别出的格式名

    // BPE 合并排名
    std::unordered_map<std::string, int> merges_rank_;

    int find_token(const std::string& s) const;
    std::vector<int> encode_bpe(const std::string& text) const;
    std::vector<int> encode_sp(const std::string& text) const;
};

}  // namespace cllm
