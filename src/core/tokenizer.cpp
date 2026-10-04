#include "cllm/core/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace cllm {

namespace {
// UTF-8 解码：读取一个码点（返回字节数），输出到 cp
int utf8_decode(const std::string& s, size_t pos, uint32_t& cp) {
    unsigned char c = (unsigned char)s[pos];
    if (c < 0x80) { cp = c; return 1; }
    int len;
    uint32_t val;
    if ((c & 0xE0) == 0xC0) { len = 2; val = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { len = 3; val = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { len = 4; val = c & 0x07; }
    else { cp = c; return 1; }
    for (int i = 1; i < len; ++i) {
        if (pos + i >= s.size()) { cp = c; return 1; }
        val = (val << 6) | ((unsigned char)s[pos + i] & 0x3F);
    }
    cp = val;
    return len;
}

// UTF-8 编码：码点 → UTF-8 字节
std::string utf8_encode(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

// gpt2 的 bytes_to_unicode 映射：字节 → Unicode 字符（UTF-8 字符串）
std::unordered_map<uint8_t, std::string> build_bytes_to_unicode() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);       // 33-126
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);     // 161-172
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);     // 174-255
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    std::unordered_map<uint8_t, std::string> m;
    for (size_t i = 0; i < bs.size(); ++i) {
        m[(uint8_t)bs[i]] = utf8_encode((uint32_t)cs[i]);
    }
    return m;
}

// bytes_to_unicode 的逆映射：Unicode 码点 → 原始字节
std::unordered_map<uint32_t, uint8_t> build_unicode_to_bytes() {
    const auto b2u = build_bytes_to_unicode();
    std::unordered_map<uint32_t, uint8_t> u2b;
    for (const auto& [b, u] : b2u) {
        uint32_t cp = 0;
        utf8_decode(u, 0, cp);
        u2b[cp] = b;
    }
    return u2b;
}
}  // namespace

Tokenizer::Tokenizer(const GgufModel& model)
    : tokens_(model.tokens), bos_id_(model.bos_id), eos_id_(model.eos_id) {
    is_gpt2_ = (model.tokenizer_model == "gpt2");
    add_bos_token_ = model.add_bos_token;
    has_chat_template_ = !model.chat_template.empty();
    // BPE 合并排名：rank = 索引（越靠前优先级越高）
    for (size_t i = 0; i < model.merges.size(); ++i) {
        merges_rank_[model.merges[i]] = (int)i;
    }
}

std::vector<int> Tokenizer::apply_chat_template(const std::string& user_message,
                                                const std::string& system_message) const {
    const int im_start = find_token("<|im_start|>");
    const int im_end = find_token("<|im_end|>");

    // 无 Qwen 特殊 token 时回退为普通编码
    if (im_start < 0 || im_end < 0) {
        return encode(user_message, add_bos_token_);
    }

    // 换行与普通文本一律走 encode，避免 bytes_to_unicode 映射差异
    std::vector<int> ids;
    auto push_text = [&](const std::string& text) {
        auto part = encode(text, false);
        ids.insert(ids.end(), part.begin(), part.end());
    };

    // 默认 system prompt（Qwen 系列）
    std::string sys = system_message;
    if (sys.empty()) {
        sys = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    }

    ids.push_back(im_start);
    push_text("system\n" + sys);
    ids.push_back(im_end);
    push_text("\n");

    ids.push_back(im_start);
    push_text("user\n" + user_message);
    ids.push_back(im_end);
    push_text("\n");

    ids.push_back(im_start);
    push_text("assistant\n");

    return ids;
}

int Tokenizer::find_token(const std::string& s) const {
    auto it = std::find(tokens_.begin(), tokens_.end(), s);
    if (it != tokens_.end()) return (int)(it - tokens_.begin());
    return -1;
}

// SentencePiece（llama 类）：直接按整词/字符匹配
std::vector<int> Tokenizer::encode_sp(const std::string& text) const {
    std::vector<int> ids;
    size_t i = 0;
    while (i < text.size()) {
        // 先尝试最长匹配（从当前位置开始的整词）
        int best = -1;
        // 简化：按空格分词 + 逐词匹配；对单个字符回退
        size_t j = i;
        while (j < text.size() && text[j] != ' ') ++j;
        std::string word = text.substr(i, j - i);
        if (!word.empty()) {
            best = find_token(word);
        }
        if (best >= 0) {
            ids.push_back(best);
            i = j;
        } else {
            // 逐字符（UTF-8 码点）
            uint32_t cp;
            int len = utf8_decode(text, i, cp);
            std::string ch = text.substr(i, len);
            int t = find_token(ch);
            if (t >= 0) ids.push_back(t);
            i += len;
        }
        // 跳过空格
        while (i < text.size() && text[i] == ' ') ++i;
    }
    return ids;
}

// BPE：字节级 BPE 编码（gpt2 类）
std::vector<int> Tokenizer::encode_bpe(const std::string& text) const {
    static const auto b2u = build_bytes_to_unicode();
    // 1. 文本字节 → bytes_to_unicode 字符 → 初始 token id
    std::vector<int> ids;
    ids.reserve(text.size());
    for (unsigned char b : text) {
        auto it = b2u.find(b);
        if (it == b2u.end()) return {};
        int t = find_token(it->second);
        if (t < 0) return {};
        ids.push_back(t);
    }

    // 2. BPE 合并（按 merges rank 贪心，merge key 为 "a b" 带空格）
    while (ids.size() > 1) {
        int best_rank = INT32_MAX;
        size_t best_pos = 0;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            std::string pair = tokens_[ids[i]] + " " + tokens_[ids[i + 1]];
            auto it = merges_rank_.find(pair);
            if (it != merges_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }
        if (best_rank == INT32_MAX) break;
        std::string merged = tokens_[ids[best_pos]] + tokens_[ids[best_pos + 1]];
        int t = find_token(merged);
        if (t < 0) break;
        ids[best_pos] = t;
        ids.erase(ids.begin() + best_pos + 1);
    }
    return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> ids = is_gpt2_ ? encode_bpe(text) : encode_sp(text);
    if (add_bos && bos_id_ >= 0) {
        ids.insert(ids.begin(), bos_id_);
    }
    return ids;
}

std::string Tokenizer::decode(int token) const {
    if (token < 0 || token >= (int)tokens_.size()) return "";
    static const auto u2b = build_unicode_to_bytes();
    const std::string& tok = tokens_[token];
    std::string bytes;
    size_t i = 0;
    while (i < tok.size()) {
        uint32_t cp = 0;
        int len = utf8_decode(tok, i, cp);
        auto it = u2b.find(cp);
        if (it != u2b.end()) {
            bytes += (char)it->second;
        } else {
            bytes += tok.substr(i, len);  // 特殊 token（如 <|im_start|>）直接保留
        }
        i += len;
    }
    return bytes;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) out += decode(id);
    return out;
}

}  // namespace cllm
