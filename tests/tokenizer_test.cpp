// Tokenizer：哈希查找、多轮 ChatML、多 EOS、UTF-8 流式缓冲。
#include "cllm/core/tokenizer.hpp"

#include <cstdio>
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

GgufModel make_chatml_model() {
    GgufModel m;
    m.tokenizer_model = "llama";  // SentencePiece 路径：整词 / 逐字符
    m.add_bos_token = false;
    m.bos_id = 0;
    m.eos_id = 1;
    m.chat_template = "<|im_start|>";
    // 特殊 token + 测试用整词 + 单字符（供 encode_sp 回退）
    m.tokens = {
        "<s>", "</s>",
        "<|im_start|>", "<|im_end|>",
        "<|eot_id|>", "<|end_of_text|>",
        "system", "user", "assistant",
        "sys", "hello", "world", "again",
        "\n", " ",
    };
    const char* extra = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?";
    for (const char* p = extra; *p; ++p) {
        m.tokens.push_back(std::string(1, *p));
    }
    return m;
}

void test_encode_known_word() {
    std::printf("[encode lookup]\n");
    GgufModel m = make_chatml_model();
    Tokenizer tok(m);
    auto ids = tok.encode("hello", false);
    CHECK(ids.size() == 1, "hello encodes to one token");
    CHECK(!ids.empty() && tok.decode(ids[0]) == "hello", "roundtrip hello");
    auto miss = tok.encode("zzznotinvocab", false);
    // 逐字符回退：z 在词表中，未知整词不会丢崩
    CHECK(!miss.empty(), "unknown word falls back to chars");
}

void test_is_eos() {
    std::printf("[is_eos]\n");
    GgufModel m = make_chatml_model();
    Tokenizer tok(m);
    CHECK(tok.is_eos(tok.eos_id()), "eos_id is stop");
    // <|im_end|> 是 tokens[3]
    CHECK(tok.is_eos(3), "im_end is stop");
    CHECK(tok.is_eos(4), "eot_id is stop");
    CHECK(tok.is_eos(5), "end_of_text is stop");
    CHECK(!tok.is_eos(2), "im_start is not stop");
}

void test_multiturn_chatml() {
    std::printf("[multiturn chatml]\n");
    GgufModel m = make_chatml_model();
    Tokenizer tok(m);
    std::vector<ChatMessage> msgs = {
        {"system", "sys"},
        {"user", "hello"},
        {"assistant", "world"},
        {"user", "again"},
    };
    auto ids = tok.apply_chat_template(msgs);
    int im_s = 2, im_e = 3;
    int n_start = 0, n_end = 0;
    for (int id : ids) {
        if (id == im_s) ++n_start;
        if (id == im_e) ++n_end;
    }
    // system + user + assistant + user + 打开的 assistant 头 = 5 个 im_start
    // 已闭合的 4 条消息 = 4 个 im_end
    CHECK(n_start == 5, "five im_start (4 messages + open assistant)");
    CHECK(n_end == 4, "four im_end (closed messages only)");
    CHECK(!ids.empty() && ids.front() == im_s, "starts with im_start");
}

void test_single_turn_compat() {
    std::printf("[single-turn compat]\n");
    GgufModel m = make_chatml_model();
    Tokenizer tok(m);
    auto a = tok.apply_chat_template("hello", "sys");
    auto b = tok.apply_chat_template(std::vector<ChatMessage>{{"system", "sys"}, {"user", "hello"}});
    CHECK(a == b, "single-turn overload matches messages API");
}

void test_utf8_stream() {
    std::printf("[utf8 stream]\n");
    Utf8StreamDecoder dec;
    const std::string ni = "你";  // E4 BD A0
    CHECK(ni.size() == 3, "你 is 3 bytes");
    std::string p1 = dec.feed(std::string_view(ni.data(), 1));
    CHECK(p1.empty(), "incomplete UTF-8 held back");
    std::string p2 = dec.feed(std::string_view(ni.data() + 1, 2));
    CHECK(p2 == ni, "complete codepoint flushed");
    std::string p3 = dec.feed("A");
    CHECK(p3 == "A", "ascii passes through");
    Utf8StreamDecoder dec2;
    dec2.feed(std::string_view(ni.data(), 2));
    std::string leftover = dec2.flush();
    CHECK(leftover.size() == 2, "flush emits remainder");
}

}  // namespace

int main() {
    test_encode_known_word();
    test_is_eos();
    test_multiturn_chatml();
    test_single_turn_compat();
    test_utf8_stream();
    std::printf("==== tokenizer_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
