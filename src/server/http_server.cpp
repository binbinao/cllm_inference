#include "cllm/server/http_server.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace cllm {

namespace {

// ---- 极简 JSON 值提取（仅用于解析 OpenAI 请求体） ----

// 在 json 文本中查找 key，返回其字符串值（去掉引号），找不到返回空
std::string json_get_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t p = json.find(needle);
    if (p == std::string::npos) return "";
    p = json.find(':', p + needle.size());
    if (p == std::string::npos) return "";
    p = json.find('"', p + 1);
    if (p == std::string::npos) return "";
    size_t q = json.find('"', p + 1);
    if (q == std::string::npos) return "";
    return json.substr(p + 1, q - p - 1);
}

// 查找 key 的数值（int/float），返回 found 标志
bool json_get_number(const std::string& json, const std::string& key, double& out) {
    std::string needle = "\"" + key + "\"";
    size_t p = json.find(needle);
    if (p == std::string::npos) return false;
    p = json.find(':', p + needle.size());
    if (p == std::string::npos) return false;
    // 跳过空白
    ++p;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) ++p;
    char* end = nullptr;
    out = std::strtod(json.c_str() + p, &end);
    return end != json.c_str() + p;
}

// 查找 key 的布尔值
bool json_get_bool(const std::string& json, const std::string& key, bool& out) {
    std::string needle = "\"" + key + "\"";
    size_t p = json.find(needle);
    if (p == std::string::npos) return false;
    p = json.find(':', p + needle.size());
    if (p == std::string::npos) return false;
    size_t t = json.find("true", p);
    size_t f = json.find("false", p);
    if (t != std::string::npos && (f == std::string::npos || t < f)) { out = true; return true; }
    if (f != std::string::npos) { out = false; return true; }
    return false;
}

// JSON 字符串转义
std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default: o += c;
        }
    }
    return o;
}

// 读取一行（以 \r\n 或 \n 结尾）
std::string read_line(int fd) {
    std::string line;
    char c;
    while (read(fd, &c, 1) == 1) {
        if (c == '\n') break;
        if (c != '\r') line += c;
    }
    return line;
}

// 发送全部数据
bool send_all(int fd, const std::string& data) {
    size_t off = 0;
    while (off < data.size()) {
        ssize_t n = write(fd, data.data() + off, data.size() - off);
        if (n <= 0) return false;
        off += n;
    }
    return true;
}

}  // namespace

struct HttpServer::Impl {
    TransformerEngine* engine;
    Tokenizer* tokenizer;
    uint16_t port;
    int listen_fd = -1;
    std::atomic<bool> running{false};
    std::mutex engine_mtx;   // 保护引擎推理（KV Cache 共享状态）
    std::vector<std::thread> threads;

    ~Impl() {
        running = false;
        if (listen_fd >= 0) { shutdown(listen_fd, SHUT_RDWR); close(listen_fd); }
        for (auto& t : threads) if (t.joinable()) t.join();
    }

    // 处理单个连接
    void handle_connection(int client_fd) {
        std::string request_line = read_line(client_fd);
        if (request_line.empty()) { close(client_fd); return; }

        // 解析请求行：METHOD PATH HTTP/1.1
        std::istringstream rl(request_line);
        std::string method, path, ver;
        rl >> method >> path >> ver;

        // 读取 header（跳过，只关心 Content-Length）
        int content_length = 0;
        std::string line;
        while (!(line = read_line(client_fd)).empty()) {
            if (line.rfind("Content-Length:", 0) == 0) {
                content_length = std::atoi(line.c_str() + 15);
            }
        }

        // 读取 body
        std::string body;
        if (content_length > 0) {
            body.resize(content_length);
            size_t off = 0;
            while (off < (size_t)content_length) {
                ssize_t n = read(client_fd, &body[off], content_length - off);
                if (n <= 0) break;
                off += n;
            }
        }

        if (path == "/health") {
            std::string resp =
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                "Content-Length: 15\r\nConnection: close\r\n\r\n"
                "{\"status\":\"ok\"}";
            send_all(client_fd, resp);
        } else if (path == "/v1/completions" || path == "/v1/chat/completions") {
            handle_generate(client_fd, body, path);
        } else {
            std::string resp =
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(client_fd, resp);
        }
        close(client_fd);
    }

    // 从请求体中提取 prompt（兼容 completions 与 chat）
    std::string extract_prompt(const std::string& body, const std::string& path) {
        if (path == "/v1/chat/completions") {
            // chat 模式：messages 数组中 content 字段，取最后一个 user 内容
            std::string prompt;
            // 简化：提取所有 "content":"..." 并拼接（以换行分隔）
            size_t p = 0;
            std::string key = "\"content\"";
            while ((p = body.find(key, p)) != std::string::npos) {
                size_t q = body.find(':', p + key.size());
                if (q == std::string::npos) break;
                size_t s = body.find('"', q + 1);
                if (s == std::string::npos) break;
                size_t e = body.find('"', s + 1);
                if (e == std::string::npos) break;
                if (!prompt.empty()) prompt += "\n";
                prompt += body.substr(s + 1, e - s - 1);
                p = e + 1;
            }
            return prompt;
        }
        return json_get_string(body, "prompt");
    }

    void handle_generate(int client_fd, const std::string& body, const std::string& path) {
        std::string prompt = extract_prompt(body, path);

        // 解析采样参数
        SampleParams sp;
        double d;
        if (json_get_number(body, "temperature", d)) sp.temperature = (float)d;
        if (json_get_number(body, "top_p", d)) sp.top_p = (float)d;
        if (json_get_number(body, "top_k", d)) sp.top_k = (int)d;
        int max_tokens = 64;
        if (json_get_number(body, "max_tokens", d)) max_tokens = (int)d;
        bool stream = false;
        json_get_bool(body, "stream", stream);

        // 生成（加锁保护引擎）
        std::vector<int> ids;
        if (tokenizer->has_chat_template()) {
            ids = tokenizer->apply_chat_template(prompt);
        } else {
            ids = tokenizer->encode(prompt, tokenizer->add_bos_token());
        }

        std::lock_guard<std::mutex> lock(engine_mtx);
        engine->reset_kv_cache();
        std::mt19937 rng(std::random_device{}());

        // 先 forward 整个 prompt
        std::vector<float> logits;
        for (int id : ids) logits = engine->forward(id);

        if (stream) {
            // SSE 流式响应
            std::string head =
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
            send_all(client_fd, head);

            for (int i = 0; i < max_tokens; ++i) {
                int next = Sampler::sample(logits, sp, rng);
                if (next == tokenizer->eos_id()) break;
                ids.push_back(next);

                std::string piece = tokenizer->decode(next);
                std::string data =
                    "data: {\"choices\":[{\"delta\":{\"content\":\"" +
                    json_escape(piece) + "\"}}]}\n\n";
                send_all(client_fd, data);

                logits = engine->forward(next);
            }
            send_all(client_fd, "data: [DONE]\n\n");
        } else {
            // 非流式：一次性生成并返回 JSON
            std::string full_text;
            for (int i = 0; i < max_tokens; ++i) {
                int next = Sampler::sample(logits, sp, rng);
                if (next == tokenizer->eos_id()) break;
                ids.push_back(next);
                full_text += tokenizer->decode(next);
                logits = engine->forward(next);
            }

            std::string content =
                "{\"id\":\"cmpl-1\",\"object\":\"text_completion\","
                "\"choices\":[{\"text\":\"" + json_escape(full_text) +
                "\",\"index\":0,\"finish_reason\":\"stop\"}]}";
            std::string resp =
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                "Content-Length: " + std::to_string(content.size()) +
                "\r\nConnection: close\r\n\r\n" + content;
            send_all(client_fd, resp);
        }
    }
};

HttpServer::HttpServer(TransformerEngine& engine, Tokenizer& tokenizer, uint16_t port)
    : impl_(std::make_unique<Impl>()) {
    impl_->engine = &engine;
    impl_->tokenizer = &tokenizer;
    impl_->port = port;
}

HttpServer::~HttpServer() = default;

void HttpServer::stop() {
    if (impl_->running) {
        impl_->running = false;
        if (impl_->listen_fd >= 0) shutdown(impl_->listen_fd, SHUT_RDWR);
    }
}

void HttpServer::start() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket failed");
    impl_->listen_fd = fd;

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(impl_->port);
    if (bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) throw std::runtime_error("bind failed");
    if (listen(fd, 128) < 0) throw std::runtime_error("listen failed");

    impl_->running = true;
    while (impl_->running) {
        sockaddr_in client{};
        socklen_t len = sizeof(client);
        int cfd = accept(fd, (sockaddr*)&client, &len);
        if (cfd < 0) {
            if (!impl_->running) break;
            continue;
        }
        impl_->threads.emplace_back([this, cfd] { impl_->handle_connection(cfd); });
    }
}

}  // namespace cllm
