#include "cllm/server/http_server.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

// 统一的 CORS 响应头片段（允许所有源，便于本地 web 前端直接访问）
const char* CORS_HEADERS =
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";

// 从 OpenAI chat messages 数组中解析 (role, content) 对，保留顺序。
// 极简解析：按 "role":"..." 与 "content":"..." 顺序配对，兼容单个 messages 对象或数组。
std::vector<std::pair<std::string, std::string>>
parse_messages(const std::string& body) {
    std::vector<std::pair<std::string, std::string>> out;
    size_t p = body.find("\"messages\"");
    if (p == std::string::npos) return out;
    size_t arr_start = body.find('[', p);
    size_t arr_end = body.find(']', arr_start == std::string::npos ? p : arr_start);
    if (arr_start == std::string::npos || arr_end == std::string::npos) return out;

    size_t cur = arr_start;
    while (cur < arr_end) {
        size_t r = body.find("\"role\"", cur);
        if (r == std::string::npos || r > arr_end) break;
        size_t rc = body.find(':', r);
        size_t rs = body.find('"', rc + 1);
        size_t re = body.find('"', rs + 1);
        if (rs == std::string::npos || re == std::string::npos) break;
        std::string role = body.substr(rs + 1, re - rs - 1);

        size_t c = body.find("\"content\"", re);
        if (c == std::string::npos || c > arr_end) break;
        size_t cc = body.find(':', c);
        size_t cs = body.find('"', cc + 1);
        if (cs == std::string::npos) break;
        // content 字符串需要处理转义：读到未转义的 "
        std::string content;
        size_t i = cs + 1;
        for (; i < arr_end; ++i) {
            char ch = body[i];
            if (ch == '\\' && i + 1 < arr_end) {
                char n = body[i + 1];
                if (n == 'n') content += '\n';
                else if (n == 't') content += '\t';
                else if (n == 'r') content += '\r';
                else content += n;
                ++i;
            } else if (ch == '"') {
                break;
            } else {
                content += ch;
            }
        }
        out.emplace_back(std::move(role), std::move(content));
        cur = i + 1;
    }
    return out;
}

// 根据路径返回 MIME 类型
std::string guess_mime(const std::string& path) {
    auto ends = [&](const char* ext) {
        size_t n = std::strlen(ext);
        return path.size() >= n && std::memcmp(path.data() + path.size() - n, ext, n) == 0;
    };
    if (ends(".html") || ends(".htm")) return "text/html; charset=utf-8";
    if (ends(".js")) return "application/javascript; charset=utf-8";
    if (ends(".css")) return "text/css; charset=utf-8";
    if (ends(".json")) return "application/json; charset=utf-8";
    if (ends(".svg")) return "image/svg+xml";
    if (ends(".png")) return "image/png";
    return "application/octet-stream";
}

// 读取文件全部内容；失败返回空 string + ok=false
std::string read_file_all(const std::string& path, bool& ok) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { ok = false; return {}; }
    std::ostringstream ss; ss << f.rdbuf();
    ok = true;
    return ss.str();
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
    std::string model_path;   // 当前加载模型文件路径（用于 /v1/model/info）
    std::string web_root;     // 前端静态目录（存在则托管 /、/web/*）
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

        // CORS 预检
        if (method == "OPTIONS") {
            std::string resp =
                std::string("HTTP/1.1 204 No Content\r\n") + CORS_HEADERS +
                "Content-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(client_fd, resp);
            close(client_fd);
            return;
        }

        if (path == "/health") {
            std::string body_s = "{\"status\":\"ok\"}";
            std::string resp =
                std::string("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n") + CORS_HEADERS +
                "Content-Length: " + std::to_string(body_s.size()) +
                "\r\nConnection: close\r\n\r\n" + body_s;
            send_all(client_fd, resp);
        } else if (path == "/v1/model/info" || path == "/v1/models") {
            handle_model_info(client_fd, path);
        } else if (path == "/v1/completions" || path == "/v1/chat/completions") {
            handle_generate(client_fd, body, path);
        } else if (path == "/" || path.rfind("/web/", 0) == 0 || path.rfind("/models.json", 0) == 0) {
            handle_static(client_fd, path);
        } else {
            std::string resp =
                std::string("HTTP/1.1 404 Not Found\r\n") + CORS_HEADERS +
                "Content-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(client_fd, resp);
        }
        close(client_fd);
    }

    // 返回当前加载模型的元信息；/v1/models 走 OpenAI 兼容格式
    void handle_model_info(int client_fd, const std::string& path) {
        const auto& cfg = engine->config();
        // 从 model_path 中提取文件名作为 id
        std::string id = model_path;
        size_t slash = id.find_last_of('/');
        if (slash != std::string::npos) id = id.substr(slash + 1);
        if (id.empty()) id = cfg.arch;

        std::string info =
            "{\"id\":\"" + json_escape(id) + "\","
            "\"path\":\"" + json_escape(model_path) + "\","
            "\"architecture\":\"" + json_escape(cfg.arch) + "\","
            "\"n_layers\":" + std::to_string(cfg.n_layers) + ","
            "\"n_embd\":" + std::to_string(cfg.n_embd) + ","
            "\"n_head\":" + std::to_string(cfg.n_head) + ","
            "\"n_head_kv\":" + std::to_string(cfg.n_head_kv) + ","
            "\"n_ctx\":" + std::to_string(cfg.n_ctx) + ","
            "\"vocab_size\":" + std::to_string(cfg.vocab_size) + ","
            "\"has_chat_template\":" + std::string(tokenizer->has_chat_template() ? "true" : "false") + ","
            "\"chat_template_format\":\"" + json_escape(tokenizer->chat_template_format()) + "\"}";

        std::string body_s;
        if (path == "/v1/models") {
            // OpenAI 兼容：{ "object":"list", "data":[ {...} ] }
            body_s = "{\"object\":\"list\",\"data\":[" + info + "]}";
        } else {
            body_s = info;
        }
        std::string resp =
            std::string("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n") + CORS_HEADERS +
            "Content-Length: " + std::to_string(body_s.size()) +
            "\r\nConnection: close\r\n\r\n" + body_s;
        send_all(client_fd, resp);
    }

    // 静态文件托管：仅服务 web_root 下白名单路径，避免目录穿越
    void handle_static(int client_fd, const std::string& path) {
        if (web_root.empty()) {
            std::string resp =
                std::string("HTTP/1.1 404 Not Found\r\n") + CORS_HEADERS +
                "Content-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(client_fd, resp);
            return;
        }
        // 规整请求路径：/ -> index.html；/web/x -> x
        std::string rel = (path == "/") ? "index.html" : path.substr(1);
        if (rel.rfind("web/", 0) == 0) rel = rel.substr(4);
        // 禁止路径穿越
        if (rel.find("..") != std::string::npos) {
            std::string resp =
                std::string("HTTP/1.1 403 Forbidden\r\n") + CORS_HEADERS +
                "Content-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(client_fd, resp);
            return;
        }
        bool ok = false;
        std::string body_s = read_file_all(web_root + "/" + rel, ok);
        if (!ok) {
            std::string resp =
                std::string("HTTP/1.1 404 Not Found\r\n") + CORS_HEADERS +
                "Content-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(client_fd, resp);
            return;
        }
        std::string resp =
            "HTTP/1.1 200 OK\r\nContent-Type: " + guess_mime(rel) + "\r\n" +
            CORS_HEADERS +
            "Content-Length: " + std::to_string(body_s.size()) +
            "\r\nConnection: close\r\n\r\n" + body_s;
        send_all(client_fd, resp);
    }

    // 从请求体中提取对话输入
    struct ChatInput {
        std::string system;  // system 消息（若存在）
        std::string user;    // 最后一条 user 消息（主请求）
        std::vector<std::pair<std::string, std::string>> history;  // 完整有序历史
    };

    ChatInput extract_chat(const std::string& body, const std::string& path) {
        ChatInput ci;
        if (path == "/v1/chat/completions") {
            ci.history = parse_messages(body);
            for (const auto& m : ci.history) {
                if (m.first == "system" && ci.system.empty()) ci.system = m.second;
                if (m.first == "user") ci.user = m.second;  // 取最后一条 user
            }
            // 兜底：若未解析到任何 user 消息，拼接所有 content
            if (ci.user.empty()) {
                for (const auto& m : ci.history)
                    if (m.first != "system") ci.user += m.second + "\n";
            }
        } else {
            ci.user = json_get_string(body, "prompt");
        }
        return ci;
    }

    void handle_generate(int client_fd, const std::string& body, const std::string& path) {
        ChatInput ci = extract_chat(body, path);

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

        // 生成（加锁保护引擎）。
        // chat 模式下，若有多轮历史，拼接为 "user: xxx\nassistant: yyy\n..." 后交给模板，
        // 以尽可能保留上下文（真正的多轮模板渲染需要 Jinja，当前为近似实现）。
        std::string user_prompt = ci.user;
        if (path == "/v1/chat/completions" && ci.history.size() > 1) {
            std::string hist;
            for (size_t i = 0; i + 1 < ci.history.size(); ++i) {
                const auto& m = ci.history[i];
                if (m.first == "system") continue;
                hist += m.first + ": " + m.second + "\n";
            }
            if (!hist.empty()) user_prompt = hist + "user: " + ci.user;
        }
        std::vector<int> ids;
        if (tokenizer->has_chat_template()) {
            ids = tokenizer->apply_chat_template(user_prompt, ci.system);
        } else {
            ids = tokenizer->encode(user_prompt, tokenizer->add_bos_token());
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
                std::string("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n") +
                CORS_HEADERS +
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

            // chat 模式下返回 OpenAI chat.completion 结构，便于前端复用
            std::string content;
            if (path == "/v1/chat/completions") {
                content =
                    "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\","
                    "\"choices\":[{\"index\":0,\"finish_reason\":\"stop\","
                    "\"message\":{\"role\":\"assistant\",\"content\":\"" + json_escape(full_text) +
                    "\"}}]}";
            } else {
                content =
                    "{\"id\":\"cmpl-1\",\"object\":\"text_completion\","
                    "\"choices\":[{\"text\":\"" + json_escape(full_text) +
                    "\",\"index\":0,\"finish_reason\":\"stop\"}]}";
            }
            std::string resp =
                std::string("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n") + CORS_HEADERS +
                "Content-Length: " + std::to_string(content.size()) +
                "\r\nConnection: close\r\n\r\n" + content;
            send_all(client_fd, resp);
        }
    }
};

HttpServer::HttpServer(TransformerEngine& engine, Tokenizer& tokenizer, uint16_t port,
                       std::string model_path)
    : impl_(std::make_unique<Impl>()) {
    impl_->engine = &engine;
    impl_->tokenizer = &tokenizer;
    impl_->port = port;
    impl_->model_path = std::move(model_path);
    // 自动探测前端目录：优先 ./web；若 CLLM_WEB_ROOT 环境变量存在则使用之
    if (const char* env = std::getenv("CLLM_WEB_ROOT")) {
        impl_->web_root = env;
    } else {
        std::ifstream probe("web/index.html");
        if (probe) impl_->web_root = "web";
    }
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
