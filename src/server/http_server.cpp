#include "cllm/server/http_server.hpp"

#include <atomic>
#include <cctype>
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

std::string json_unescape(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[++i]) {
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                default: o += s[i]; break;
            }
        } else {
            o += s[i];
        }
    }
    return o;
}

void json_get_stop(const std::string& json, std::vector<std::string>& out) {
    const std::string needle = "\"stop\"";
    size_t p = json.find(needle);
    if (p == std::string::npos) return;
    p = json.find(':', p + needle.size());
    if (p == std::string::npos) return;
    ++p;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n')) ++p;

    auto parse_str = [&](size_t& i) -> std::string {
        if (i >= json.size() || json[i] != '"') return "";
        ++i;
        std::string raw;
        while (i < json.size()) {
            if (json[i] == '\\' && i + 1 < json.size()) {
                raw += json[i];
                raw += json[i + 1];
                i += 2;
                continue;
            }
            if (json[i] == '"') {
                ++i;
                break;
            }
            raw += json[i++];
        }
        return json_unescape(raw);
    };

    if (p < json.size() && json[p] == '"') {
        std::string s = parse_str(p);
        if (!s.empty()) out.push_back(std::move(s));
    } else if (p < json.size() && json[p] == '[') {
        ++p;
        while (p < json.size() && json[p] != ']') {
            while (p < json.size() && (json[p] == ' ' || json[p] == '\t' ||
                                       json[p] == '\n' || json[p] == ',')) ++p;
            if (p < json.size() && json[p] == '"') {
                std::string s = parse_str(p);
                if (!s.empty()) out.push_back(std::move(s));
            } else {
                break;
            }
        }
    }
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

size_t json_skip_ws(const std::string& s, size_t i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    return i;
}

size_t json_skip_string(const std::string& s, size_t i) {
    if (i >= s.size() || s[i] != '"') return i;
    ++i;
    while (i < s.size()) {
        if (s[i] == '\\' && i + 1 < s.size()) { i += 2; continue; }
        if (s[i] == '"') return i + 1;
        ++i;
    }
    return i;
}

size_t json_match(const std::string& s, size_t open) {
    if (open >= s.size()) return open;
    const char oc = s[open];
    const char cc = (oc == '{') ? '}' : (oc == '[' ? ']' : 0);
    if (!cc) return open;
    int depth = 0;
    for (size_t i = open; i < s.size(); ) {
        if (s[i] == '"') { i = json_skip_string(s, i); continue; }
        if (s[i] == oc) ++depth;
        else if (s[i] == cc && --depth == 0) return i + 1;
        ++i;
    }
    return s.size();
}

bool json_read_string_at(const std::string& s, size_t quote, std::string& out, size_t& end) {
    if (quote >= s.size() || s[quote] != '"') return false;
    out.clear();
    size_t i = quote + 1;
    for (; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char n = s[i + 1];
            if (n == 'n') out += '\n';
            else if (n == 't') out += '\t';
            else if (n == 'r') out += '\r';
            else out += n;
            ++i;
        } else if (s[i] == '"') {
            end = i + 1;
            return true;
        } else {
            out += s[i];
        }
    }
    return false;
}

std::string flatten_content_value(const std::string& s, size_t colon) {
    size_t i = json_skip_ws(s, colon + 1);
    if (i >= s.size()) return "";
    if (s[i] == '"') {
        std::string text;
        size_t end = i;
        if (json_read_string_at(s, i, text, end)) return text;
        return "";
    }
    if (s[i] != '[') return "";
    const size_t arr_end = json_match(s, i);
    std::string text;
    size_t cur = i + 1;
    while (cur < arr_end) {
        cur = json_skip_ws(s, cur);
        if (cur >= arr_end || s[cur] == ']') break;
        if (s[cur] == '{') {
            const size_t obj_end = json_match(s, cur);
            const std::string obj = s.substr(cur, obj_end - cur);
            size_t tp = obj.find("\"type\"");
            std::string typ;
            if (tp != std::string::npos) {
                size_t c = obj.find(':', tp);
                size_t q = obj.find('"', c == std::string::npos ? 0 : c + 1);
                size_t e = q;
                json_read_string_at(obj, q, typ, e);
            }
            if (typ == "image_url" || obj.find("\"image_url\"") != std::string::npos) {
                if (!text.empty()) text += "\n";
                text += "[用户附带了一张图片；当前引擎按文本推理，看不到像素。]";
            } else {
                size_t tx = obj.find("\"text\"");
                if (tx != std::string::npos) {
                    size_t c = obj.find(':', tx);
                    size_t q = (c == std::string::npos) ? std::string::npos : json_skip_ws(obj, c + 1);
                    std::string part;
                    size_t e = 0;
                    if (q != std::string::npos && json_read_string_at(obj, q, part, e) && !part.empty()) {
                        if (!text.empty()) text += "\n";
                        text += part;
                    }
                }
            }
            cur = obj_end;
        } else if (s[cur] == '"') {
            std::string part;
            if (json_read_string_at(s, cur, part, cur) && !part.empty()) {
                if (!text.empty()) text += "\n";
                text += part;
            }
        } else {
            ++cur;
        }
        cur = json_skip_ws(s, cur);
        if (cur < arr_end && s[cur] == ',') ++cur;
    }
    return text;
}

std::string extract_tools_prompt(const std::string& body) {
    size_t p = body.find("\"tools\"");
    if (p == std::string::npos) return "";
    size_t colon = body.find(':', p + 7);
    if (colon == std::string::npos) return "";
    size_t arr = json_skip_ws(body, colon + 1);
    if (arr >= body.size() || body[arr] != '[') return "";
    size_t end = json_match(body, arr);
    if (end <= arr + 2) return "";
    const std::string tools = body.substr(arr, end - arr);
    if (tools.find('{') == std::string::npos) return "";
    return std::string(
        "\n\n# Tools\n"
        "你可以调用下列工具。需要调用时立刻输出，不要思考过程：\n"
        "<tool_call>\n{\"name\":\"工具名\",\"arguments\":{...}}\n</tool_call>\n"
        "不要编造工具结果。可用工具 JSON：\n") + tools + "\n";
}

struct ParsedToolCall {
    std::string name;
    std::string arguments;
};

void json_trim(std::string& s) {
    size_t a = 0;
    while (a < s.size() && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r')) ++a;
    size_t b = s.size();
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r')) --b;
    s = s.substr(a, b - a);
}

std::vector<ParsedToolCall> parse_tool_calls(const std::string& text) {
    std::vector<ParsedToolCall> out;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t a = text.find("<tool_call>", pos);
        if (a == std::string::npos) break;
        size_t b = text.find("</tool_call>", a + 11);
        if (b == std::string::npos) break;
        std::string inner = text.substr(a + 11, b - (a + 11));
        json_trim(inner);
        ParsedToolCall tc;
        size_t brace = inner.find('{');
        if (brace != std::string::npos) {
            const std::string obj = inner.substr(brace);
            size_t np = obj.find("\"name\"");
            if (np != std::string::npos) {
                size_t c = obj.find(':', np);
                size_t q = (c == std::string::npos) ? std::string::npos : json_skip_ws(obj, c + 1);
                size_t e = 0;
                if (q != std::string::npos) json_read_string_at(obj, q, tc.name, e);
            }
            size_t ap = obj.find("\"arguments\"");
            if (ap == std::string::npos) ap = obj.find("\"parameters\"");
            if (ap != std::string::npos) {
                size_t c = obj.find(':', ap);
                size_t v = (c == std::string::npos) ? std::string::npos : json_skip_ws(obj, c + 1);
                if (v != std::string::npos && v < obj.size()) {
                    if (obj[v] == '{') {
                        const size_t end = json_match(obj, v);
                        tc.arguments = obj.substr(v, end - v);
                    } else if (obj[v] == '"') {
                        json_read_string_at(obj, v, tc.arguments, v);
                    }
                }
            }
        }
        if (tc.name.empty()) {
            size_t line_end = inner.find('\n');
            tc.name = (line_end == std::string::npos) ? inner : inner.substr(0, line_end);
            json_trim(tc.name);
        }
        if (tc.arguments.empty()) tc.arguments = "{}";
        if (!tc.name.empty()) out.push_back(std::move(tc));
        pos = b + 12;
    }
    return out;
}

std::string tool_calls_json(const std::vector<ParsedToolCall>& calls) {
    std::string s = "[";
    for (size_t i = 0; i < calls.size(); ++i) {
        if (i) s += ",";
        s += "{\"id\":\"call_" + std::to_string(i) + "\",\"type\":\"function\",\"function\":{\"name\":\"" +
             json_escape(calls[i].name) + "\",\"arguments\":\"" + json_escape(calls[i].arguments) + "\"}}";
    }
    s += "]";
    return s;
}

// 从 OpenAI chat messages 解析 (role, content)。content 可以是字符串或多模态数组。
std::vector<std::pair<std::string, std::string>>
parse_messages(const std::string& body) {
    std::vector<std::pair<std::string, std::string>> out;
    size_t p = body.find("\"messages\"");
    if (p == std::string::npos) return out;
    size_t colon = body.find(':', p + 10);
    if (colon == std::string::npos) return out;
    size_t arr_start = json_skip_ws(body, colon + 1);
    if (arr_start >= body.size() || body[arr_start] != '[') return out;
    const size_t arr_end = json_match(body, arr_start);
    size_t cur = arr_start + 1;
    while (cur < arr_end) {
        cur = json_skip_ws(body, cur);
        if (cur >= arr_end || body[cur] == ']') break;
        if (body[cur] != '{') { ++cur; continue; }
        const size_t obj_end = json_match(body, cur);
        const std::string obj = body.substr(cur, obj_end - cur);
        size_t rp = obj.find("\"role\"");
        std::string role;
        if (rp != std::string::npos) {
            size_t c = obj.find(':', rp);
            size_t q = (c == std::string::npos) ? std::string::npos : json_skip_ws(obj, c + 1);
            size_t e = 0;
            if (q != std::string::npos) json_read_string_at(obj, q, role, e);
        }
        std::string content;
        size_t cp = obj.find("\"content\"");
        if (cp != std::string::npos) {
            size_t c = obj.find(':', cp);
            if (c != std::string::npos) content = flatten_content_value(obj, c);
        }
        if (role == "tool" && content.find("[tool]") != 0) {
            content = std::string("[tool result]\n") + content;
        }
        if (!role.empty()) out.emplace_back(std::move(role), std::move(content));
        cur = json_skip_ws(body, obj_end);
        if (cur < arr_end && body[cur] == ',') ++cur;
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

constexpr int kMaxBodyBytes = 12 * 1024 * 1024;

// 发送全部数据；MSG_NOSIGNAL 避免客户端断开时 SIGPIPE 杀进程
bool send_all(int fd, const std::string& data) {
    size_t off = 0;
    while (off < data.size()) {
        ssize_t n = send(fd, data.data() + off, data.size() - off, MSG_NOSIGNAL);
        if (n <= 0) return false;
        off += n;
    }
    return true;
}

std::string header_name_lower(const std::string& line) {
    size_t colon = line.find(':');
    if (colon == std::string::npos) return {};
    std::string key = line.substr(0, colon);
    for (char& c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return key;
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

    ~Impl() {
        running = false;
        if (listen_fd >= 0) { shutdown(listen_fd, SHUT_RDWR); close(listen_fd); }
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
            if (header_name_lower(line) == "content-length") {
                size_t colon = line.find(':');
                content_length = std::atoi(line.c_str() + colon + 1);
            }
        }

        if (content_length > kMaxBodyBytes) {
            std::string resp =
                std::string("HTTP/1.1 413 Payload Too Large\r\n") + CORS_HEADERS +
                "Content-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(client_fd, resp);
            close(client_fd);
            return;
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
            "\"chat_template_format\":\"" + json_escape(tokenizer->chat_template_format()) + "\","
            "\"capabilities\":{\"vision\":false,\"tools\":true}}";

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

    void send_error(int client_fd, int code, const std::string& msg) {
        std::string status = (code == 400) ? "400 Bad Request" : "500 Internal Server Error";
        std::string body_s = "{\"error\":\"" + json_escape(msg) + "\"}";
        std::string resp =
            "HTTP/1.1 " + status + "\r\nContent-Type: application/json\r\n" + CORS_HEADERS +
            "Content-Length: " + std::to_string(body_s.size()) +
            "\r\nConnection: close\r\n\r\n" + body_s;
        send_all(client_fd, resp);
    }

    void handle_generate(int client_fd, const std::string& body, const std::string& path) {
        ChatInput ci = extract_chat(body, path);

        SampleParams sp;
        double d;
        if (json_get_number(body, "temperature", d)) sp.temperature = (float)d;
        if (json_get_number(body, "top_p", d)) sp.top_p = (float)d;
        if (json_get_number(body, "top_k", d)) sp.top_k = (int)d;
        int max_tokens = 64;
        if (json_get_number(body, "max_tokens", d)) max_tokens = (int)d;
        bool stream = false;
        json_get_bool(body, "stream", stream);
        bool has_seed = json_get_number(body, "seed", d);
        const uint32_t seed = has_seed ? (uint32_t)d : 0;
        std::vector<std::string> stops;
        json_get_stop(body, stops);

        const std::string tools_prompt = extract_tools_prompt(body);
        if (!tools_prompt.empty()) {
            bool injected = false;
            for (auto& m : ci.history) {
                if (m.first == "system") {
                    m.second += tools_prompt;
                    injected = true;
                    break;
                }
            }
            if (!injected) {
                ci.history.insert(ci.history.begin(), {"system", "You are a helpful assistant." + tools_prompt});
            }
            if (ci.system.empty()) ci.system = "You are a helpful assistant.";
            ci.system += tools_prompt;
            for (auto it = ci.history.rbegin(); it != ci.history.rend(); ++it) {
                if (it->first == "user") {
                    if (it->second.find("/no_think") == std::string::npos) it->second += "\n/no_think";
                    break;
                }
            }
        }

        std::vector<int> ids;
        if (path == "/v1/chat/completions" && !ci.history.empty() && tokenizer->has_chat_template()) {
            std::vector<ChatMessage> msgs;
            msgs.reserve(ci.history.size());
            for (const auto& m : ci.history) msgs.push_back({m.first, m.second});
            ids = tokenizer->apply_chat_template(msgs);
        } else if (tokenizer->has_chat_template()) {
            ids = tokenizer->apply_chat_template(ci.user, ci.system);
        } else {
            ids = tokenizer->encode(ci.user, tokenizer->add_bos_token());
        }
        if (ids.empty()) {
            send_error(client_fd, 400, "empty prompt");
            return;
        }

        const int prompt_tokens = (int)ids.size();
        std::lock_guard<std::mutex> lock(engine_mtx);
        std::mt19937 rng;
        if (has_seed) rng.seed(seed);
        else rng.seed(std::random_device{}());

        std::vector<float> logits;
        try {
            logits = engine->prefill(ids);
        } catch (const std::exception& e) {
            send_error(client_fd, 400, e.what());
            return;
        }

        auto usage_json = [&](int completion_tokens) {
            return std::string("\"usage\":{\"prompt_tokens\":") +
                   std::to_string(prompt_tokens) + ",\"completion_tokens\":" +
                   std::to_string(completion_tokens) + ",\"total_tokens\":" +
                   std::to_string(prompt_tokens + completion_tokens) + "}";
        };

        if (stream) {
            std::string head =
                std::string("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n") +
                CORS_HEADERS +
                "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
            if (!send_all(client_fd, head)) return;

            Utf8StreamDecoder dec;
            int n_gen = 0;
            std::string generated;
            for (int i = 0; i < max_tokens; ++i) {
                int next = Sampler::sample(logits, sp, rng);
                if (tokenizer->is_eos(next)) break;
                ids.push_back(next);
                ++n_gen;
                std::string piece = dec.feed(tokenizer->decode(next));
                generated += piece;
                if (!piece.empty()) {
                    std::string data =
                        "data: {\"choices\":[{\"delta\":{\"content\":\"" +
                        json_escape(piece) + "\"}}]}\n\n";
                    if (!send_all(client_fd, data)) return;
                }
                std::string stripped;
                if (consume_stop_suffix(generated, stops, stripped)) break;
                logits = engine->forward(next);
            }
            std::string tail = dec.flush();
            generated += tail;
            if (!tail.empty()) {
                std::string data =
                    "data: {\"choices\":[{\"delta\":{\"content\":\"" +
                    json_escape(tail) + "\"}}]}\n\n";
                if (!send_all(client_fd, data)) return;
            }
            const auto calls = parse_tool_calls(generated);
            if (!calls.empty()) {
                const std::string tc =
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":" + tool_calls_json(calls) +
                    "},\"finish_reason\":\"tool_calls\"}]}\n\n";
                if (!send_all(client_fd, tc)) return;
            }
            send_all(client_fd, "data: {" + usage_json(n_gen) + "}\n\n");
            send_all(client_fd, "data: [DONE]\n\n");
        } else {
            Utf8StreamDecoder dec;
            std::string full_text;
            int n_gen = 0;
            for (int i = 0; i < max_tokens; ++i) {
                int next = Sampler::sample(logits, sp, rng);
                if (tokenizer->is_eos(next)) break;
                ids.push_back(next);
                ++n_gen;
                full_text += dec.feed(tokenizer->decode(next));
                std::string stripped;
                if (consume_stop_suffix(full_text, stops, stripped)) {
                    full_text = std::move(stripped);
                    break;
                }
                logits = engine->forward(next);
            }
            full_text += dec.flush();
            {
                std::string stripped;
                if (consume_stop_suffix(full_text, stops, stripped)) full_text = std::move(stripped);
            }

            std::string content;
            if (path == "/v1/chat/completions") {
                const auto calls = parse_tool_calls(full_text);
                const bool has_tools = !calls.empty();
                content =
                    "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\","
                    "\"choices\":[{\"index\":0,\"finish_reason\":\"" +
                    std::string(has_tools ? "tool_calls" : "stop") +
                    "\",\"message\":{\"role\":\"assistant\",\"content\":\"" + json_escape(full_text) + "\"";
                if (has_tools) content += ",\"tool_calls\":" + tool_calls_json(calls);
                content += "}}]," + usage_json(n_gen) + "}";
            } else {
                content =
                    "{\"id\":\"cmpl-1\",\"object\":\"text_completion\","
                    "\"choices\":[{\"text\":\"" + json_escape(full_text) +
                    "\",\"index\":0,\"finish_reason\":\"stop\"}]," + usage_json(n_gen) + "}";
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
        std::thread([this, cfd] { impl_->handle_connection(cfd); }).detach();
    }
}

}  // namespace cllm
