#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "cllm/core/engine.hpp"
#include "cllm/core/sampler.hpp"
#include "cllm/core/tokenizer.hpp"
#include "cllm/core/thread_pool.hpp"

namespace cllm {

// 纯 C++ 标准库自研的 HTTP 推理服务器（OpenAI 兼容 + SSE 流式）
class HttpServer {
public:
    HttpServer(TransformerEngine& engine, Tokenizer& tokenizer, uint16_t port);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // 阻塞监听循环，处理请求直到服务停止
    void start();

    // 停止服务（可从其他线程调用）
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cllm
