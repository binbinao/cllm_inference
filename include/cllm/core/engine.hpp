#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "cllm/core/gguf.hpp"
#include "cllm/core/thread_pool.hpp"

namespace cllm {

// 通用 Transformer 推理引擎（RMSNorm + RoPE + GQA Attention + SwiGLU FFN + KV Cache）
// 通过 ModelConfig 参数化，覆盖 Llama/Qwen/Mistral 同族架构。
class TransformerEngine {
public:
    // use_quant=true（默认）：量化权重零拷贝引用 mmap，forward 走整数点积快速路径。
    // use_quant=false：将所有量化权重反量化为 fp32 常驻内存，forward 走纯 float 矩阵乘，
    //                 用于「量化 vs fp32」的公平性能对比。
    TransformerEngine(const GgufModel& model, ThreadPool& pool, bool use_quant = true);
    ~TransformerEngine();

    TransformerEngine(const TransformerEngine&) = delete;
    TransformerEngine& operator=(const TransformerEngine&) = delete;
    TransformerEngine(TransformerEngine&&) noexcept;
    TransformerEngine& operator=(TransformerEngine&&) noexcept;

    // 单步前向：输入一个 token，输出 vocab 维 logits，并更新 KV Cache
    std::vector<float> forward(int token);

    // 重置 KV Cache（开始新序列）
    void reset_kv_cache();

    const ModelConfig& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cllm
