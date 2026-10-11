#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "cllm/core/gguf.hpp"
#include "cllm/core/thread_pool.hpp"

namespace cllm {

// 权重推理精度模式：
//   Quant：量化权重零拷贝引用 mmap，forward 走整数点积快速路径（默认）。
//   Fp32 ：将所有量化权重反量化为 fp32 常驻内存，forward 走纯 float 矩阵乘。
//   Fp16 ：将所有量化权重反量化为 fp16（半精度）常驻内存，forward 时 half -> float 累加。
enum class WeightMode : uint8_t {
    Quant = 0,
    Fp32  = 1,
    Fp16  = 2,
};

// 通用 Transformer 推理引擎（RMSNorm + RoPE + GQA Attention + SwiGLU FFN + KV Cache）
// 通过 ModelConfig 参数化，覆盖 Llama/Qwen/Mistral 同族架构。
class TransformerEngine {
public:
    // mode=Quant（默认）：量化权重零拷贝引用 mmap，forward 走整数点积快速路径。
    // mode=Fp32：将所有量化权重反量化为 fp32 常驻内存，forward 走纯 float 矩阵乘。
    // mode=Fp16：将所有量化权重反量化为 fp16 常驻内存，forward 走 half->float 矩阵乘。
    // 后两者用于「量化 vs 反量化 fp32/fp16」的公平性能对比。
    TransformerEngine(const GgufModel& model, ThreadPool& pool,
                      WeightMode mode = WeightMode::Quant);
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
