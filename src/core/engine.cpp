#include "cllm/core/engine.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cllm {

namespace {

// ---- 基础算子 ----

// x[1,in] @ W[out,in]^T -> y[1,out]
void matmul(const std::vector<float>& x, const std::vector<float>& W,
            int in_dim, int out_dim, std::vector<float>& y, ThreadPool* pool) {
    y.assign(out_dim, 0.0f);
    if (pool && out_dim >= 1024) {
        pool->parallel_for(0, (size_t)out_dim, [&](size_t b, size_t e) {
            for (size_t j = b; j < e; ++j) {
                const float* wr = W.data() + j * in_dim;
                float s = 0.0f;
                for (int i = 0; i < in_dim; ++i) s += x[i] * wr[i];
                y[j] = s;
            }
        });
    } else {
        for (int j = 0; j < out_dim; ++j) {
            const float* wr = W.data() + j * in_dim;
            float s = 0.0f;
            for (int i = 0; i < in_dim; ++i) s += x[i] * wr[i];
            y[j] = s;
        }
    }
}

// RMSNorm：out = x / rms(x) * w
void rms_norm(const std::vector<float>& x, const std::vector<float>& w,
              float eps, std::vector<float>& out) {
    int n = (int)x.size();
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += x[i] * x[i];
    float rms = 1.0f / std::sqrt(sum / n + eps);
    out.resize(n);
    for (int i = 0; i < n; ++i) out[i] = x[i] * rms * w[i];
}

// SwiGLU 激活
inline float silu(float v) { return v / (1.0f + std::exp(-v)); }

// RoPE（NEOX 风格：head_dim 前半与后半配对旋转）
// 对每个 head，将 head_dim 分成两半 [0:half) 与 [half:head_dim)，
// 第 i 对频率 theta = pos / rope_theta^(i/half)，i = 0..half-1
void apply_rope(float* q, int n_head, int head_dim, int pos, float rope_theta) {
    int half = head_dim / 2;
    for (int h = 0; h < n_head; ++h) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < half; ++i) {
            float theta = pos / std::pow(rope_theta, (float)i / half);
            float c = std::cos(theta), s = std::sin(theta);
            float x0 = qh[i];
            float x1 = qh[i + half];
            qh[i]        = x0 * c - x1 * s;
            qh[i + half] = x0 * s + x1 * c;
        }
    }
}

// GQA Attention：q[n_head*head_dim] 对历史 KV 计算加权输出
void attention(const std::vector<float>& q,
               const std::vector<float>& kc, const std::vector<float>& vc,
               int seq, int n_head, int n_head_kv, int head_dim,
               std::vector<float>& out) {
    out.assign(n_head * head_dim, 0.0f);
    int n_rep = n_head / n_head_kv;
    float scale = 1.0f / std::sqrt((float)head_dim);
    std::vector<float> scores(seq);
    for (int h = 0; h < n_head; ++h) {
        int kv_h = h / n_rep;
        const float* qh = q.data() + h * head_dim;
        float* oh = out.data() + h * head_dim;
        float max_s = -INFINITY;
        for (int t = 0; t < seq; ++t) {
            const float* kh = kc.data() + t * (n_head_kv * head_dim) + kv_h * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; ++d) s += qh[d] * kh[d];
            s *= scale;
            scores[t] = s;
            if (s > max_s) max_s = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < seq; ++t) { scores[t] = std::exp(scores[t] - max_s); sum += scores[t]; }
        for (int t = 0; t < seq; ++t) {
            const float* vh = vc.data() + t * (n_head_kv * head_dim) + kv_h * head_dim;
            float a = scores[t] / sum;
            for (int d = 0; d < head_dim; ++d) oh[d] += a * vh[d];
        }
    }
}

}  // namespace

struct TransformerEngine::Impl {
    ModelConfig cfg;
    ThreadPool* pool = nullptr;

    // 反量化后的权重缓存（float32）
    std::unordered_map<std::string, std::vector<float>> W;
    // bias 缓存（可选，Qwen2 等架构的 qkv 投影带 bias）
    std::unordered_map<std::string, std::vector<float>> B;
    // 记录每个线性权重矩阵的 out/in 维
    struct WeightMeta { int out_dim = 0; int in_dim = 0; };
    std::unordered_map<std::string, WeightMeta> meta;

    // KV Cache：每层 [seq * n_head_kv * head_dim]
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;
    int seq_len = 0;

    int n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0;

    void load_tensor(const GgufModel& model, const std::string& name,
                     int out_dim, int in_dim) {
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            throw std::runtime_error("missing tensor: " + name);
        }
        W[name] = it->second.dequantize();
        meta[name] = {out_dim, in_dim};
    }

    // 可选加载 bias（不存在则跳过，返回是否加载成功）
    bool load_bias(const GgufModel& model, const std::string& name) {
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) return false;
        B[name] = it->second.dequantize();
        return true;
    }
};

TransformerEngine::TransformerEngine(const GgufModel& model, ThreadPool& pool)
    : impl_(std::make_unique<Impl>()) {
    auto& c = impl_->cfg = model.config;
    impl_->pool = &pool;
    impl_->n_embd = c.n_embd;
    impl_->n_head = c.n_head;
    impl_->n_head_kv = c.n_head_kv;
    impl_->head_dim = c.n_embd / c.n_head;

    if (c.arch.empty()) throw std::runtime_error("unknown architecture (missing general.architecture)");

    // 加载 embedding 与输出
    impl_->load_tensor(model, "token_embd.weight", c.vocab_size, c.n_embd);
    // 输出层可能共享 embedding（tied）
    if (model.tensors.count("output.weight")) {
        impl_->load_tensor(model, "output.weight", c.vocab_size, c.n_embd);
    } else {
        impl_->W["output.weight"] = impl_->W["token_embd.weight"];
        impl_->meta["output.weight"] = {(int)c.vocab_size, (int)c.n_embd};
    }
    impl_->load_tensor(model, "output_norm.weight", c.n_embd, 1);

    // 每层权重
    impl_->k_cache.resize(c.n_layers);
    impl_->v_cache.resize(c.n_layers);
    int kv_dim = c.n_head_kv * impl_->head_dim;
    for (uint32_t l = 0; l < c.n_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        impl_->load_tensor(model, p + "attn_norm.weight", c.n_embd, 1);
        impl_->load_tensor(model, p + "attn_q.weight", c.n_embd, c.n_embd);
        impl_->load_tensor(model, p + "attn_k.weight", kv_dim, c.n_embd);
        impl_->load_tensor(model, p + "attn_v.weight", kv_dim, c.n_embd);
        impl_->load_bias(model, p + "attn_q.bias");
        impl_->load_bias(model, p + "attn_k.bias");
        impl_->load_bias(model, p + "attn_v.bias");
        impl_->load_tensor(model, p + "attn_output.weight", c.n_embd, c.n_embd);
        impl_->load_tensor(model, p + "ffn_norm.weight", c.n_embd, 1);
        impl_->load_tensor(model, p + "ffn_gate.weight", c.n_ff, c.n_embd);
        impl_->load_tensor(model, p + "ffn_up.weight", c.n_ff, c.n_embd);
        impl_->load_tensor(model, p + "ffn_down.weight", c.n_embd, c.n_ff);
    }
}

TransformerEngine::~TransformerEngine() = default;

TransformerEngine::TransformerEngine(TransformerEngine&&) noexcept = default;
TransformerEngine& TransformerEngine::operator=(TransformerEngine&&) noexcept = default;

const ModelConfig& TransformerEngine::config() const { return impl_->cfg; }

void TransformerEngine::reset_kv_cache() {
    for (auto& k : impl_->k_cache) k.clear();
    for (auto& v : impl_->v_cache) v.clear();
    impl_->seq_len = 0;
}

std::vector<float> TransformerEngine::forward(int token) {
    auto& I = *impl_;
    auto& c = I.cfg;
    int n_embd = I.n_embd, head_dim = I.head_dim;
    int kv_dim = I.n_head_kv * head_dim;

    // embedding
    const auto& emb = I.W["token_embd.weight"];
    std::vector<float> x(emb.begin() + (size_t)token * n_embd,
                         emb.begin() + (size_t)(token + 1) * n_embd);

    std::vector<float> h, q, k, v, attn_out, h2, gate, up, down, tmp;

    for (uint32_t l = 0; l < c.n_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";

        // ---- attention ----
        rms_norm(x, I.W[p + "attn_norm.weight"], c.norm_eps, h);
        matmul(h, I.W[p + "attn_q.weight"], n_embd, n_embd, q, I.pool);
        matmul(h, I.W[p + "attn_k.weight"], n_embd, kv_dim, k, I.pool);
        matmul(h, I.W[p + "attn_v.weight"], n_embd, kv_dim, v, I.pool);
        // 加 qkv 投影 bias（Qwen2 等架构带 bias，llama 无则跳过）
        if (auto it = I.B.find(p + "attn_q.bias"); it != I.B.end())
            for (int i = 0; i < n_embd; ++i) q[i] += it->second[i];
        if (auto it = I.B.find(p + "attn_k.bias"); it != I.B.end())
            for (int i = 0; i < kv_dim; ++i) k[i] += it->second[i];
        if (auto it = I.B.find(p + "attn_v.bias"); it != I.B.end())
            for (int i = 0; i < kv_dim; ++i) v[i] += it->second[i];

        int pos = I.seq_len;
        apply_rope(q.data(), I.n_head, head_dim, pos, c.rope_theta);
        apply_rope(k.data(), I.n_head_kv, head_dim, pos, c.rope_theta);

        // 追加 KV
        auto& kc = I.k_cache[l];
        auto& vc = I.v_cache[l];
        kc.insert(kc.end(), k.begin(), k.end());
        vc.insert(vc.end(), v.begin(), v.end());

        attention(q, kc, vc, I.seq_len + 1, I.n_head, I.n_head_kv, head_dim, attn_out);
        matmul(attn_out, I.W[p + "attn_output.weight"], n_embd, n_embd, tmp, I.pool);
        for (int i = 0; i < n_embd; ++i) x[i] += tmp[i];

        // ---- FFN (SwiGLU) ----
        rms_norm(x, I.W[p + "ffn_norm.weight"], c.norm_eps, h2);
        matmul(h2, I.W[p + "ffn_gate.weight"], n_embd, c.n_ff, gate, I.pool);
        matmul(h2, I.W[p + "ffn_up.weight"], n_embd, c.n_ff, up, I.pool);
        for (int i = 0; i < (int)c.n_ff; ++i) gate[i] = silu(gate[i]) * up[i];
        matmul(gate, I.W[p + "ffn_down.weight"], c.n_ff, n_embd, down, I.pool);
        for (int i = 0; i < n_embd; ++i) x[i] += down[i];
    }

    // ---- 输出层 ----
    rms_norm(x, I.W["output_norm.weight"], c.norm_eps, h);
    std::vector<float> logits;
    matmul(h, I.W["output.weight"], n_embd, c.vocab_size, logits, I.pool);

    ++I.seq_len;
    return logits;
}

}  // namespace cllm
