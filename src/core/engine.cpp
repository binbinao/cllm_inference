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
    const GgufModel* model = nullptr;   // 持有模型引用（数据来自其 mmap 区域）

    // 量化线性权重：直接引用 mmap 中数据，零拷贝，不常驻 float
    struct QuantWeight {
        const void* data = nullptr;
        GgmlType type = GgmlType::F32;
        int out_dim = 0;
        int in_dim = 0;
    };
    std::unordered_map<std::string, QuantWeight> QW;
    // norm 权重（量级小，反量化为 float 供 rms_norm 使用）
    std::unordered_map<std::string, std::vector<float>> NW;
    // bias 缓存（可选，Qwen2 等架构的 qkv 投影带 bias）
    std::unordered_map<std::string, std::vector<float>> B;

    // KV Cache：每层 [seq * n_head_kv * head_dim]
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;
    int seq_len = 0;

    int n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0;

    // 加载量化线性权重（记录类型与形状，数据零拷贝指向 mmap）
    void load_weight(const GgufModel& model, const std::string& name,
                     int out_dim, int in_dim) {
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            throw std::runtime_error("missing tensor: " + name);
        }
        QW[name] = {it->second.data, it->second.type, out_dim, in_dim};
    }

    // 加载并反量化一维 norm 权重为 float
    void load_norm(const GgufModel& model, const std::string& name, int n) {
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            throw std::runtime_error("missing tensor: " + name);
        }
        NW[name] = it->second.dequantize();
        (void)n;
    }

    // 可选加载 bias（不存在则跳过，返回是否加载成功）
    bool load_bias(const GgufModel& model, const std::string& name) {
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) return false;
        B[name] = it->second.dequantize();
        return true;
    }

    // 量化权重矩阵乘：y = x · Wᵀ (+ bias)，按输出行惰性反量化 + 并行
    void mm(const std::string& w_name, const std::string& b_name,
            const std::vector<float>& x, std::vector<float>& y) {
        const auto& w = QW.at(w_name);
        const std::vector<float>* b = nullptr;
        auto it = B.find(b_name);
        if (it != B.end()) b = &it->second;
        y.assign(w.out_dim, 0.0f);
        matmul_quant(x.data(), w.data, w.type, b ? b->data() : nullptr,
                     w.in_dim, w.out_dim, y.data(), pool);
    }

    // 取 embedding 矩阵第 token 行并反量化为 float（embedding 本身也可能是量化类型）
    void get_embedding(int token, std::vector<float>& x) {
        const auto& w = QW.at("token_embd.weight");
        const int bs = block_size(w.type);
        const size_t row_bytes = (size_t)(w.in_dim / bs) * type_size(w.type);
        const uint8_t* base = static_cast<const uint8_t*>(w.data);
        x.resize(w.in_dim);
        dequantize_block(base + (size_t)token * row_bytes, x.data(), w.type, w.in_dim);
    }
};

TransformerEngine::TransformerEngine(const GgufModel& model, ThreadPool& pool)
    : impl_(std::make_unique<Impl>()) {
    auto& c = impl_->cfg = model.config;
    impl_->pool = &pool;
    impl_->model = &model;
    impl_->n_embd = c.n_embd;
    impl_->n_head = c.n_head;
    impl_->n_head_kv = c.n_head_kv;
    impl_->head_dim = c.n_embd / c.n_head;

    if (c.arch.empty()) throw std::runtime_error("unknown architecture (missing general.architecture)");

    // 加载 embedding 与输出（线性权重零拷贝引用 mmap，不常驻 float）
    impl_->load_weight(model, "token_embd.weight", c.vocab_size, c.n_embd);
    // 输出层可能共享 embedding（tied）
    if (model.tensors.count("output.weight")) {
        impl_->load_weight(model, "output.weight", c.vocab_size, c.n_embd);
    } else {
        impl_->QW["output.weight"] = impl_->QW["token_embd.weight"];
    }
    impl_->load_norm(model, "output_norm.weight", c.n_embd);

    // 每层权重
    impl_->k_cache.resize(c.n_layers);
    impl_->v_cache.resize(c.n_layers);
    int kv_dim = c.n_head_kv * impl_->head_dim;
    for (uint32_t l = 0; l < c.n_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        impl_->load_norm(model, p + "attn_norm.weight", c.n_embd);
        impl_->load_weight(model, p + "attn_q.weight", c.n_embd, c.n_embd);
        impl_->load_weight(model, p + "attn_k.weight", kv_dim, c.n_embd);
        impl_->load_weight(model, p + "attn_v.weight", kv_dim, c.n_embd);
        impl_->load_bias(model, p + "attn_q.bias");
        impl_->load_bias(model, p + "attn_k.bias");
        impl_->load_bias(model, p + "attn_v.bias");
        impl_->load_weight(model, p + "attn_output.weight", c.n_embd, c.n_embd);
        impl_->load_norm(model, p + "ffn_norm.weight", c.n_embd);
        impl_->load_weight(model, p + "ffn_gate.weight", c.n_ff, c.n_embd);
        impl_->load_weight(model, p + "ffn_up.weight", c.n_ff, c.n_embd);
        impl_->load_weight(model, p + "ffn_down.weight", c.n_embd, c.n_ff);
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

    // embedding：反量化第 token 行
    std::vector<float> x;
    I.get_embedding(token, x);

    std::vector<float> h, q, k, v, attn_out, h2, gate, up, down, tmp;

    for (uint32_t l = 0; l < c.n_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";

        // ---- attention ----
        rms_norm(x, I.NW[p + "attn_norm.weight"], c.norm_eps, h);
        // 量化权重矩阵乘（bias 在 mm 内直接加入）
        I.mm(p + "attn_q.weight", p + "attn_q.bias", h, q);
        I.mm(p + "attn_k.weight", p + "attn_k.bias", h, k);
        I.mm(p + "attn_v.weight", p + "attn_v.bias", h, v);

        int pos = I.seq_len;
        apply_rope(q.data(), I.n_head, head_dim, pos, c.rope_theta);
        apply_rope(k.data(), I.n_head_kv, head_dim, pos, c.rope_theta);

        // 追加 KV
        auto& kc = I.k_cache[l];
        auto& vc = I.v_cache[l];
        kc.insert(kc.end(), k.begin(), k.end());
        vc.insert(vc.end(), v.begin(), v.end());

        attention(q, kc, vc, I.seq_len + 1, I.n_head, I.n_head_kv, head_dim, attn_out);
        I.mm(p + "attn_output.weight", "", attn_out, tmp);
        for (int i = 0; i < n_embd; ++i) x[i] += tmp[i];

        // ---- FFN (SwiGLU) ----
        rms_norm(x, I.NW[p + "ffn_norm.weight"], c.norm_eps, h2);
        I.mm(p + "ffn_gate.weight", "", h2, gate);
        I.mm(p + "ffn_up.weight", "", h2, up);
        for (int i = 0; i < (int)c.n_ff; ++i) gate[i] = silu(gate[i]) * up[i];
        I.mm(p + "ffn_down.weight", "", gate, down);
        for (int i = 0; i < n_embd; ++i) x[i] += down[i];
    }

    // ---- 输出层 ----
    rms_norm(x, I.NW["output_norm.weight"], c.norm_eps, h);
    std::vector<float> logits;
    I.mm("output.weight", "", h, logits);

    ++I.seq_len;
    return logits;
}

}  // namespace cllm
