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
void rms_norm_ptr(const float* x, const float* w, int n, float eps, float* out) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += x[i] * x[i];
    float rms = 1.0f / std::sqrt(sum / n + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * rms * w[i];
}

void rms_norm(const std::vector<float>& x, const std::vector<float>& w,
              float eps, std::vector<float>& out) {
    int n = (int)x.size();
    out.resize(n);
    rms_norm_ptr(x.data(), w.data(), n, eps, out.data());
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
void attention(const float* q,
               const std::vector<float>& kc, const std::vector<float>& vc,
               int seq, int n_head, int n_head_kv, int head_dim,
               float* out) {
    const int q_dim = n_head * head_dim;
    std::fill(out, out + q_dim, 0.0f);
    int n_rep = n_head / n_head_kv;
    float scale = 1.0f / std::sqrt((float)head_dim);
    std::vector<float> scores(seq);
    for (int h = 0; h < n_head; ++h) {
        int kv_h = h / n_rep;
        const float* qh = q + h * head_dim;
        float* oh = out + h * head_dim;
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
    WeightMode mode = WeightMode::Quant; // Quant: 整数点积; Fp32/Fp16: 反量化常驻

    // 量化线性权重：直接引用 mmap 中数据，零拷贝，不常驻 float
    struct QuantWeight {
        const void* data = nullptr;
        GgmlType type = GgmlType::F32;
        int out_dim = 0;
        int in_dim = 0;
    };
    std::unordered_map<std::string, QuantWeight> QW;
    // fp32 常驻权重缓冲（mode=Fp32 时，QW[].data 指向这里的 float 数据）
    std::unordered_map<std::string, std::vector<float>> FW;
    // fp16 常驻权重缓冲（mode=Fp16 时，QW[].data 指向这里的 uint16_t 位模式）
    std::unordered_map<std::string, std::vector<uint16_t>> FH;
    // norm 权重（量级小，反量化为 float 供 rms_norm 使用）
    std::unordered_map<std::string, std::vector<float>> NW;
    // bias 缓存（可选，Qwen2 等架构的 qkv 投影带 bias）
    std::unordered_map<std::string, std::vector<float>> B;

    // KV Cache：每层 [seq * n_head_kv * head_dim]
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;
    std::vector<int> cached_tokens_;
    int seq_len = 0;

    int n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0;

    // 加载量化线性权重（记录类型与形状，数据零拷贝指向 mmap）
    void load_weight(const GgufModel& model, const std::string& name,
                     int out_dim, int in_dim) {
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            throw std::runtime_error("missing tensor: " + name);
        }
        if (mode == WeightMode::Quant) {
            QW[name] = {it->second.data, it->second.type, out_dim, in_dim};
        } else if (mode == WeightMode::Fp32) {
            std::vector<float> f = it->second.dequantize();
            FW[name] = std::move(f);
            QW[name] = {FW[name].data(), GgmlType::F32, out_dim, in_dim};
        } else {  // Fp16
            std::vector<uint16_t> h = it->second.dequantize_f16();
            FH[name] = std::move(h);
            QW[name] = {FH[name].data(), GgmlType::F16, out_dim, in_dim};
        }
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

    // 可选加载 norm 权重（张量不存在则跳过，返回是否成功）。
    // 用途：Qwen3 的 attn_q_norm/attn_k_norm 在 Qwen2/Llama 中不存在。
    bool load_norm_optional(const GgufModel& model, const std::string& name) {
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) return false;
        NW[name] = it->second.dequantize();
        return true;
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
        if (mode == WeightMode::Quant) {
            matmul_quant(x.data(), w.data, w.type, b ? b->data() : nullptr,
                         w.in_dim, w.out_dim, y.data(), pool);
        } else if (mode == WeightMode::Fp32) {
            // fp32 常驻：纯 float 矩阵乘（并行）
            const float* W = static_cast<const float*>(w.data);
            auto row = [&](int j) {
                const float* wr = W + (size_t)j * w.in_dim;
                float s = 0.0f;
                for (int i = 0; i < w.in_dim; ++i) s += x[i] * wr[i];
                if (b) s += (*b)[j];
                y[j] = s;
            };
            if (pool && w.out_dim >= 256) {
                pool->parallel_for(0, (size_t)w.out_dim, [&](size_t bb, size_t ee) {
                    for (size_t j = bb; j < ee; ++j) row((int)j);
                });
            } else {
                for (int j = 0; j < w.out_dim; ++j) row(j);
            }
        } else {
            // fp16 常驻：权重为半精度位模式，逐元素 half->float 后累加（并行）
            const uint16_t* W = static_cast<const uint16_t*>(w.data);
            auto row = [&](int j) {
                const uint16_t* wr = W + (size_t)j * w.in_dim;
                float s = 0.0f;
                for (int i = 0; i < w.in_dim; ++i) s += half_to_float(wr[i]) * x[i];
                if (b) s += (*b)[j];
                y[j] = s;
            };
            if (pool && w.out_dim >= 256) {
                pool->parallel_for(0, (size_t)w.out_dim, [&](size_t bb, size_t ee) {
                    for (size_t j = bb; j < ee; ++j) row((int)j);
                });
            } else {
                for (int j = 0; j < w.out_dim; ++j) row(j);
            }
        }
    }

    void mm_batch(const std::string& w_name, const std::string& b_name,
                  const float* x, int n_rows, float* y) {
        const auto& w = QW.at(w_name);
        const std::vector<float>* b = nullptr;
        auto it = B.find(b_name);
        if (it != B.end()) b = &it->second;
        const float* bp = b ? b->data() : nullptr;
        if (mode == WeightMode::Quant) {
            matmul_quant_batch(x, w.data, w.type, bp, w.in_dim, w.out_dim, n_rows, y, pool);
            return;
        }
        auto col = [&](int j) {
            for (int t = 0; t < n_rows; ++t) {
                const float* xt = x + (size_t)t * w.in_dim;
                float s = bp ? bp[j] : 0.0f;
                if (mode == WeightMode::Fp32) {
                    const float* wr = static_cast<const float*>(w.data) + (size_t)j * w.in_dim;
                    for (int i = 0; i < w.in_dim; ++i) s += xt[i] * wr[i];
                } else {
                    const uint16_t* wr = static_cast<const uint16_t*>(w.data) + (size_t)j * w.in_dim;
                    for (int i = 0; i < w.in_dim; ++i) s += half_to_float(wr[i]) * xt[i];
                }
                y[(size_t)t * w.out_dim + j] = s;
            }
        };
        if (pool && w.out_dim >= 256) {
            pool->parallel_for(0, (size_t)w.out_dim, [&](size_t bb, size_t ee) {
                for (size_t j = bb; j < ee; ++j) col((int)j);
            });
        } else {
            for (int j = 0; j < w.out_dim; ++j) col(j);
        }
    }

    void get_embedding(int token, float* dst) {
        const auto& w = QW.at("token_embd.weight");
        if (mode == WeightMode::Quant) {
            const int bs = block_size(w.type);
            const size_t row_bytes = (size_t)(w.in_dim / bs) * type_size(w.type);
            const uint8_t* base = static_cast<const uint8_t*>(w.data);
            dequantize_block(base + (size_t)token * row_bytes, dst, w.type, w.in_dim);
        } else if (mode == WeightMode::Fp32) {
            const float* base = static_cast<const float*>(w.data);
            std::copy(base + (size_t)token * w.in_dim,
                      base + (size_t)token * w.in_dim + w.in_dim, dst);
        } else {
            const uint16_t* base = static_cast<const uint16_t*>(w.data);
            const uint16_t* row = base + (size_t)token * w.in_dim;
            for (int i = 0; i < w.in_dim; ++i) dst[i] = half_to_float(row[i]);
        }
    }

    void get_embedding(int token, std::vector<float>& x) {
        x.resize(n_embd);
        get_embedding(token, x.data());
    }

    void truncate_kv(int n) {
        if (n < 0) n = 0;
        const int kv_dim = n_head_kv * head_dim;
        const size_t elems = (size_t)n * kv_dim;
        for (auto& k : k_cache) k.resize(elems);
        for (auto& v : v_cache) v.resize(elems);
        if ((int)cached_tokens_.size() > n) cached_tokens_.resize(n);
        seq_len = n;
    }

    void ensure_ctx_room(int add) const {
        if (cfg.n_ctx > 0 && seq_len + add > (int)cfg.n_ctx) {
            throw std::runtime_error("context length exceeded");
        }
    }
};

TransformerEngine::TransformerEngine(const GgufModel& model, ThreadPool& pool, WeightMode mode)
    : impl_(std::make_unique<Impl>()) {
    auto& c = impl_->cfg = model.config;
    impl_->pool = &pool;
    impl_->model = &model;
    impl_->mode = mode;
    impl_->n_embd = c.n_embd;
    impl_->n_head = c.n_head;
    impl_->n_head_kv = c.n_head_kv;
    // head_dim 使用 ModelConfig 的派生方法：若 GGUF 显式提供 attention.key_length
    // （Qwen3 等）则直接采用；否则退化为 n_embd / n_head（Qwen2/Llama 等）
    impl_->head_dim = c.head_dim();

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

    // 每层权重：
    // - Q 投影：out = n_head * head_dim（Qwen3 可能 != n_embd）
    // - K/V 投影：out = n_head_kv * head_dim（GQA）
    // - 输出投影：in = n_head * head_dim，out = n_embd
    impl_->k_cache.resize(c.n_layers);
    impl_->v_cache.resize(c.n_layers);
    {
        const size_t kv_elems = (size_t)c.n_ctx * (size_t)c.n_head_kv * (size_t)impl_->head_dim;
        for (uint32_t l = 0; l < c.n_layers; ++l) {
            impl_->k_cache[l].reserve(kv_elems);
            impl_->v_cache[l].reserve(kv_elems);
        }
    }
    const int head_dim = impl_->head_dim;
    const int q_dim = c.n_head * head_dim;
    const int kv_dim = c.n_head_kv * head_dim;
    for (uint32_t l = 0; l < c.n_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        impl_->load_norm(model, p + "attn_norm.weight", c.n_embd);
        impl_->load_weight(model, p + "attn_q.weight", q_dim, c.n_embd);
        impl_->load_weight(model, p + "attn_k.weight", kv_dim, c.n_embd);
        impl_->load_weight(model, p + "attn_v.weight", kv_dim, c.n_embd);
        // QKV bias（Qwen2 有；Qwen3/Llama 无，load_bias 返回 false 不报错）
        impl_->load_bias(model, p + "attn_q.bias");
        impl_->load_bias(model, p + "attn_k.bias");
        impl_->load_bias(model, p + "attn_v.bias");
        // Q/K 层级归一化（Qwen3 新增，size = head_dim；其他架构没有，静默跳过）
        impl_->load_norm_optional(model, p + "attn_q_norm.weight");
        impl_->load_norm_optional(model, p + "attn_k_norm.weight");
        impl_->load_weight(model, p + "attn_output.weight", c.n_embd, q_dim);
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
    impl_->cached_tokens_.clear();
    impl_->seq_len = 0;
}

int TransformerEngine::seq_len() const { return impl_->seq_len; }

namespace {

void apply_qk_norm(std::vector<float>& q, std::vector<float>& k,
                   const std::unordered_map<std::string, std::vector<float>>& NW,
                   const std::string& p, int n_head, int n_head_kv, int head_dim,
                   float norm_eps) {
    auto itq = NW.find(p + "attn_q_norm.weight");
    if (itq != NW.end()) {
        const auto& wq = itq->second;
        for (int hd = 0; hd < n_head; ++hd) {
            float* qh = q.data() + hd * head_dim;
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d) sum += qh[d] * qh[d];
            float rms = 1.0f / std::sqrt(sum / head_dim + norm_eps);
            for (int d = 0; d < head_dim; ++d) qh[d] = qh[d] * rms * wq[d];
        }
    }
    auto itk = NW.find(p + "attn_k_norm.weight");
    if (itk != NW.end()) {
        const auto& wk = itk->second;
        for (int hd = 0; hd < n_head_kv; ++hd) {
            float* kh = k.data() + hd * head_dim;
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d) sum += kh[d] * kh[d];
            float rms = 1.0f / std::sqrt(sum / head_dim + norm_eps);
            for (int d = 0; d < head_dim; ++d) kh[d] = kh[d] * rms * wk[d];
        }
    }
}

}  // namespace

std::vector<float> TransformerEngine::forward(int token, bool compute_logits) {
    auto& I = *impl_;
    auto& c = I.cfg;
    int n_embd = I.n_embd, head_dim = I.head_dim;
    I.ensure_ctx_room(1);

    std::vector<float> x;
    I.get_embedding(token, x);

    std::vector<float> h, q, k, v, attn_out, h2, gate, up, down, tmp;
    const int q_dim = I.n_head * head_dim;

    for (uint32_t l = 0; l < c.n_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";

        rms_norm(x, I.NW[p + "attn_norm.weight"], c.norm_eps, h);
        I.mm(p + "attn_q.weight", p + "attn_q.bias", h, q);
        I.mm(p + "attn_k.weight", p + "attn_k.bias", h, k);
        I.mm(p + "attn_v.weight", p + "attn_v.bias", h, v);
        apply_qk_norm(q, k, I.NW, p, I.n_head, I.n_head_kv, head_dim, c.norm_eps);

        int pos = I.seq_len;
        apply_rope(q.data(), I.n_head, head_dim, pos, c.rope_theta);
        apply_rope(k.data(), I.n_head_kv, head_dim, pos, c.rope_theta);

        auto& kc = I.k_cache[l];
        auto& vc = I.v_cache[l];
        kc.insert(kc.end(), k.begin(), k.end());
        vc.insert(vc.end(), v.begin(), v.end());

        attn_out.assign(q_dim, 0.0f);
        attention(q.data(), kc, vc, I.seq_len + 1, I.n_head, I.n_head_kv, head_dim, attn_out.data());
        I.mm(p + "attn_output.weight", "", attn_out, tmp);
        for (int i = 0; i < n_embd; ++i) x[i] += tmp[i];

        rms_norm(x, I.NW[p + "ffn_norm.weight"], c.norm_eps, h2);
        I.mm(p + "ffn_gate.weight", "", h2, gate);
        I.mm(p + "ffn_up.weight", "", h2, up);
        for (int i = 0; i < (int)c.n_ff; ++i) gate[i] = silu(gate[i]) * up[i];
        I.mm(p + "ffn_down.weight", "", gate, down);
        for (int i = 0; i < n_embd; ++i) x[i] += down[i];
    }

    I.cached_tokens_.push_back(token);
    ++I.seq_len;

    if (!compute_logits) return {};

    rms_norm(x, I.NW["output_norm.weight"], c.norm_eps, h);
    std::vector<float> logits;
    I.mm("output.weight", "", h, logits);
    return logits;
}

std::vector<float> TransformerEngine::prefill(const std::vector<int>& tokens) {
    auto& I = *impl_;
    auto& c = I.cfg;
    if (tokens.empty()) throw std::runtime_error("prefill: empty tokens");
    if (c.n_ctx > 0 && tokens.size() > c.n_ctx) {
        throw std::runtime_error("context length exceeded");
    }

    size_t lcp = 0;
    const size_t max_lcp = std::min(tokens.size(), I.cached_tokens_.size());
    while (lcp < max_lcp && tokens[lcp] == I.cached_tokens_[lcp]) ++lcp;

    if (lcp == tokens.size()) {
        I.truncate_kv(static_cast<int>(lcp - 1));
        return forward(tokens.back(), true);
    }

    I.truncate_kv(static_cast<int>(lcp));
    const int T = static_cast<int>(tokens.size() - lcp);
    if (T == 1) return forward(tokens.back(), true);

    I.ensure_ctx_room(T);
    const int n_embd = I.n_embd, head_dim = I.head_dim;
    const int q_dim = I.n_head * head_dim;
    const int kv_dim = I.n_head_kv * head_dim;
    const int pos0 = I.seq_len;

    std::vector<float> X((size_t)T * n_embd);
    for (int t = 0; t < T; ++t) {
        I.get_embedding(tokens[lcp + static_cast<size_t>(t)], X.data() + (size_t)t * n_embd);
    }

    std::vector<float> h((size_t)T * n_embd), q((size_t)T * q_dim), k((size_t)T * kv_dim);
    std::vector<float> v((size_t)T * kv_dim), attn((size_t)T * q_dim), tmp((size_t)T * n_embd);
    std::vector<float> h2((size_t)T * n_embd), gate((size_t)T * c.n_ff), up((size_t)T * c.n_ff);
    std::vector<float> down((size_t)T * n_embd);
    std::vector<float> qrow(q_dim), krow(kv_dim);

    for (uint32_t l = 0; l < c.n_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        for (int t = 0; t < T; ++t) {
            rms_norm_ptr(X.data() + (size_t)t * n_embd, I.NW[p + "attn_norm.weight"].data(),
                         n_embd, c.norm_eps, h.data() + (size_t)t * n_embd);
        }
        I.mm_batch(p + "attn_q.weight", p + "attn_q.bias", h.data(), T, q.data());
        I.mm_batch(p + "attn_k.weight", p + "attn_k.bias", h.data(), T, k.data());
        I.mm_batch(p + "attn_v.weight", p + "attn_v.bias", h.data(), T, v.data());

        for (int t = 0; t < T; ++t) {
            qrow.assign(q.begin() + (size_t)t * q_dim, q.begin() + (size_t)(t + 1) * q_dim);
            krow.assign(k.begin() + (size_t)t * kv_dim, k.begin() + (size_t)(t + 1) * kv_dim);
            apply_qk_norm(qrow, krow, I.NW, p, I.n_head, I.n_head_kv, head_dim, c.norm_eps);
            apply_rope(qrow.data(), I.n_head, head_dim, pos0 + t, c.rope_theta);
            apply_rope(krow.data(), I.n_head_kv, head_dim, pos0 + t, c.rope_theta);
            std::copy(qrow.begin(), qrow.end(), q.begin() + (size_t)t * q_dim);
            std::copy(krow.begin(), krow.end(), k.begin() + (size_t)t * kv_dim);
        }

        auto& kc = I.k_cache[l];
        auto& vc = I.v_cache[l];
        kc.insert(kc.end(), k.begin(), k.end());
        vc.insert(vc.end(), v.begin(), v.end());

        for (int t = 0; t < T; ++t) {
            attention(q.data() + (size_t)t * q_dim, kc, vc, pos0 + t + 1,
                      I.n_head, I.n_head_kv, head_dim, attn.data() + (size_t)t * q_dim);
        }
        I.mm_batch(p + "attn_output.weight", "", attn.data(), T, tmp.data());
        for (int i = 0; i < T * n_embd; ++i) X[i] += tmp[i];

        for (int t = 0; t < T; ++t) {
            rms_norm_ptr(X.data() + (size_t)t * n_embd, I.NW[p + "ffn_norm.weight"].data(),
                         n_embd, c.norm_eps, h2.data() + (size_t)t * n_embd);
        }
        I.mm_batch(p + "ffn_gate.weight", "", h2.data(), T, gate.data());
        I.mm_batch(p + "ffn_up.weight", "", h2.data(), T, up.data());
        for (int i = 0; i < T * (int)c.n_ff; ++i) gate[i] = silu(gate[i]) * up[i];
        I.mm_batch(p + "ffn_down.weight", "", gate.data(), T, down.data());
        for (int i = 0; i < T * n_embd; ++i) X[i] += down[i];
    }

    for (int t = 0; t < T; ++t) I.cached_tokens_.push_back(tokens[lcp + static_cast<size_t>(t)]);
    I.seq_len += T;

    std::vector<float> last(n_embd);
    std::copy(X.end() - n_embd, X.end(), last.begin());
    std::vector<float> hn, logits;
    rms_norm(last, I.NW["output_norm.weight"], c.norm_eps, hn);
    I.mm("output.weight", "", hn, logits);
    return logits;
}

}  // namespace cllm
