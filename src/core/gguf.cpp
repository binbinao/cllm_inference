#include "cllm/core/gguf.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cllm {

namespace {

// GGUF metadata 值类型
enum GgufValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
};

uint8_t read_u8(FILE* f) { uint8_t v; if (fread(&v, 1, 1, f) != 1) throw std::runtime_error("read_u8"); return v; }
uint32_t read_u32(FILE* f) { uint32_t v; if (fread(&v, 4, 1, f) != 1) throw std::runtime_error("read_u32"); return v; }
uint64_t read_u64(FILE* f) { uint64_t v; if (fread(&v, 8, 1, f) != 1) throw std::runtime_error("read_u64"); return v; }
int32_t read_i32(FILE* f) { int32_t v; if (fread(&v, 4, 1, f) != 1) throw std::runtime_error("read_i32"); return v; }
float read_f32(FILE* f) { float v; if (fread(&v, 4, 1, f) != 1) throw std::runtime_error("read_f32"); return v; }

std::string read_string(FILE* f) {
    uint64_t len = read_u64(f);
    std::string s(len, '\0');
    if (len > 0 && fread(s.data(), 1, len, f) != len) throw std::runtime_error("read_string");
    return s;
}

// 跳过未知类型的值
void skip_value(FILE* f, uint32_t type) {
    switch (type) {
        case UINT8: case INT8: case BOOL: fseek(f, 1, SEEK_CUR); break;
        case UINT16: case INT16: fseek(f, 2, SEEK_CUR); break;
        case UINT32: case INT32: case FLOAT32: fseek(f, 4, SEEK_CUR); break;
        case UINT64: case INT64: case FLOAT64: fseek(f, 8, SEEK_CUR); break;
        case STRING: { uint64_t len = read_u64(f); fseek(f, (long)len, SEEK_CUR); break; }
        case ARRAY: {
            uint32_t et = read_u32(f);
            uint64_t n = read_u64(f);
            for (uint64_t i = 0; i < n; ++i) skip_value(f, et);
            break;
        }
        default: throw std::runtime_error("unknown value type");
    }
}

GgufValue read_value(FILE* f, uint32_t type) {
    GgufValue v;
    switch (type) {
        case UINT32: v.kind = GgufValue::Kind::U32; v.u32 = read_u32(f); break;
        case INT32:  v.kind = GgufValue::Kind::I32; v.i32 = read_i32(f); break;
        case FLOAT32: v.kind = GgufValue::Kind::F32; v.f32 = read_f32(f); break;
        case BOOL:   v.kind = GgufValue::Kind::Bool; v.b = read_u8(f) != 0; break;
        case STRING: v.kind = GgufValue::Kind::Str; v.str = read_string(f); break;
        case ARRAY: {
            uint32_t et = read_u32(f);
            uint64_t n = read_u64(f);
            if (et == STRING) {
                v.kind = GgufValue::Kind::ArrStr;
                v.arr_str.reserve(n);
                for (uint64_t i = 0; i < n; ++i) v.arr_str.push_back(read_string(f));
            } else if (et == FLOAT32) {
                v.kind = GgufValue::Kind::ArrF32;
                v.arr_f32.reserve(n);
                for (uint64_t i = 0; i < n; ++i) v.arr_f32.push_back(read_f32(f));
            } else if (et == INT32) {
                v.kind = GgufValue::Kind::ArrI32;
                v.arr_i32.reserve(n);
                for (uint64_t i = 0; i < n; ++i) v.arr_i32.push_back(read_i32(f));
            } else {
                for (uint64_t i = 0; i < n; ++i) skip_value(f, et);
            }
            break;
        }
        default: skip_value(f, type); break;
    }
    return v;
}

}  // namespace

GgufModel::~GgufModel() {
    if (mmap_base) { munmap(mmap_base, mmap_size); mmap_base = nullptr; }
}

GgufModel::GgufModel(GgufModel&& o) noexcept {
    *this = std::move(o);
}

GgufModel& GgufModel::operator=(GgufModel&& o) noexcept {
    if (this != &o) {
        if (mmap_base) munmap(mmap_base, mmap_size);
        config = o.config;
        tensors = std::move(o.tensors);
        tensor_list = std::move(o.tensor_list);
        tokens = std::move(o.tokens);
        token_scores = std::move(o.token_scores);
        token_types = std::move(o.token_types);
        merges = std::move(o.merges);
        tokenizer_model = std::move(o.tokenizer_model);
        bos_id = o.bos_id;
        eos_id = o.eos_id;
        add_bos_token = o.add_bos_token;
        chat_template = std::move(o.chat_template);
        mmap_base = o.mmap_base;
        mmap_size = o.mmap_size;
        o.mmap_base = nullptr;
        o.mmap_size = 0;
    }
    return *this;
}

GgufModel GgufLoader::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open model file: " + path);

    GgufModel model;

    // ---- 头部 ----
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "GGUF", 4) != 0) {
        fclose(f);
        throw std::runtime_error("invalid GGUF magic");
    }
    uint32_t version = read_u32(f);
    if (version != 2 && version != 3) {
        fclose(f);
        throw std::runtime_error("unsupported GGUF version: " + std::to_string(version));
    }
    uint64_t tensor_count = read_u64(f);
    uint64_t kv_count = read_u64(f);

    // ---- metadata KV ----
    std::unordered_map<std::string, GgufValue> kv;
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = read_string(f);
        uint32_t type = read_u32(f);
        kv[key] = read_value(f, type);
    }

    // ---- tensor info ----
    struct TensorInfo { std::string name; uint64_t offset; };
    std::vector<TensorInfo> infos;
    infos.reserve(tensor_count);
    for (uint64_t i = 0; i < tensor_count; ++i) {
        std::string name = read_string(f);
        uint32_t n_dims = read_u32(f);
        std::vector<uint64_t> dims(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) dims[d] = read_u64(f);
        uint32_t type = read_u32(f);
        uint64_t offset = read_u64(f);

        Tensor t;
        t.type = static_cast<GgmlType>(type);
        t.offset = offset;
        // GGUF dims 为倒序存储，反转为常规顺序 [rows, cols]
        t.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) t.shape[d] = static_cast<uint32_t>(dims[n_dims - 1 - d]);
        model.tensor_list.push_back(std::move(t));
        infos.push_back({name, offset});
    }

    // 数据区起始位置（当前文件位置），GGUF 要求 32 字节对齐
    long data_offset = ftell(f);
    data_offset = (data_offset + 31) & ~31L;
    int fd = fileno(f);

    // ---- mmap 整个文件 ----
    struct stat st;
    if (fstat(fd, &st) != 0) { fclose(f); throw std::runtime_error("fstat failed"); }
    void* base = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { fclose(f); throw std::runtime_error("mmap failed"); }
    model.mmap_base = base;
    model.mmap_size = st.st_size;

    // ---- 填充 tensor data 指针 ----
    for (size_t i = 0; i < model.tensor_list.size(); ++i) {
        model.tensor_list[i].data =
            static_cast<const char*>(base) + data_offset + infos[i].offset;
        model.tensors[infos[i].name] = model.tensor_list[i];
    }

    fclose(f);

    // ---- 提取 ModelConfig ----
    auto get_str = [&](const char* k) -> std::string {
        auto it = kv.find(k);
        return (it != kv.end() && it->second.kind == GgufValue::Kind::Str) ? it->second.str : "";
    };
    auto get_u32 = [&](const char* k) -> uint32_t {
        auto it = kv.find(k);
        return (it != kv.end() && it->second.kind == GgufValue::Kind::U32) ? it->second.u32 : 0;
    };

    model.config.arch = get_str("general.architecture");

    // 元数据字段前缀：架构名（qwen2 / llama / qwen35moe 等），
    // 部分旧转换器仍用 llama 前缀，读取时回退到 llama.
    const std::string arch = model.config.arch;
    auto get_u32_field = [&](const std::string& suffix) -> uint32_t {
        auto it = kv.find(arch + "." + suffix);
        if (it != kv.end() && it->second.kind == GgufValue::Kind::U32) return it->second.u32;
        it = kv.find("llama." + suffix);
        if (it != kv.end() && it->second.kind == GgufValue::Kind::U32) return it->second.u32;
        return 0;
    };
    auto get_f32_field = [&](const std::string& suffix) -> float {
        auto it = kv.find(arch + "." + suffix);
        if (it != kv.end() && it->second.kind == GgufValue::Kind::F32) return it->second.f32;
        it = kv.find("llama." + suffix);
        if (it != kv.end() && it->second.kind == GgufValue::Kind::F32) return it->second.f32;
        return 0.0f;
    };

    model.config.n_layers = get_u32_field("block_count");
    model.config.n_embd = get_u32_field("embedding_length");
    model.config.n_head = get_u32_field("attention.head_count");
    model.config.n_head_kv = get_u32_field("attention.head_count_kv");
    model.config.n_ctx = get_u32_field("context_length");
    model.config.n_ff = get_u32_field("feed_forward_length");
    float rope = get_f32_field("rope.freq_base");
    if (rope > 0) model.config.rope_theta = rope;
    float eps = get_f32_field("attention.layer_norm_rms_epsilon");
    if (eps > 0) model.config.norm_eps = eps;
    model.config.vocab_size = get_u32_field("vocab_size");

    // ---- 词表 ----
    model.tokenizer_model = get_str("tokenizer.ggml.model");
    auto it_tok = kv.find("tokenizer.ggml.tokens");
    if (it_tok != kv.end() && it_tok->second.kind == GgufValue::Kind::ArrStr) {
        model.tokens = it_tok->second.arr_str;
    }
    auto it_score = kv.find("tokenizer.ggml.scores");
    if (it_score != kv.end() && it_score->second.kind == GgufValue::Kind::ArrF32) {
        model.token_scores = it_score->second.arr_f32;
    }
    auto it_tt = kv.find("tokenizer.ggml.token_type");
    if (it_tt != kv.end() && it_tt->second.kind == GgufValue::Kind::ArrI32) {
        model.token_types = it_tt->second.arr_i32;
    }
    auto it_merges = kv.find("tokenizer.ggml.merges");
    if (it_merges != kv.end() && it_merges->second.kind == GgufValue::Kind::ArrStr) {
        model.merges = it_merges->second.arr_str;
    }
    model.bos_id = (int)get_u32("tokenizer.ggml.bos_token_id");
    model.eos_id = (int)get_u32("tokenizer.ggml.eos_token_id");

    // add_bos_token（Qwen 等模型为 false）
    auto it_ab = kv.find("tokenizer.ggml.add_bos_token");
    if (it_ab != kv.end() && it_ab->second.kind == GgufValue::Kind::Bool) {
        model.add_bos_token = it_ab->second.b;
    }
    // chat_template
    auto it_ct = kv.find("tokenizer.chat_template");
    if (it_ct != kv.end() && it_ct->second.kind == GgufValue::Kind::Str) {
        model.chat_template = it_ct->second.str;
    }

    if (model.config.vocab_size == 0) model.config.vocab_size = (uint32_t)model.tokens.size();
    if (model.config.n_head_kv == 0) model.config.n_head_kv = model.config.n_head;

    return model;
}

}  // namespace cllm
