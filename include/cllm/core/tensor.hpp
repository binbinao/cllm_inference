#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "cllm/core/quant.hpp"

namespace cllm {

// 张量：保存形状、量化类型、指向 mmap 区域的数据指针
// 数据本身不拷贝，反量化时按需还原为 float
struct Tensor {
    std::vector<uint32_t> shape;   // GGUF 中维度为倒序存储（最内层在前）
    GgmlType type = GgmlType::F32;
    const void* data = nullptr;    // 指向 mmap 区域的只读指针
    uint64_t offset = 0;           // 文件内偏移（用于 mmap 定位）

    // 元素总数
    uint64_t numel() const {
        uint64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    // 反量化为 float 向量
    std::vector<float> dequantize() const;
};

}  // namespace cllm