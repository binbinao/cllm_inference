# CPU runtime next Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. TDD.

**Goal:** 落地 `--ctx`、mmap 预热、Q4_0/Q5_0 AVX2、stop/seed、attention SIMD、bench 改 prefill。

**Architecture:** 引擎构造增加可选 `ctx_limit`；生成循环持有 seed/stop；量化与 attention 在现有函数内加 SIMD 分支。

**Tech Stack:** C++20、POSIX mmap/madvise、AVX2。

## Tasks

1. Engine `--ctx` + reserve 上限 + mmap `MADV_WILLNEED`
2. Q4_0/Q5_0 AVX2
3. `consume_stop_suffix` + CLI/HTTP seed/stop
4. Attention SIMD
5. model_bench + README
