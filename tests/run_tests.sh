#!/usr/bin/env bash
# cllm_inference 测试与基准一键运行脚本
# 用法：./tests/run_tests.sh [--bench]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CXX="${CXX:-clang++}"
CXXFLAGS="-std=c++20 -O2 -Iinclude"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

run_case() {
    local name="$1"; shift
    echo "== 编译并运行 $name =="
    "$CXX" $CXXFLAGS "tests/$name.cpp" "$@" -o "$OUT/$name" -pthread
    "$OUT/$name"
    echo
}

run_case quant_test src/core/quant.cpp src/core/thread_pool.cpp
run_case threadpool_test src/core/thread_pool.cpp
run_case threadpool_stress_test src/core/thread_pool.cpp
run_case sampler_test src/core/sampler.cpp

if [[ "${1:-}" == "--bench" ]]; then
    echo "== 编译并运行 bench =="
    "$CXX" $CXXFLAGS tests/bench.cpp src/core/quant.cpp src/core/thread_pool.cpp -o "$OUT/bench" -pthread
    "$OUT/bench" 30
fi

echo "全部测试通过 ✅"
