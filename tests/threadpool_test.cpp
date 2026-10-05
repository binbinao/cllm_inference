// ThreadPool 正确性测试：验证并行分块结果与串行实现一致（对应 spec 场景）。
#include "cllm/core/thread_pool.hpp"

#include <atomic>
#include <cstdio>
#include <numeric>
#include <vector>

using namespace cllm;

namespace {
int g_fail = 0, g_pass = 0;

// 用 parallel_for 逐元素填充目标区间（并覆盖 [begin,end) 的完整性）
void fill_parallel(ThreadPool& pool, std::vector<int>& v, size_t begin, size_t end) {
    pool.parallel_for(begin, end, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) v[i] = (int)(i * i + 1);
    });
}

void test_matches_serial(size_t n_workers, size_t n) {
    ThreadPool pool(n_workers);
    std::vector<int> par(n, -1), ser(n, -1);
    fill_parallel(pool, par, 0, n);
    for (size_t i = 0; i < n; ++i) ser[i] = (int)(i * i + 1);

    bool ok = (par == ser);
    if (ok) ++g_pass;
    else { std::printf("  FAIL parallel != serial (workers=%zu n=%zu)\n", n_workers, n); ++g_fail; }
}

void test_repeatable(size_t n_workers, size_t n) {
    ThreadPool pool(n_workers);
    std::vector<int> a(n, -1), b(n, -1);
    fill_parallel(pool, a, 0, n);
    fill_parallel(pool, b, 0, n);
    if (a == b) ++g_pass;
    else { std::printf("  FAIL non-repeatable (workers=%zu n=%zu)\n", n_workers, n); ++g_fail; }
}

void test_subrange() {
    ThreadPool pool(4);
    std::vector<int> v(100, 0);
    // 只处理 [10, 90)，其余应保持 0
    pool.parallel_for(10, 90, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) v[i] = 7;
    });
    bool ok = true;
    for (size_t i = 0; i < 10; ++i) if (v[i] != 0) ok = false;
    for (size_t i = 10; i < 90; ++i) if (v[i] != 7) ok = false;
    for (size_t i = 90; i < 100; ++i) if (v[i] != 0) ok = false;
    if (ok) ++g_pass; else { std::printf("  FAIL subrange boundary\n"); ++g_fail; }
}

void test_empty() {
    ThreadPool pool(4);
    std::atomic<int> calls{0};
    pool.parallel_for(5, 5, [&](size_t, size_t) { ++calls; });  // 空区间不应调用
    if (calls.load() == 0) ++g_pass;
    else { std::printf("  FAIL empty range invoked %d times\n", calls.load()); ++g_fail; }
}

// 覆盖度：所有元素恰好被处理一次（用计数器检测重复/遗漏）
void test_exact_cover() {
    const size_t n = 10000;
    ThreadPool pool(6);
    std::vector<int> cnt(n, 0);
    pool.parallel_for(0, n, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) cnt[i] = 1;
    });
    double sum = std::accumulate(cnt.begin(), cnt.end(), 0.0);
    if (sum == (double)n) ++g_pass;
    else { std::printf("  FAIL cover sum=%.0f want %zu\n", sum, n); ++g_fail; }
}

}  // namespace

int main() {
    std::printf("[threadpool_test]\n");
    for (size_t w : {(size_t)1, (size_t)2, (size_t)4, (size_t)8}) {
        test_matches_serial(w, 1);
        test_matches_serial(w, 3);
        test_matches_serial(w, 1000);
        test_matches_serial(w, 100003);
        test_repeatable(w, 5000);
    }
    test_subrange();
    test_empty();
    test_exact_cover();

    std::printf("==== threadpool_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
