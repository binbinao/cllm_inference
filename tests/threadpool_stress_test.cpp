// ThreadPool 高频压力测试（回归用）。
// 背景：早期 parallel_for 把等待状态放在栈局部变量，任务用引用捕获；
// 高频调用时栈地址被复用，旧世代的迟到 notify 会污染新一轮状态，导致主线程永久阻塞。
// 本测试用「极短任务 + 极密调用」最大化该竞态窗口，锁死修复效果。
#include "cllm/core/thread_pool.hpp"

#include <cstdio>
#include <vector>

using namespace cllm;

int main() {
    const int kReps = 20;
    const int kIters = 20000;
    const size_t kN = 10;

    for (int rep = 0; rep < kReps; ++rep) {
        ThreadPool pool(10);
        // 每个分块写自己的槽位，避免测试自身的数据竞争
        std::vector<long long> hits(kN, 0);
        for (int k = 0; k < kIters; ++k) {
            pool.parallel_for(0, kN, [&](size_t b, size_t e) {
                for (size_t i = b; i < e; ++i) hits[i] += 1;
            });
        }
        long long sum = 0;
        for (long long v : hits) sum += v;
        long long expect = (long long)kIters * kN;
        if (sum != expect) {
            std::printf("  FAIL rep %d: sum=%lld want %lld\n", rep, sum, expect);
            return 1;
        }
    }

    std::printf("==== threadpool_stress_test: %d reps x %d iters passed, 0 failed ====\n",
                kReps, kIters);
    return 0;
}
