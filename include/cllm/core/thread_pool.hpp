#pragma once

#include <cstddef>
#include <functional>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace cllm {

// 基于 std::thread + 任务队列的手写线程池，零第三方依赖
class ThreadPool {
public:
    explicit ThreadPool(size_t n_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // 分块并行执行 [begin, end)，fn(range_begin, range_end)
    void parallel_for(size_t begin, size_t end,
                      const std::function<void(size_t, size_t)>& fn);

    size_t size() const { return workers_.size(); }

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_ = false;
};

}  // namespace cllm
