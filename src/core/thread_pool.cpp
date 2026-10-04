#include "cllm/core/thread_pool.hpp"

#include <algorithm>

namespace cllm {

ThreadPool::ThreadPool(size_t n_threads) {
    if (n_threads == 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 1;
    }
    workers_.reserve(n_threads);
    for (size_t i = 0; i < n_threads; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

void ThreadPool::parallel_for(size_t begin, size_t end,
                              const std::function<void(size_t, size_t)>& fn) {
    if (end <= begin) return;
    size_t n_workers = workers_.size();
    size_t total = end - begin;

    // 任务太小时直接串行，避免调度开销
    if (n_workers <= 1 || total <= 1) {
        fn(begin, end);
        return;
    }

    size_t chunk = std::max<size_t>(1, (total + n_workers - 1) / n_workers);
    size_t n_chunks = (total + chunk - 1) / chunk;

    std::mutex counter_mtx;
    std::condition_variable counter_cv;
    size_t remaining = n_chunks;

    for (size_t c = 0; c < n_chunks; ++c) {
        size_t b = begin + c * chunk;
        size_t e = std::min(end, b + chunk);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            tasks_.emplace([b, e, &fn, &counter_mtx, &counter_cv, &remaining] {
                fn(b, e);
                {
                    std::lock_guard<std::mutex> lk(counter_mtx);
                    --remaining;
                }
                counter_cv.notify_one();
            });
        }
        cv_.notify_one();
    }

    std::unique_lock<std::mutex> lock(counter_mtx);
    counter_cv.wait(lock, [&remaining] { return remaining == 0; });
}

}  // namespace cllm
