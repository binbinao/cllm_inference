#include "cllm/core/thread_pool.hpp"

#include <algorithm>
#include <memory>

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

    // 同步状态用 shared_ptr 持有：任务按值捕获 shared_ptr 副本，即使 parallel_for 提前返回、
    // 其栈帧被销毁（地址又被后续调用复用），未跑完的旧任务仍操作同一份状态，
    // 避免「栈地址复用 + 旧任务迟到 notify」造成的计数丢失与永久阻塞。
    struct Sync {
        std::mutex mtx;
        std::condition_variable cv;
        size_t remaining = 0;
    };
    auto sync = std::make_shared<Sync>();
    sync->remaining = n_chunks;

    for (size_t c = 0; c < n_chunks; ++c) {
        size_t b = begin + c * chunk;
        size_t e = std::min(end, b + chunk);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            tasks_.emplace([b, e, &fn, sync] {
                fn(b, e);
                {
                    std::lock_guard<std::mutex> lk(sync->mtx);
                    --sync->remaining;
                }
                sync->cv.notify_one();
            });
        }
        cv_.notify_one();
    }

    std::unique_lock<std::mutex> lock(sync->mtx);
    sync->cv.wait(lock, [&sync] { return sync->remaining == 0; });
}

}  // namespace cllm
