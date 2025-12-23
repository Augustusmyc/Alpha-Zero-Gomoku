#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <type_traits>

class ThreadPool {
public:
    using task_type = std::function<void()>;

    explicit ThreadPool(unsigned short thread_num = std::thread::hardware_concurrency())
        : run(true), idl_thread_num(thread_num) {
        for (unsigned i = 0; i < thread_num; ++i) {
            pool.emplace_back([this] {
                while (run.load(std::memory_order_acquire)) {
                    task_type task;
                    
                    {
                        std::unique_lock<std::mutex> lock(this->lock);
                        cv.wait(lock, [this] {
                            return !tasks.empty() || !run.load(std::memory_order_acquire);
                        });

                        if (!run.load(std::memory_order_acquire) && tasks.empty())
                            return;

                        task = std::move(tasks.front());
                        tasks.pop();
                    }

                    idl_thread_num.fetch_sub(1, std::memory_order_relaxed);
                    task();
                    idl_thread_num.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
    }

    ~ThreadPool() {
        run.store(false, std::memory_order_release);
        cv.notify_all();
        
        for (auto& thread : pool) {
            if (thread.joinable()) thread.join();
        }
    }

    template <typename F, typename... Args>
    auto commit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using return_type = std::invoke_result_t<F, Args...>;
        
        if (!run.load(std::memory_order_acquire))
            throw std::runtime_error("ThreadPool is stopped");

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            [func = std::forward<F>(f), ...args = std::forward<Args>(args)]() mutable {
                return func(std::forward<Args>(args)...);
            }
        );

        {
            std::lock_guard<std::mutex> lock(this->lock);
            tasks.emplace([task]() { (*task)(); });
        }

        cv.notify_one();
        return task->get_future();
    }

    int get_idl_num() const { 
        return idl_thread_num.load(std::memory_order_relaxed); 
    }

private:
    std::vector<std::thread> pool;
    std::queue<task_type> tasks;
    mutable std::mutex lock;
    std::condition_variable cv;

    std::atomic<bool> run;
    std::atomic<unsigned> idl_thread_num;
};