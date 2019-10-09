// Copyright (c) 2012 Jakob Progsch, Václav Zeman

// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.

// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:

//   1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.

//   2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.

//   3. This notice may not be removed or altered from any source
// distribution.

/**
 * This file was modified by Darius Rueckert for libsaiga.
 */

#pragma once

#include "saiga/config.h"

#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include <condition_variable>

namespace Saiga
{
class SAIGA_CORE_API ThreadPool
{
   public:
    ThreadPool(size_t threads, const std::string& name = "ThreadPool");
    ~ThreadPool();

    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

    void quit();

    size_t queueSize()
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        return tasks.size();
    }
    size_t getWorkingThreads() { return workingThreads; }

   private:
    // number of currently working threads
    size_t workingThreads = 0;
    std::string name;
    // need to keep track of threads so we can join them
    std::vector<std::thread> workers;
    // the task queue
    std::queue<std::function<void()> > tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

template <class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;


    auto task = std::make_shared<std::packaged_task<return_type()> >(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<return_type> res = task->get_future();


    if (workers.size() == 0)
    {
        // This is an empty thread pool
        // -> execute this task here without adding it to the queue
        // -> emulate single threaded behaviour
        (*task)();
        return res;
    }


    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

/**
 * A global thread pool that can be used from everywhere.
 * Create it at the beginning with createGlobalThreadPool.
 *
 * -1 initializes the thread count with omp_get_thread_num
 */
extern SAIGA_CORE_API std::unique_ptr<ThreadPool> globalThreadPool;
extern SAIGA_CORE_API void createGlobalThreadPool(int threads = -1);

}  // namespace Saiga
