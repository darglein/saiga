/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/core/StreamingPool.h"

#include "saiga/core/util/assert.h"

#include <chrono>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#elif defined(__linux__)
#    include <sys/resource.h>
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

namespace Saiga
{
namespace Engine
{
namespace
{
// Best effort: a lower scheduling priority than the main thread (the frame must not wait for generation)
void lowerThreadPriority()
{
#if defined(_WIN32)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#elif defined(__linux__)
    // On Linux the nice value is per thread (setpriority with the thread ID)
    const auto tid = static_cast<id_t>(syscall(SYS_gettid));
    (void)setpriority(PRIO_PROCESS, tid, 5);
#endif
}
}  // namespace

StreamingPool::StreamingPool(int workers)
{
    SAIGA_ASSERT(workers >= 1, "StreamingPool: at least one worker");
    for (int i = 0; i < workers; ++i) threads.emplace_back([this]() { workerLoop(); });
}

StreamingPool::~StreamingPool()
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
        queue.clear();
        for (auto& [id, flag] : runningJobs) flag->store(true);
    }
    workAvailable.notify_all();
    for (auto& t : threads) t.join();
    // Completions and exceptions not drained are discarded
    done.clear();
}

uint64_t StreamingPool::submit(Job job, int priority, Completion completion)
{
    SAIGA_ASSERT(job, "StreamingPool::submit: no job");
    uint64_t id;
    {
        std::lock_guard<std::mutex> lock(mutex);
        id = nextId++;
        Entry e;
        e.id                                = id;
        e.job                               = std::move(job);
        e.done                              = std::move(completion);
        e.cancelled                         = std::make_shared<std::atomic<bool>>(false);
        queue[std::make_pair(priority, id)] = std::move(e);
    }
    workAvailable.notify_one();
    return id;
}

bool StreamingPool::cancel(uint64_t id)
{
    std::lock_guard<std::mutex> lock(mutex);
    for (auto it = queue.begin(); it != queue.end(); ++it)
    {
        if (it->first.second == id)
        {
            queue.erase(it);
            if (queue.empty() && runningJobs.empty()) idle.notify_all();
            return true;
        }
    }
    auto r = runningJobs.find(id);
    if (r != runningJobs.end()) r->second->store(true);
    return false;
}

bool StreamingPool::takeFirst(Entry& out)
{
    if (queue.empty()) return false;
    auto it = queue.begin();
    out     = std::move(it->second);
    queue.erase(it);
    runningJobs[out.id] = out.cancelled;
    return true;
}

void StreamingPool::execute(Entry& e)
{
    Finished f;
    f.id = e.id;
    try
    {
        e.job(*e.cancelled);
    }
    catch (...)
    {
        f.error = std::current_exception();
    }
    // The job's captures (copies of sources, results) are released outside of the lock
    e.job = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex);
        runningJobs.erase(e.id);
        f.cancelled = e.cancelled->load();
        f.done      = std::move(e.done);
        done.push_back(std::move(f));
        if (queue.empty() && runningJobs.empty()) idle.notify_all();
    }
}

void StreamingPool::workerLoop()
{
    lowerThreadPriority();
    for (;;)
    {
        Entry e;
        {
            std::unique_lock<std::mutex> lock(mutex);
            workAvailable.wait(lock, [this]() { return stopping || !queue.empty(); });
            if (stopping) return;
            takeFirst(e);
        }
        execute(e);
    }
}

bool StreamingPool::runOneOnCaller()
{
    Entry e;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!takeFirst(e)) return false;
    }
    execute(e);
    return true;
}

void StreamingPool::drain()
{
    for (;;)
    {
        Finished f;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (done.empty()) return;
            f = std::move(done.front());
            done.pop_front();
        }
        // Outside of the lock: a completion may submit new jobs
        if (f.error) std::rethrow_exception(f.error);
        if (f.done) f.done(f.cancelled);
    }
}

void StreamingPool::waitIdle()
{
    std::unique_lock<std::mutex> lock(mutex);
    idle.wait(lock, [this]() { return queue.empty() && runningJobs.empty(); });
}

bool StreamingPool::waitIdleFor(double seconds)
{
    std::unique_lock<std::mutex> lock(mutex);
    const auto timeout = std::chrono::duration<double>(seconds > 0 ? seconds : 0);
    return idle.wait_for(lock, std::chrono::duration_cast<std::chrono::nanoseconds>(timeout),
                         [this]() { return queue.empty() && runningJobs.empty(); });
}

int StreamingPool::queued() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return int(queue.size());
}

int StreamingPool::running() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return int(runningJobs.size());
}

int StreamingPool::finished() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return int(done.size());
}

}  // namespace Engine
}  // namespace Saiga
