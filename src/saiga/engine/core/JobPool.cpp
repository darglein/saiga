/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/core/JobPool.h"

#include "saiga/core/util/assert.h"

#include <atomic>
#include <exception>

namespace Saiga
{
namespace Engine
{
namespace
{
// Set while this thread runs chunks: a nested parallelFor runs serially
thread_local bool insideChunk = false;

struct InsideChunk
{
    bool previous;
    InsideChunk() : previous(insideChunk) { insideChunk = true; }
    ~InsideChunk() { insideChunk = previous; }
};

// The bounds of chunk k in unsigned arithmetic: no overflow for any int64 range (k * chunk < end - begin)
void chunkBounds(int64_t begin, int64_t end, int64_t chunk, int64_t k, int64_t& b, int64_t& e)
{
    b                  = int64_t(uint64_t(begin) + uint64_t(k) * uint64_t(chunk));
    uint64_t remaining = uint64_t(end) - uint64_t(b);
    e                  = remaining > uint64_t(chunk) ? int64_t(uint64_t(b) + uint64_t(chunk)) : end;
}
}  // namespace

// A parallelFor in progress. Lives on the caller's stack; parallelFor returns only after every worker that registered
// on it (active) has left, so no worker touches a finished job.
struct JobPool::Job
{
    int64_t begin, end, chunk, chunkCount;
    const std::function<void(int64_t, int64_t)>* fn;
    std::atomic<int64_t> nextChunk{0};
    // Set by the first exception: the chunks not started yet are skipped
    std::atomic<bool> failed{false};
    std::mutex errorMutex;
    std::exception_ptr error;
    // Workers that took the job and have not left it yet (guarded by JobPool::mutex)
    int active = 0;
};

JobPool::JobPool(int workerThreads)
{
    startWorkers(workerThreads);
}

JobPool::~JobPool()
{
    stopWorkers();
}

void JobPool::setWorkerThreads(int count)
{
    // From inside a chunk it would wait for its own parallelFor (the flag is per thread, so this also rejects a chunk of
    // another pool, which would be safe; there is no use for it)
    SAIGA_ASSERT(!insideChunk, "JobPool::setWorkerThreads from inside a parallelFor chunk");
    std::lock_guard<std::mutex> call(callMutex);
    stopWorkers();
    startWorkers(count);
}

void JobPool::startWorkers(int count)
{
    SAIGA_ASSERT(count >= 0);
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = false;
    }
    workers.reserve(count);
    for (int i = 0; i < count; ++i) workers.emplace_back([this]() { workerLoop(); });
}

void JobPool::stopWorkers()
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    workAvailable.notify_all();
    for (auto& t : workers) t.join();
    workers.clear();
}

void JobPool::workerLoop()
{
    uint64_t seen = 0;
    std::unique_lock<std::mutex> lock(mutex);
    while (true)
    {
        workAvailable.wait(lock, [&]() { return stop || (job && generation != seen); });
        if (stop) return;
        seen   = generation;
        Job* j = job;
        j->active++;
        lock.unlock();

        runChunks(*j);

        lock.lock();
        // The caller waits for every registered worker before the job goes out of scope
        if (--j->active == 0) workersLeft.notify_all();
    }
}

void JobPool::runChunks(Job& job)
{
    InsideChunk inside;
    while (!job.failed.load(std::memory_order_relaxed))
    {
        int64_t k = job.nextChunk.fetch_add(1, std::memory_order_relaxed);
        if (k >= job.chunkCount) break;
        int64_t b, e;
        chunkBounds(job.begin, job.end, job.chunk, k, b, e);
        try
        {
            (*job.fn)(b, e);
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(job.errorMutex);
            if (!job.error) job.error = std::current_exception();
            job.failed = true;
        }
    }
}

void JobPool::parallelFor(int64_t begin, int64_t end, int64_t chunk, const std::function<void(int64_t, int64_t)>& fn)
{
    SAIGA_ASSERT(chunk > 0, "JobPool::parallelFor: chunk must be positive");
    if (end <= begin) return;

    // Unsigned: end - begin may exceed INT64_MAX (begin < 0 < end). The chunk count must fit int64 (only a range wider
    // than 2^63 with tiny chunks does not)
    uint64_t span = uint64_t(end) - uint64_t(begin);
    SAIGA_ASSERT((span - 1) / uint64_t(chunk) < uint64_t(INT64_MAX), "JobPool::parallelFor: too many chunks");
    int64_t chunkCount = int64_t((span - 1) / uint64_t(chunk) + 1);

    auto serial = [&]()
    {
        InsideChunk inside;
        for (int64_t k = 0; k < chunkCount; ++k)
        {
            int64_t b, e;
            chunkBounds(begin, end, chunk, k, b, e);
            fn(b, e);
        }
    };
    // Nested: serially, without the call lock (the outer parallelFor holds it)
    if (insideChunk)
    {
        serial();
        return;
    }

    std::lock_guard<std::mutex> call(callMutex);
    if (workers.empty())
    {
        serial();
        return;
    }
    Job j;
    j.begin      = begin;
    j.end        = end;
    j.chunk      = chunk;
    j.chunkCount = chunkCount;
    j.fn         = &fn;
    {
        std::lock_guard<std::mutex> lock(mutex);
        job = &j;
        generation++;
    }
    workAvailable.notify_all();

    // The caller participates
    runChunks(j);

    {
        std::unique_lock<std::mutex> lock(mutex);
        // No new worker may take the job; then wait for the ones that did
        job = nullptr;
        workersLeft.wait(lock, [&]() { return j.active == 0; });
    }
    if (j.error) std::rethrow_exception(j.error);
}

}  // namespace Engine
}  // namespace Saiga
