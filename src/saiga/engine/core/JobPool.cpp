/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// JobPool: worker threads, chunk claiming and the hand-over between parallelFor and the workers.

#include "saiga/engine/core/JobPool.h"

#include "saiga/core/util/assert.h"

#include <atomic>
#include <exception>
#include <stdexcept>

namespace Saiga
{
namespace Engine
{
namespace
{
// The pools whose chunks this thread is running (innermost first). A parallelFor of a pool in this chain is nested: it
// runs serially without the call lock (an outer frame of the same pool holds it, or waits for this thread).
struct ChunkFrame
{
    const JobPool* pool;
    const ChunkFrame* outer;
};
thread_local const ChunkFrame* chunkFrames = nullptr;

bool insideChunkOf(const JobPool* pool)
{
    for (const ChunkFrame* f = chunkFrames; f; f = f->outer)
        if (f->pool == pool) return true;
    return false;
}

struct InsideChunk
{
    ChunkFrame frame;
    explicit InsideChunk(const JobPool* pool) : frame{pool, chunkFrames} { chunkFrames = &frame; }
    ~InsideChunk() { chunkFrames = frame.outer; }
    InsideChunk(const InsideChunk&)            = delete;
    InsideChunk& operator=(const InsideChunk&) = delete;
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
    // From inside one of its own chunks it would wait for its own parallelFor
    SAIGA_ASSERT(!insideChunkOf(this), "JobPool::setWorkerThreads from inside a chunk of this pool");
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
    try
    {
        for (int i = 0; i < count; ++i) workers.emplace_back([this]() { workerLoop(); });
    }
    catch (...)
    {
        // A thread could not be started: join the ones that were (a joinable std::thread would terminate on
        // destruction); the pool is left with 0 workers
        stopWorkers();
        throw;
    }
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
    InsideChunk inside(this);
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
    // In every build (not an assert): skipping the loop silently would look like an empty range
    if ((span - 1) / uint64_t(chunk) >= uint64_t(INT64_MAX))
        throw std::length_error("JobPool::parallelFor: more than INT64_MAX chunks (use a larger chunk)");
    int64_t chunkCount = int64_t((span - 1) / uint64_t(chunk) + 1);

    auto serial = [&]()
    {
        InsideChunk inside(this);
        for (int64_t k = 0; k < chunkCount; ++k)
        {
            int64_t b, e;
            chunkBounds(begin, end, chunk, k, b, e);
            fn(b, e);
        }
    };
    // Nested in a chunk of this pool: serially, without the call lock (held by the outer parallelFor)
    if (insideChunkOf(this))
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
