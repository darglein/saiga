/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include <condition_variable>

namespace Saiga
{
namespace Engine
{
/**
 * A pool of worker threads for parallel loops (docs/engine_design.md §8.5, R-PAR-1). parallelFor splits a range into
 * fixed chunks; the workers and the calling thread take chunks until all are done. The chunk boundaries do not depend
 * on the number of threads, only which thread runs a chunk does.
 *
 * - 0 workers: the chunks run serially on the calling thread, in order.
 * - A parallelFor called from inside a chunk (nested) runs serially on that thread (no deadlock).
 * - parallelFor calls from different threads are serialized (one job at a time).
 * - An exception thrown by a chunk is rethrown in the caller after all started chunks have finished; chunks are
 *   skipped from the moment a thread sees the failure (best effort: chunks claimed before may still run).
 */
class SAIGA_ENGINE_API JobPool
{
   public:
    explicit JobPool(int workerThreads = 0);
    ~JobPool();
    JobPool(const JobPool&)            = delete;
    JobPool& operator=(const JobPool&) = delete;

    // Stops and restarts the workers; waits for a running parallelFor of another thread. Not from inside a chunk
    // (asserted).
    void setWorkerThreads(int count);
    // Not synchronized with setWorkerThreads of another thread
    int workerThreads() const { return int(workers.size()); }

    // Calls fn(chunkBegin, chunkEnd) for [begin + k * chunk, min(begin + (k + 1) * chunk, end)), k = 0, 1, ...
    // (chunk > 0, asserted). Returns when every chunk is done. An empty range (end <= begin) calls nothing.
    void parallelFor(int64_t begin, int64_t end, int64_t chunk, const std::function<void(int64_t, int64_t)>& fn);

   private:
    struct Job;

    void startWorkers(int count);
    void stopWorkers();
    void workerLoop();
    // Runs chunks of the job until none is left (any thread)
    static void runChunks(Job& job);

    // One parallelFor at a time (calls from different threads wait here)
    std::mutex callMutex;

    // Guards job, generation, stop and Job::active
    std::mutex mutex;
    std::condition_variable workAvailable;
    std::condition_variable workersLeft;
    Job* job            = nullptr;
    uint64_t generation = 0;
    bool stop           = false;

    std::vector<std::thread> workers;
};

}  // namespace Engine
}  // namespace Saiga
