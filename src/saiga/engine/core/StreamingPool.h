/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// StreamingPool (docs/engine_design.md §8.5; S20): the worker threads of content generation (terrain tiles, pyramids,
// grass masks; later meshing and IO). Jobs run on the workers (or on the caller through runOneOnCaller), their
// completions run on the main thread in drain(), in finishing order.

#pragma once

#include "saiga/export.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <condition_variable>

namespace Saiga
{
namespace Engine
{
/**
 * A priority queue of jobs and a few low-priority workers (§8.5, D8):
 * - submit(job, priority, done): lower priorities run first, FIFO among equal priorities. Jobs capture copies (never
 *   the World or a store) and write only into their own result object; the completion moves it where it belongs.
 * - cancel(id): a queued job is dropped (its completion is never called, true); a running job gets its `cancelled`
 *   flag set (it may stop early or ignore it) and its completion is called with cancelled = true at the next drain
 *   (false).
 * - drain() (main thread): the completions of the finished jobs in finishing order. A job that threw: its exception is
 *   rethrown here (its completion is not called; the completions behind it stay queued for the next drain).
 * The pool keeps no registry of what a job produces: its bookkeeping (queue entry, running count) ends with the job,
 * whatever the outcome; callers track their requests themselves.
 */
class SAIGA_ENGINE_API StreamingPool
{
   public:
    using Job        = std::function<void(const std::atomic<bool>& cancelled)>;  // worker thread
    using Completion = std::function<void(bool cancelled)>;                      // main thread, in drain()

    // 2 workers by default (§8.5: 2-3); the workers lower their priority (best effort: SetThreadPriority / setpriority)
    explicit StreamingPool(int workers = 2);
    // Cancels the queued jobs, sets `cancelled` of the running ones and joins them; completions and exceptions not
    // drained are discarded
    ~StreamingPool();
    StreamingPool(const StreamingPool&)            = delete;
    StreamingPool& operator=(const StreamingPool&) = delete;

    // Lower runs first; FIFO among equal priorities. Returns the job's ID (> 0)
    uint64_t submit(Job job, int priority, Completion done);
    // Queued: dropped, `done` is never called, returns true. Running: its `cancelled` flag is set, done(true) at the
    // next drain, returns false. Unknown or finished: false
    bool cancel(uint64_t id);
    // Main thread: the completions of the finished jobs in finishing order (see the class comment for exceptions)
    void drain();
    // The calling thread runs the first queued job itself (priority order), exactly as a worker would (its completion
    // or exception is queued for drain). False if nothing was queued
    bool runOneOnCaller();
    // Blocks until no job is queued or running; never throws (exceptions stay captured until drain)
    void waitIdle();
    // The same with a timeout: false if jobs remain
    bool waitIdleFor(double seconds);

    int queued() const;
    int running() const;
    // Finished jobs whose completion (or exception) was not drained yet
    int finished() const;
    int workers() const { return int(threads.size()); }

   private:
    struct Entry
    {
        uint64_t id = 0;
        Job job;
        Completion done;
        std::shared_ptr<std::atomic<bool>> cancelled;
    };
    struct Finished
    {
        uint64_t id = 0;
        Completion done;
        bool cancelled = false;
        std::exception_ptr error;
    };

    mutable std::mutex mutex;
    std::condition_variable workAvailable, idle;
    // Keyed by (priority, id): priority order, FIFO among equal priorities
    std::map<std::pair<int, uint64_t>, Entry> queue;
    // The cancel flags of the running jobs by ID
    std::map<uint64_t, std::shared_ptr<std::atomic<bool>>> runningJobs;
    std::deque<Finished> done;
    uint64_t nextId = 1;
    bool stopping   = false;
    std::vector<std::thread> threads;

    void workerLoop();
    // Takes the first queued entry (the mutex is held); false if none
    bool takeFirst(Entry& out);
    // Runs the entry's job outside the lock and queues its completion
    void execute(Entry& e);
};

}  // namespace Engine
}  // namespace Saiga
