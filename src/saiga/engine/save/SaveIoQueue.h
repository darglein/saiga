/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SaveIoQueue (docs/engine_design.md §15.1; S25): the IO lane of a save directory on the World's streaming pool
// (R-PAR-1: no step phase does IO). A queue of tasks of which at most one runs at a time: one pool job carries the head
// task (the lowest priority value, FIFO among equal ones) at its priority; a task that arrives with a better priority
// while that job is still queued (not running) cancels and resubmits it; a running task always finishes first. A task's
// run() runs on a worker (or on the caller), its done() on the main thread (the pool's drain, or the caller). Commits,
// the reads of stored groups, compaction and deletes all go through it, so records are appended in commit order and a
// read never sees a file that a later task deletes. drainOnCaller runs the lane on the caller (waitForCommits, close);
// the World's TerrainSystem notifies the lane before it resets its pool (the outstanding job's completion is discarded:
// its task is queued again and resubmitted).

#pragma once

#include "saiga/export.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>

namespace Saiga
{
namespace Engine
{
class World;

class SAIGA_ENGINE_API SaveIoQueue
{
   public:
    using Run  = std::function<void()>;
    using Done = std::function<void()>;

    explicit SaveIoQueue(World& world);
    ~SaveIoQueue();
    SaveIoQueue(const SaveIoQueue&)            = delete;
    SaveIoQueue& operator=(const SaveIoQueue&) = delete;

    // Queues a task (synchronous: runs it and its completion now, after the queue drained on the caller)
    void submit(int priority, bool read, Run run, Done done, bool synchronous);
    // Tasks queued or running
    bool busy() const;
    // Runs every task on the caller (waits for a running one, then runs the queued ones inline)
    void drainOnCaller();
    // The pool is about to be reset: the outstanding job's completion will be discarded
    void poolReset();
    // Submits the head task if no job is outstanding (after a pool reset, between steps)
    void pump();
    // Tests: runs on the worker before every task (a latch)
    std::function<void()> hookForTests;

   private:
    struct Task
    {
        uint64_t id  = 0;
        int priority = 0;
        bool read    = false;
        Run run;
        Done done;
    };
    World& w;
    std::deque<Task> queue;  // sorted by (priority, id)
    bool outstanding = false;
    int draining     = 0;  // drainOnCaller runs (a depth: a done callback may drain again): no job is submitted
    uint64_t jobId   = 0;
    Task current;
    uint64_t nextId = 1;
    // Shared with the job's completion (the queue may be destroyed first: then the completion does nothing)
    std::shared_ptr<bool> alive;

    void insert(Task t);
    void submitHead();
};

}  // namespace Engine
}  // namespace Saiga
