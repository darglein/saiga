/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/save/SaveIoQueue.h"

#include "saiga/engine/core/StreamingPool.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <vector>

namespace Saiga
{
namespace Engine
{
SaveIoQueue::SaveIoQueue(World& world) : w(world), alive(std::make_shared<bool>(true)) {}

SaveIoQueue::~SaveIoQueue()
{
    *alive = false;
}

void SaveIoQueue::insert(Task t)
{
    // FIFO among equal priorities: after every task with a priority <= t's
    auto it =
        std::upper_bound(queue.begin(), queue.end(), t.priority, [](int p, const Task& x) { return p < x.priority; });
    queue.insert(it, std::move(t));
}

void SaveIoQueue::submit(int priority, bool read, Run run, Done done, bool synchronous)
{
    if (synchronous)
    {
        drainOnCaller();
        if (hookForTests) hookForTests();
        run();
        done();
        return;
    }
    Task t;
    t.id       = nextId++;
    t.priority = priority;
    t.read     = read;
    t.run      = std::move(run);
    t.done     = std::move(done);
    // A better task than the queued (not running) job's: the job is cancelled and resubmitted with the new head
    if (outstanding && !draining && t.priority < current.priority && w.terrain().streamingPool().cancel(jobId))
    {
        outstanding = false;
        insert(std::move(current));
        current = Task();
    }
    insert(std::move(t));
    pump();
}

bool SaveIoQueue::busy() const
{
    return outstanding || !queue.empty();
}

void SaveIoQueue::pump()
{
    if (!outstanding && !draining && !queue.empty()) submitHead();
}

void SaveIoQueue::submitHead()
{
    current = std::move(queue.front());
    queue.pop_front();
    outstanding                      = true;
    const std::shared_ptr<bool> flag = alive;
    const Run run                    = current.run;
    const std::function<void()> hook = hookForTests;
    StreamingPool::Job job           = [run, hook](const std::atomic<bool>&)
    {
        if (hook) hook();
        run();
    };
    StreamingPool::Completion completion = [this, flag](bool)
    {
        if (!*flag) return;
        outstanding = false;
        Done d      = std::move(current.done);
        current     = Task();
        if (d) d();
        pump();
    };
    jobId = w.terrain().streamingPool().submit(std::move(job), current.priority, std::move(completion));
}

void SaveIoQueue::drainOnCaller()
{
    // A depth with a guard: a nested drain (from a done callback) keeps the outer one's state, and an exception
    // does not leave the lane draining
    struct DrainGuard
    {
        int& depth;
        explicit DrainGuard(int& d) : depth(d) { ++depth; }
        ~DrainGuard() { --depth; }
    } guard(draining);
    while (outstanding)
    {
        StreamingPool& pool = w.terrain().streamingPool();
        if (pool.cancel(jobId))
        {
            // Still queued: it runs on the caller below
            outstanding = false;
            queue.push_front(std::move(current));
            current = Task();
            break;
        }
        if (!pool.runOneOnCaller()) pool.waitIdleFor(0.001);
        w.terrain().drainStreamingPool();
    }
    while (!queue.empty())
    {
        Task t = std::move(queue.front());
        queue.pop_front();
        if (hookForTests) hookForTests();
        t.run();
        if (t.done) t.done();
    }
}

void SaveIoQueue::poolReset()
{
    if (!outstanding) return;
    // The job's completion is discarded with the pool and the task is queued again. Only a read may be re-run (it is
    // idempotent); a commit is not (it truncates behind a renamed world file), and loads drain the lane before they
    // reset the pool, so no commit can be outstanding here
    SAIGA_ASSERT(current.read, "SaveIoQueue::poolReset with a commit outstanding (a load must drain the lane first)");
    outstanding = false;
    queue.push_front(std::move(current));
    current = Task();
}

}  // namespace Engine
}  // namespace Saiga
