//
// Created by Peter Eichinger on 2019-01-21.
//

#include "Defragger.h"

#include <saiga/util/threadName.h>

using namespace Saiga::Vulkan::Memory;

void Defragger::start()
{
    if (running)
    {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(start_mutex);
        running = true;
    }
    start_condition.notify_one();
}

void Defragger::stop()
{
    running = false;
    std::unique_lock<std::mutex> lock(running_mutex);
}

void Defragger::worker_func()
{
    Saiga::setThreadName("Defragger");
    while (true)
    {
        std::unique_lock<std::mutex> lock(start_mutex);
        start_condition.wait(lock, [this] { return running || quit; });
        if (quit)
        {
            return;
        }
        std::unique_lock<std::mutex> running_lock(running_mutex);

        if (allocations->empty())
        {
            continue;
        }
        auto chunk_iter = allocations->rbegin();
        auto alloc_iter = chunk_iter->allocations.rbegin();
        while (running)
        {
            if (alloc_iter == chunk_iter->allocations.rend())
            {
                ++chunk_iter;
            }
            if (chunk_iter == allocations->rend())
            {
                chunk_iter = allocations->rbegin();
            }
            {
                using namespace std::chrono_literals;
                // std::this_thread::sleep_for(50ms);
            }
        }
    }
}

void Defragger::invalidate(vk::DeviceMemory memory) {}
