//
// Created by Peter Eichinger on 2019-01-21.
//

#pragma once

#include "saiga/util/easylogging++.h"

#include "ChunkAllocation.h"
#include "FitStrategy.h"
#include "MemoryLocation.h"

#include <atomic>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include <condition_variable>

namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
class Defragger
{
   public:
    struct OperationPenalties
    {
        float target_small_hole     = 100.0f;
        float source_create_hole    = 200.0f;
        float source_not_last_alloc = 100.0f;
        float source_not_last_chunk = 400.0f;
    };

   private:
    struct DefragOperation
    {
        MemoryLocation source;
        vk::DeviceMemory targetMemory;
        FreeListEntry target;
        float weight;


        bool operator<(const DefragOperation& second) const { return this->weight > second.weight; }
    };

    bool enabled;
    std::vector<ChunkAllocation>* chunks;
    FitStrategy* strategy;
    std::set<DefragOperation> defrag_operations;

    std::atomic_bool running, quit;

    std::mutex start_mutex, running_mutex, invalidate_mutex;
    std::condition_variable start_condition;
    std::thread worker;

    void worker_func();

    std::set<vk::DeviceMemory> invalidate_set;


    // Defrag thread functions
    float get_operation_penalty(ConstChunkIterator target_chunk, ConstFreeIterator target_location,
                                ConstChunkIterator source_chunk, ConstAllocationIterator source_location) const;

    void apply_invalidations();

    void perform_defragmentation();
    // end defrag thread functions
   public:
    OperationPenalties penalties;
    Defragger(std::vector<ChunkAllocation>* _chunks, FitStrategy* _strategy)
        : enabled(false),
          chunks(_chunks),
          strategy(_strategy),
          defrag_operations(),
          running(false),
          quit(false),
          worker(&Defragger::worker_func, this)
    {
    }

    Defragger(const Defragger& other) = delete;
    Defragger& operator=(const Defragger& other) = delete;

    virtual ~Defragger() { exit(); }

    void exit()
    {
        running = false;
        std::unique_lock<std::mutex> lock(running_mutex);

        {
            std::lock_guard<std::mutex> lock(start_mutex);
            running = false;
            quit    = true;
        }
        start_condition.notify_one();
        worker.join();
    }

    void start();
    void stop();

    void setEnabled(bool enable) { enabled = enable; }

    void invalidate(vk::DeviceMemory memory);
};

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga