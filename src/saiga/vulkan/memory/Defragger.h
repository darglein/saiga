//
// Created by Peter Eichinger on 2019-01-21.
//

#pragma once

#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/threadName.h"
#include "saiga/vulkan/Queue.h"

#include "BufferChunkAllocator.h"
#include "BufferMemoryLocation.h"
#include "ChunkAllocation.h"
#include "FitStrategy.h"
#include "ImageMemoryLocation.h"

#include <atomic>
#include <list>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include <condition_variable>

namespace Saiga::Vulkan
{
struct VulkanBase;
}

namespace Saiga::Vulkan::Memory
{
class ImageCopyComputeShader;
struct OperationPenalties
{
    float target_small_hole     = 100.0f;
    float source_create_hole    = 200.0f;
    float source_not_last_alloc = 100.0f;
    float source_not_last_chunk = 400.0f;
    float same_chunk            = 500.0f;
};

struct DefraggerConfiguration
{
    float weight_chunk      = 1.0f;
    float weight_offset     = 0.5f;
    float weight_small_free = 0.5f;
    uint32_t max_targets    = 3;
};

template <typename T>
class Defragger
{
   protected:
    struct DefragOperation
    {
        T* source;
        vk::DeviceMemory targetMemory;
        FreeListEntry target;
        float weight;


        bool operator<(const DefragOperation& second) const { return this->weight < second.weight; }
    };

    struct FreeOperation
    {
        T *target, *source;
        int32_t delay;
    };

    bool valid;
    bool enabled;
    int32_t dealloc_delay;
    VulkanBase* base;
    vk::Device device;
    BaseChunkAllocator<T>* allocator;
    std::multiset<DefragOperation> defrag_operations;

    std::list<FreeOperation> free_operations;

    std::atomic_bool running, quit;
    std::atomic_int frame_number;

    std::mutex start_mutex, running_mutex, invalidate_mutex;
    std::condition_variable start_condition;
    std::thread worker;

    void worker_func();

    std::set<vk::DeviceMemory> invalidate_set;

    // Defrag thread functions
    float get_operation_penalty(ConstChunkIterator<T> target_chunk, ConstFreeIterator<T> target_location,
                                ConstChunkIterator<T> source_chunk, ConstAllocationIterator<T> source_location) const;

    void apply_invalidations();

    void run();


    void find_defrag_ops();
    bool perform_defrag();
    bool perform_free_operations();

    // end defrag thread functions
   public:
    OperationPenalties penalties;
    DefraggerConfiguration config;

    Defragger(VulkanBase* _base, vk::Device _device, BaseChunkAllocator<T>* _allocator, uint32_t _dealloc_delay = 0)
        : valid(true),
          enabled(false),
          dealloc_delay(_dealloc_delay + 15),
          // dealloc_delay(240),
          base(_base),
          device(_device),
          allocator(_allocator),
          defrag_operations(),
          free_operations(),
          running(false),
          quit(false),
          frame_number(0),
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
    void invalidate(T* location);

    void update(uint32_t _frame_number) { frame_number = _frame_number; }

   protected:
    virtual std::optional<FreeOperation> execute_copy_operation(const DefragOperation& op) = 0;

   private:
    template <typename Iter>
    inline vk::DeviceSize begin(Iter iter) const
    {
        return (**iter).offset;
    }

    template <typename Iter>
    inline vk::DeviceSize end(Iter iter) const
    {
        return (**iter).offset + (**iter).size;
    }

    template <typename Iter>
    inline vk::DeviceSize size(Iter iter) const
    {
        return (**iter).size;
    }

    inline float get_free_weight(vk::DeviceSize size) const
    {
        const vk::DeviceSize cap = 10 * 1024 * 1024;
        if (size == 0 || size > cap)
        {
            return 0;
        }

        return glm::mix(config.weight_small_free, 0.0f, static_cast<float>(size) / cap);
    }

    inline float get_target_weight(uint32_t chunkIndex, vk::DeviceSize chunkSize, vk::DeviceSize begin,
                                   vk::DeviceSize first, vk::DeviceSize second) const
    {
        return config.weight_chunk * chunkIndex + config.weight_offset * (static_cast<float>(begin) / chunkSize) +
               get_free_weight(first) + get_free_weight(second);
    }

    template <typename Iter>
    inline float get_weight(uint32_t chunkIndex, vk::DeviceSize chunkSize, Iter alloc, vk::DeviceSize first,
                            vk::DeviceSize second) const
    {
        return config.weight_chunk * chunkIndex +
               config.weight_offset * (static_cast<float>(begin(alloc)) / chunkSize) + get_free_weight(first) +
               get_free_weight(second);
    }

    template <typename Iter>
    inline T* get(Iter iter) const
    {
        return (*iter).get();
    }
};



class BufferDefragger : public Defragger<BufferMemoryLocation>
{
   public:
    BufferDefragger(VulkanBase* base, vk::Device device, BaseChunkAllocator<BufferMemoryLocation>* allocator,
                    uint32_t dealloc_delay)
        : Defragger(base, device, allocator, dealloc_delay)
    {
    }

   protected:
    std::optional<FreeOperation> execute_copy_operation(const DefragOperation& op) override;
};

class ImageDefragger : public Defragger<ImageMemoryLocation>
{
   private:
    ImageCopyComputeShader* img_copy_shader;

   public:
    ImageDefragger(VulkanBase* base, vk::Device device, BaseChunkAllocator<ImageMemoryLocation>* allocator,
                   uint32_t dealloc_delay, ImageCopyComputeShader* _img_copy_shader);

   protected:
    std::optional<FreeOperation> execute_copy_operation(const DefragOperation& op) override;
};

}  // namespace Saiga::Vulkan::Memory
