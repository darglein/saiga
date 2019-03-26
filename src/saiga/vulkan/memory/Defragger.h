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
#include <ostream>
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
        T *source, *targetLocation;
        vk::DeviceMemory targetMemory;
        FreeListEntry target;
        float weight;
        vk::CommandBuffer copy_cmd = nullptr;
        bool operator<(const DefragOperation& second) const { return this->weight < second.weight; }

        friend std::ostream& operator<<(std::ostream& os, const DefragOperation& operation)
        {
            os << "src: " << *operation.source << " tLoc: ";

            if (operation.targetLocation)
            {
                os << *operation.targetLocation;
            }
            else
            {
                os << "nullptr";
            }
            os << " tMem: " << operation.targetMemory << " t: " << operation.target << " cmd: " << std::hex
               << operation.copy_cmd << std::dec;
            return os;
        }
    };

    struct FreeOperation
    {
        T *target, *source;
        uint64_t delay;

        friend std::ostream& operator<<(std::ostream& os, const FreeOperation& operation)
        {
            os << "target: ";
            if (operation.target)
                os << *operation.target;
            else
                os << "nullptr";

            os << " source: ";
            if (operation.source)
                os << *operation.source;
            else
                os << "nullptr";
            os << " delay: " << operation.delay;
            return os;
        }
    };

    std::vector<std::vector<vk::DeviceSize>> all_free;
    VulkanBase* base;
    uint64_t dealloc_delay;
    vk::Device device;
    BaseChunkAllocator<T>* allocator;
    std::vector<DefragOperation> defrag_operations;

    std::list<FreeOperation> free_operations;

    bool valid, enabled;
    std::atomic_bool running, quit;
    std::atomic_uint64_t frame_number;

    std::mutex start_mutex, running_mutex, defrag_mutex;
    std::condition_variable start_condition;
    std::thread worker;

    void worker_func();


    vk::QueryPool queryPool;
    double kbPerNanoSecond = std::numeric_limits<double>::infinity();

    // Defrag thread functions

    void run();


    void find_defrag_ops();
    void fill_free_list();
    bool create_copy_commands();
    bool perform_free_operations();

    // end defrag thread functions
   public:
    DefraggerConfiguration config;

    Defragger(VulkanBase* _base, vk::Device _device, BaseChunkAllocator<T>* _allocator, uint32_t _dealloc_delay = 0);


    Defragger(const Defragger& other) = delete;
    Defragger& operator=(const Defragger& other) = delete;

    virtual ~Defragger() { exit(); }

    void exit();

    void start();
    void stop();

    void setEnabled(bool enable) { enabled = enable; }

    void update(uint32_t _frame_number) { frame_number = _frame_number; }

    void perform_single_defrag(DefragOperation& op);
    int64_t perform_defrag(int64_t allowed_time);

   protected:
    virtual bool create_copy_command(DefragOperation& op, vk::CommandBuffer cmd) = 0;

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



class BufferDefragger final : public Defragger<BufferMemoryLocation>
{
   public:
    BufferDefragger(VulkanBase* base, vk::Device device, BaseChunkAllocator<BufferMemoryLocation>* allocator,
                    uint32_t dealloc_delay)
        : Defragger(base, device, allocator, dealloc_delay)
    {
    }

   protected:
    bool create_copy_command(DefragOperation& op, vk::CommandBuffer cmd) override;
};

struct CommandHash
{
    std::size_t operator()(vk::CommandBuffer const& cmd) const noexcept
    {
        return std::hash<uint64_t>{}(reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(cmd)));
    }
};

class ImageDefragger final : public Defragger<ImageMemoryLocation>
{
   private:
    std::unordered_map<vk::CommandBuffer, vk::DescriptorSet, CommandHash> usedSets;
    ImageCopyComputeShader* img_copy_shader;

   public:
    ImageDefragger(VulkanBase* base, vk::Device device, BaseChunkAllocator<ImageMemoryLocation>* allocator,
                   uint32_t dealloc_delay, ImageCopyComputeShader* _img_copy_shader);

   protected:
    bool create_copy_command(DefragOperation& op, vk::CommandBuffer cmd) override;
};

}  // namespace Saiga::Vulkan::Memory
