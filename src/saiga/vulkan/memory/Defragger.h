//
// Created by Peter Eichinger on 2019-01-21.
//

#pragma once
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/Thread/threadName.h"
#include "saiga/vulkan/CommandPool.h"
#include "saiga/vulkan/Queue.h"

#include "BufferMemoryLocation.h"
#include "Chunk.h"
#include "ChunkAllocator.h"
#include "FitStrategy.h"
#include "ImageMemoryLocation.h"

#include <atomic>
#include <deque>
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
    float weight_location   = 1.0f;
    float weight_small_free = 0.5f;
    uint32_t max_targets    = 3;
    float size_small_factor = 0.001f;
    float size_limit_factor = 0.01;

    vk::DeviceSize size_small, size_limit;

    void update_sizes(vk::DeviceSize chunkSize)
    {
        size_small = static_cast<vk::DeviceSize>(size_small_factor * chunkSize);
        size_limit = static_cast<vk::DeviceSize>(size_limit_factor * chunkSize);
    }
};

template <typename T>
class Defragger
{
   public:
    template <typename PType>
    class PointerOutput
    {
       private:
        PType* type;

       public:
        explicit PointerOutput(PType* _type) : type(_type) {}

        inline friend std::ostream& operator<<(std::ostream& os, const PointerOutput& output)
        {
            if (output.type)
            {
                os << *(output.type);
            }
            else
            {
                os << "nullptr";
            }
            return os;
        }
    };
    struct Target
    {
        Target(vk::DeviceMemory _mem, vk::DeviceSize _offset, vk::DeviceSize _size)
            : memory(_mem), offset(_offset), size(_size)
        {
        }

        Target() = default;

        vk::DeviceMemory memory;
        vk::DeviceSize offset, size;

        friend std::ostream& operator<<(std::ostream& os, const Target& target)
        {
            os << std::hex << target.memory << std::dec << " " << target.offset << "-" << target.size;
            return os;
        }

        inline explicit operator FreeListEntry() const { return FreeListEntry{offset, size}; }
    };
    struct PossibleOp
    {
        PossibleOp(Target _target, T* _src, float _weight) : target(_target), source(_src), weight(_weight) {}

        Target target;
        T* source;
        float weight;

        friend std::ostream& operator<<(std::ostream& os, const PossibleOp& op)
        {
            os << op.target << "<-" << PointerOutput<T>(op.source) << " w: " << op.weight;
            return os;
        }
    };
    struct DefragOp
    {
        DefragOp(T* _target, T* src, vk::CommandBuffer _cmd) : target(_target), source(src), cmd(_cmd) {}

        T *target, *source;
        vk::CommandBuffer cmd;

        friend std::ostream& operator<<(std::ostream& os, const DefragOp& op)
        {
            os << PointerOutput<T>(op.target) << "<-" << PointerOutput<T>(op.source) << " cmd: " << op.cmd;
            return os;
        }
    };
    struct CopyOp
    {
        CopyOp(T* _target, T* src, vk::CommandBuffer _cmd, vk::Fence _fence, vk::Semaphore _wait, vk::Semaphore _signal)
            : target(_target), source(src), cmd(_cmd), fence(_fence), wait_semaphore(_wait), signal_semaphore(_signal)
        {
        }
        T *target, *source;
        vk::CommandBuffer cmd;
        vk::Fence fence;
        vk::Semaphore wait_semaphore, signal_semaphore;

        friend std::ostream& operator<<(std::ostream& os, const CopyOp& op)
        {
            os << PointerOutput<T>(op.target) << "<-" << PointerOutput<T>(op.source) << " cmd: " << op.cmd
               << " fence: " << op.fence << " wsem: " << op.wait_semaphore << " ssem: " << op.signal_semaphore;
            return os;
        }
    };
    struct FreeOp
    {
        FreeOp(T* _target, T* src, uint64_t _delay) : target(_target), source(src), delay(_delay) {}
        T *target, *source;
        uint64_t delay;

        friend std::ostream& operator<<(std::ostream& os, const FreeOp& op)
        {
            os << PointerOutput<T>(op.target) << "<-" << PointerOutput<T>(op.source) << " delay: " << op.delay;
            return os;
        }
    };

    std::vector<std::vector<vk::DeviceSize>> all_free;
    VulkanBase* base;
    uint64_t dealloc_delay;
    vk::Device device;
    ChunkAllocator<T>* allocator;

    std::vector<PossibleOp> possibleOps;
    std::list<DefragOp> defragOps;
    std::deque<CopyOp> copyOps;
    std::list<FreeOp> freeOps;
    std::set<T*> currentDefragSources;

    bool valid, enabled;
    std::atomic_bool running, quit;
    std::atomic_uint64_t frame_number;

    std::mutex start_mutex, running_mutex, defrag_mutex, copy_mutex;
    std::condition_variable start_condition;
    std::thread worker;

    CommandPool commandPool;

    void worker_func();


    vk::QueryPool queryPool;
    double kbPerNanoSecond = std::numeric_limits<double>::infinity();

    // Defrag thread functions

    void run();

    bool find_defrag_ops();
    void fill_free_list();
    bool create_copy_commands();

    void perform_single_defrag(DefragOp& op, vk::Semaphore semaphore);

    bool complete_copy_commands();

    bool perform_free_operations();

    // end defrag thread functions
   public:
    DefraggerConfiguration config;

    Defragger(VulkanBase* _base, vk::Device _device, ChunkAllocator<T>* _allocator, uint32_t _dealloc_delay = 0);


    Defragger(const Defragger& other) = delete;
    Defragger& operator=(const Defragger& other) = delete;

    virtual ~Defragger() { exit(); }

    void exit();

    void start();
    void stop();

    std::pair<bool, int64_t> perform_defrag(int64_t allowed_time, vk::Semaphore semaphore);

    void setEnabled(bool enable) { enabled = enable; }

    void invalidate(T* memoryLocation);

    void update(uint32_t _frame_number) { frame_number = _frame_number; }


   protected:
    virtual void create_copy_command(PossibleOp& op, T*, vk::CommandBuffer cmd) = 0;

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
        if (size == 0 || size >= cap)
        {
            return 0;
        }

        if (size <= config.size_small)
        {
            return mix(0.0f, config.weight_small_free, static_cast<float>(size) / config.size_small);
        }

        return mix(config.weight_small_free, 0.0f,
                   static_cast<float>(size - config.size_small) / (config.size_limit - config.size_small));
    }

    inline float get_target_weight(uint32_t chunkIndex, vk::DeviceSize chunkSize, vk::DeviceSize begin,
                                   vk::DeviceSize first, vk::DeviceSize second) const
    {
        return config.weight_location * (chunkIndex + (static_cast<float>(begin) / chunkSize)) +
               get_free_weight(first) + get_free_weight(second);
    }

    template <typename Iter>
    inline float get_weight(uint32_t chunkIndex, vk::DeviceSize chunkSize, Iter alloc, vk::DeviceSize first,
                            vk::DeviceSize second) const
    {
        return config.weight_location * (chunkIndex + (static_cast<float>(begin(alloc)) / chunkSize)) +
               get_free_weight(first) + get_free_weight(second);
    }

    template <typename Iter>
    inline T* get(Iter iter) const
    {
        return (*iter).get();
    }

    inline bool anyOperationsRemaining() const
    {
        return !possibleOps.empty() || !defragOps.empty() || !copyOps.empty() || !freeOps.empty();
    }
};



class BufferDefragger final : public Defragger<BufferMemoryLocation>
{
   public:
    BufferDefragger(VulkanBase* base, vk::Device device, ChunkAllocator<BufferMemoryLocation>* allocator,
                    uint32_t dealloc_delay)
        : Defragger(base, device, allocator, dealloc_delay)
    {
    }

   protected:
    void create_copy_command(PossibleOp& op, BufferMemoryLocation* reservedSpace, vk::CommandBuffer cmd) override;
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
    ImageDefragger(VulkanBase* base, vk::Device device, ChunkAllocator<ImageMemoryLocation>* allocator,
                   uint32_t dealloc_delay, ImageCopyComputeShader* _img_copy_shader);

   protected:
    void create_copy_command(PossibleOp& op, ImageMemoryLocation* reservedSpace, vk::CommandBuffer cmd) override;
};

}  // namespace Saiga::Vulkan::Memory
