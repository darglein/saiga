//
// Created by Peter Eichinger on 2019-01-21.
//

#pragma once
#include "saiga/core/util/Thread/threadName.h"
#include "saiga/core/util/easylogging++.h"
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
#include <unordered_map>

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
        vk::DeviceSize offset = 0, size = 0;

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
    // A recorded copy, which is submitted by the main thread (perform_defrag).
    struct DefragOp
    {
        DefragOp(T* _target, T* src, vk::CommandBuffer _cmd, vk::QueryPool _queryPool)
            : target(_target), source(src), cmd(_cmd), queryPool(_queryPool)
        {
        }

        T *target, *source;
        vk::CommandBuffer cmd;
        // Two timestamps (begin/end) of this copy, may be null.
        vk::QueryPool queryPool;

        friend std::ostream& operator<<(std::ostream& os, const DefragOp& op)
        {
            os << PointerOutput<T>(op.target) << "<-" << PointerOutput<T>(op.source) << " cmd: " << op.cmd;
            return os;
        }
    };
    // A submitted copy. It is complete when copyTimeline reaches signal_value.
    struct CopyOp
    {
        CopyOp(T* _target, T* src, vk::CommandBuffer _cmd, vk::QueryPool _queryPool, uint64_t _signal_value)
            : target(_target), source(src), cmd(_cmd), queryPool(_queryPool), signal_value(_signal_value)
        {
        }
        T *target, *source;
        vk::CommandBuffer cmd;
        vk::QueryPool queryPool;
        uint64_t signal_value;

        friend std::ostream& operator<<(std::ostream& os, const CopyOp& op)
        {
            os << PointerOutput<T>(op.target) << "<-" << PointerOutput<T>(op.source) << " cmd: " << op.cmd
               << " signal: " << op.signal_value;
            return os;
        }
    };
    // A completed (and swapped) copy. target holds the old memory of source, which is freed when the frame timeline
    // reaches wait_value (frames in flight may still use it).
    struct FreeOp
    {
        FreeOp(T* _target, T* src, uint64_t _wait_value) : target(_target), source(src), wait_value(_wait_value) {}
        T *target, *source;
        uint64_t wait_value;

        friend std::ostream& operator<<(std::ostream& os, const FreeOp& op)
        {
            os << PointerOutput<T>(op.target) << "<-" << PointerOutput<T>(op.source) << " wait: " << op.wait_value;
            return os;
        }
    };

    std::vector<std::vector<vk::DeviceSize>> all_free;
    VulkanBase* base;
    vk::Device device;
    ChunkAllocator<T>* allocator;

    // Only accessed by the worker, or by the main thread while the worker is stopped.
    std::vector<PossibleOp> possibleOps;
    // Protected by defrag_mutex.
    std::list<DefragOp> defragOps;
    // Protected by copy_mutex.
    std::deque<CopyOp> copyOps;
    // Only accessed by the worker, or by the main thread while the worker is stopped.
    std::list<FreeOp> freeOps;
    std::set<T*> currentDefragSources;

    std::atomic_bool valid, enabled;
    std::atomic_bool running, quit;
    bool exited = false;

    std::mutex start_mutex, running_mutex, defrag_mutex, copy_mutex;
    // Number of active pause() calls, guarded by start_mutex.
    int pauseCount = 0;
    std::condition_variable start_condition;

    // The worker waits on this while it can only wait for the main thread (submission) or the GPU.
    std::mutex wait_mutex;
    std::condition_variable wait_condition;

    std::thread worker;

    CommandPool commandPool;

    // Chains the copies: copy i waits for value i - 1 and signals i. Protected by copy_mutex.
    vk::Semaphore copyTimeline;
    uint64_t copyCounter = 0;

    // Timestamps of the main queue. timestampMask == 0: not supported.
    uint64_t timestampMask = 0;
    double timestampPeriod = 1.0;  // ns per tick

    std::atomic<double> kbPerNanoSecond;

    void worker_func();

    // Defrag thread functions

    void run();

    bool find_defrag_ops();
    void fill_free_list();
    // Returns true if new copies were recorded.
    bool create_copy_commands();

    void perform_single_defrag(DefragOp& op, vk::Semaphore frameTimeline, uint64_t waitValue);

    // Submits defrag ops. Requires defrag_mutex.
    std::pair<bool, int64_t> submit_defrag_ops(int64_t allowed_time, vk::Semaphore frameTimeline, uint64_t waitValue);

    // Returns true if copies were completed.
    bool complete_copy_commands();

    // Returns true if memory was freed. force: free without waiting for the frames.
    bool perform_free_operations(bool force);

    void update_copy_speed(const CopyOp& copy);

    vk::QueryPool create_query_pool();

    // Frees the command buffer and the query pool of an op and the resources of create_copy_command.
    void release_op_resources(vk::CommandBuffer cmd, vk::QueryPool queryPool);

    // end defrag thread functions
   public:
    DefraggerConfiguration config;

    Defragger(VulkanBase* _base, vk::Device _device, ChunkAllocator<T>* _allocator);


    Defragger(const Defragger& other)            = delete;
    Defragger& operator=(const Defragger& other) = delete;

    // Derived classes must call exit() in their destructor, the worker calls their virtual functions.
    virtual ~Defragger();

    // Stops the worker and frees all pending operations. Idempotent.
    void exit();

    void start();
    void stop();

    /**
     * Counted pause for allocate/deallocate, which may run on several threads: the worker stays stopped (also
     * against start() from other threads or perform_defrag) until every pause() was matched by a resume().
     */
    void pause();
    void resume();

    /**
     * Submits the recorded copies, which take at most allowed_time (ns). They wait for frameTimeline >= waitValue.
     * Called by the main thread.
     */
    std::pair<bool, int64_t> perform_defrag(int64_t allowed_time, vk::Semaphore frameTimeline, uint64_t waitValue);

    /**
     * Defragments until nothing can be moved anymore. Blocks until all copies are complete and the old memory is
     * freed. Must be called by the thread that submits the frames, outside of the frame recording (it waits for all
     * submitted frames and frees memory they used).
     */
    void full_defrag();

    void setEnabled(bool enable) { enabled = enable; }

    void invalidate(T* memoryLocation);

   protected:
    // Records the copy from op.source to the reserved space. Returns false if the copy could not be recorded.
    virtual bool create_copy_command(PossibleOp& op, T*, vk::CommandBuffer cmd) = 0;

    // Releases per copy resources of create_copy_command (called when the copy is complete or dropped).
    virtual void release_copy_resources(vk::CommandBuffer cmd) {}

    // Returns false if the location can not be moved by this defragger.
    virtual bool is_movable(T* location) { return true; }

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
};



class BufferDefragger final : public Defragger<BufferMemoryLocation>
{
   public:
    BufferDefragger(VulkanBase* base, vk::Device device, ChunkAllocator<BufferMemoryLocation>* allocator)
        : Defragger(base, device, allocator)
    {
    }

    ~BufferDefragger() override;

   protected:
    bool create_copy_command(PossibleOp& op, BufferMemoryLocation* reservedSpace, vk::CommandBuffer cmd) override;
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
    std::mutex sets_mutex;
    std::unordered_map<vk::CommandBuffer, vk::DescriptorSet, CommandHash> usedSets;
    ImageCopyComputeShader* img_copy_shader;

   public:
    ImageDefragger(VulkanBase* base, vk::Device device, ChunkAllocator<ImageMemoryLocation>* allocator,
                   ImageCopyComputeShader* _img_copy_shader);

    ~ImageDefragger() override;

   protected:
    bool create_copy_command(PossibleOp& op, ImageMemoryLocation* reservedSpace, vk::CommandBuffer cmd) override;

    void release_copy_resources(vk::CommandBuffer cmd) override;

    bool is_movable(ImageMemoryLocation* location) override;
};

}  // namespace Saiga::Vulkan::Memory
