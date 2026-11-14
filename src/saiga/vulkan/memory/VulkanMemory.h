/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/util/assert.h"
#include "saiga/vulkan/Parameters.h"
#include "saiga/vulkan/svulkan.h"

#include "BufferMemoryLocation.h"
#include "ImageMemoryLocation.h"
#include "MemoryType.h"

#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include <unordered_map>

// VMA handles (vk_mem_alloc.h declares them the same way). The saiga headers do not include VMA.
typedef struct VmaAllocator_T* VmaAllocator;
typedef struct VmaPool_T* VmaPool;
typedef struct VmaDefragmentationContext_T* VmaDefragmentationContext;

namespace Saiga::Vulkan
{
struct VulkanBase;
}
namespace Saiga::Vulkan::Memory
{
// The sum over all defragmentation passes (VmaDefragmentationStats).
struct DefragmentationStats
{
    uint64_t bytesMoved              = 0;
    uint64_t bytesFreed              = 0;
    uint32_t allocationsMoved        = 0;
    uint32_t deviceMemoryBlocksFreed = 0;
};

/**
 * The memory allocator of the Vulkan module, a thin layer over the Vulkan Memory Allocator (VMA,
 * submodules/VulkanMemoryAllocator).
 *
 * Every buffer and image gets its own VMA allocation (sub-allocated from larger memory blocks or a dedicated
 * allocation, chosen by VMA) and its own vk::Buffer / vk::Image. All functions are thread safe.
 *
 * Defragmentation (VMA's defragmentation API): only dynamic locations are moved, see mark_dynamic(). They live in
 * separate VMA pools (one per memory type), because VMA can not defragment pools with immovable allocations well.
 * VMA chooses the moves; this class recreates the buffers/images at the new place, copies them on the main queue and
 * swaps the handles in the location (see BaseMemoryLocation).
 *  - full_defrag() defragments everything and blocks.
 *  - With enableDefragmentation, update() (called after every frame) runs one incremental pass of at most
 *    defragBytesPerFrame bytes per frame, after a dynamic location was freed.
 */
class SAIGA_VULKAN_API VulkanMemory
{
   public:
    // Incremental defragmentation after every frame (VulkanParameters::enableDefragmentation).
    bool enableDefragmentation = false;
    // The maximum number of bytes an incremental defragmentation pass moves.
    vk::DeviceSize defragBytesPerFrame = 16 * 1024 * 1024;

    VulkanMemory() = default;
    ~VulkanMemory();
    VulkanMemory(const VulkanMemory&)            = delete;
    VulkanMemory& operator=(const VulkanMemory&) = delete;

    void init(VulkanBase* base, uint32_t swapchain_frames, const VulkanParameters& parameters);

    // Frees the remaining allocations and destroys the allocator. The GPU must be idle.
    void destroy();

    void renderGUI();

    /**
     * Creates a buffer with its own memory. Host visible memory is mapped persistently (mappedPointer). The usage
     * always includes transfer src/dst, so that the buffer can be moved by the defragmentation.
     */
    BufferMemoryLocation* allocate(const BufferType& type, vk::DeviceSize size);

    /**
     * Creates the image (from image_data.image_create_info, the image must not exist yet), its memory, the view and
     * the sampler (if image_data has a sampler create info). image_data is moved into the location.
     */
    ImageMemoryLocation* allocate(const ImageType& type, ImageData& image_data);

    // Frees the location immediately: the GPU must not use it anymore (see VulkanBase::deferDelete).
    void deallocate(BufferMemoryLocation* location);
    void deallocate(ImageMemoryLocation* location);

    /**
     * Allows the defragmentation to move the location: it is moved (copied, blocking) into a defragmentable pool. Only
     * locations that are not host visible can be dynamic (a moved mapping would invalidate pointers held by the
     * application), images additionally need the usage transfer src and dst. Returns false if the location stays
     * static.
     *
     * Moving changes the handles in location->data: read them again for every use, or use a DynamicDescriptorSet, which
     * is updated after a move (other descriptor sets are not). A move copies an image from the layout tracked in
     * ImageData::layout and restores it, so the tracked layout must be correct after the submitted command buffers
     * (Texture::transitionImageLayout / setImageLayout).
     */
    bool mark_dynamic(BufferMemoryLocation* location);
    bool mark_dynamic(ImageMemoryLocation* location);

    // Flushes a host visible, non coherent range (relative to the location).
    void flush(BufferMemoryLocation* location, vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE);

    /**
     * Called after every frame was submitted (VulkanBase::finish_frame). Advances the incremental defragmentation if
     * enableDefragmentation is set.
     */
    void update();

    /**
     * Defragments all dynamic locations until VMA finds no more moves. Blocks until the copies are complete. Call it
     * from the thread that submits the frames, outside of the frame recording: command buffers that were recorded but
     * not submitted yet may still reference the old handles.
     */
    void full_defrag();

    DefragmentationStats getDefragmentationStats();

    // The number of moves of the incremental defragmentation pass that is in flight (0 if there is none).
    uint32_t pendingDefragmentationMoves();

    // The memory used by dynamic locations (the defragmentable pools): allocated blocks and used bytes.
    std::pair<vk::DeviceSize, vk::DeviceSize> dynamicMemoryUsage();

    VmaAllocator getAllocator() const { return allocator; }

   private:
    // A move of the current defragmentation pass (VmaDefragmentationMove i of the pass).
    struct Move
    {
        // One of them is set.
        BufferMemoryLocation* buffer = nullptr;
        ImageMemoryLocation* image   = nullptr;
        // The handles at the old place, destroyed at the end of the pass.
        BufferData oldBuffer = nullptr;
        ImageData oldImage   = nullptr;
        // The location was deallocated during the pass: VMA frees its memory at the end of the pass.
        bool destroyed = false;
    };

    // The pass of defragContext whose copies were submitted. The copies are complete when fence is signaled.
    struct Pass
    {
        bool active = false;
        // VmaDefragmentationPassMoveInfo::pMoves (owned by the context) and moveCount.
        void* pMoves       = nullptr;
        uint32_t moveCount = 0;
        std::vector<Move> moves;
        vk::CommandBuffer cmd;
        vk::Fence fence;
    };

    VulkanBase* base = nullptr;
    vk::Device device;
    VmaAllocator allocator = nullptr;

    // Protects the location maps, the pools and the defragmentation state.
    std::mutex mutex;

    std::unordered_map<BufferMemoryLocation*, std::unique_ptr<BufferMemoryLocation>> buffers;
    std::unordered_map<ImageMemoryLocation*, std::unique_ptr<ImageMemoryLocation>> images;

    // The defragmentable pool of each memory type index.
    std::map<uint32_t, VmaPool> movablePools;

    // Incremental defragmentation: the pools that still have to be defragmented, the current context and pass.
    std::vector<VmaPool> defragQueue;
    VmaDefragmentationContext defragContext = nullptr;
    Pass pass;
    // A dynamic location was freed since the last defragmentation.
    bool defragRequested = false;
    // The GUI requested a full defragmentation, which runs in the next update().
    bool fullDefragRequested = false;

    DefragmentationStats defragStats;

    VmaPool getMovablePool(uint32_t memoryTypeIndex);

    // Records the copy of a location to its new place.
    void recordBufferCopy(vk::CommandBuffer cmd, vk::Buffer src, vk::Buffer dst, vk::DeviceSize size);
    void recordImageCopy(vk::CommandBuffer cmd, ImageData& src, ImageData& dst);

    // Creates an image with the create info of src (not bound to memory, layout undefined).
    ImageData unboundImageCopy(const ImageData& src);
    // Binds the image of unboundImageCopy() to the allocation and creates the view and sampler.
    void bindImageCopy(ImageData& dst, VmaAllocation allocation);

    // The defragmentation functions require mutex.

    // Begins defragContext for the next pool of defragQueue. Returns false if the queue is empty.
    bool beginDefragmentation(bool full);
    /**
     * Begins a pass of defragContext: creates the resources at the new places, submits the copies and swaps the
     * handles. wait: waits for the copies and ends the pass. Returns false if the context has no more moves.
     */
    bool beginPass(bool wait);
    // Ends the pass, the copies must be complete. Returns false if the context has no more moves.
    bool endPass();
    // Waits for the copies of the active pass and ends it.
    bool finishPass();
    void endDefragmentation();
    void fullDefragLocked();

    // Destroys the handles and frees the memory.
    void freeLocation(BufferMemoryLocation* location);
    void freeLocation(ImageMemoryLocation* location);
};

}  // namespace Saiga::Vulkan::Memory
