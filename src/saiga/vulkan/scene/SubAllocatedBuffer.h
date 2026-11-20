/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/buffer/Buffer.h"

#include <cstdint>
#include <map>

// VMA's handle (vk_mem_alloc.h declares it the same way); no saiga header includes VMA
typedef struct VmaVirtualBlock_T* VmaVirtualBlock;

namespace Saiga
{
namespace Vulkan
{
/**
 * A device-local buffer sub-allocated in element units with a VMA virtual block (S18, docs/engine_design.md §10.1):
 * the vertex and index buffers of the mesh pool and the static instance buffer of RenderScene. Each range belongs to a
 * key (a mesh, an instance set).
 *
 * Growth: a virtual block has a fixed size and offsets can not be chosen, so an allocation that does not fit creates a
 * new buffer of max(2x, required) elements and a new block, re-allocates every live range in it (new offsets, read
 * them again with offset()) and records the copy of the ranges whose data is already on the GPU (resident) for the next
 * recordGrowth(); the old buffer is released through deferDelete after the copy. Several growths before a recordGrowth
 * copy from the buffer that holds the data. Main thread only.
 */
class SAIGA_VULKAN_API SubAllocatedBuffer
{
   public:
    SubAllocatedBuffer() = default;
    ~SubAllocatedBuffer();
    SubAllocatedBuffer(const SubAllocatedBuffer&)            = delete;
    SubAllocatedBuffer& operator=(const SubAllocatedBuffer&) = delete;

    void init(VulkanBase& base, uint32_t elementSize, uint64_t capacity, vk::BufferUsageFlags usage);
    void destroy();

    // Allocates count (> 0) elements for the key (not allocated yet); grows if needed. Returns the first element.
    uint64_t allocate(uint64_t key, uint64_t count);
    void free(uint64_t key);
    // The first element of the key's range (changes with a growth)
    uint64_t offset(uint64_t key) const;
    // The data of the key's range is on the GPU (its upload was recorded): a growth copies it
    void setResident(uint64_t key);

    // Records the copy of a pending growth (outside of a render pass). Returns true if a copy was recorded; the caller
    // makes it visible (transfer write) before later writes to the same ranges.
    bool recordGrowth(vk::CommandBuffer cmd);

    Buffer& buffer() { return current; }
    uint64_t capacity() const { return capacityElements; }
    uint32_t elementSize() const { return elemSize; }
    // Incremented by every growth (offsets changed)
    uint64_t generation() const { return growths; }

   private:
    struct Range
    {
        uint64_t allocation = 0;  // VmaVirtualAllocation
        uint64_t offset = 0, count = 0;
        bool resident = false;
        // Resident in the source of the pending growth at srcOffset
        bool copyPending   = false;
        uint64_t srcOffset = 0;
    };

    VulkanBase* base = nullptr;
    vk::BufferUsageFlags usage;
    uint32_t elemSize         = 1;
    uint64_t capacityElements = 0;
    uint64_t growths          = 0;
    VmaVirtualBlock block     = nullptr;
    Buffer current;
    // The buffer that holds the resident data until recordGrowth copies it (empty without a pending growth)
    Buffer pendingSource;
    std::map<uint64_t, Range> ranges;

    bool tryAllocate(Range& r);
    void grow(uint64_t required);
    void createBlock(uint64_t capacity);
    void destroyBlock();
};

}  // namespace Vulkan
}  // namespace Saiga
