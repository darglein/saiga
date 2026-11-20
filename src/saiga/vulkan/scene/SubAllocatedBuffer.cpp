/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/SubAllocatedBuffer.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include <vk_mem_alloc.h>

namespace Saiga
{
namespace Vulkan
{
namespace
{
static_assert(sizeof(VmaVirtualAllocation) <= sizeof(uint64_t), "VmaVirtualAllocation fits into 64 bits");

uint64_t toKey(VmaVirtualAllocation a)
{
    uint64_t v = 0;
    std::memcpy(&v, &a, sizeof(a));
    return v;
}
VmaVirtualAllocation fromKey(uint64_t v)
{
    VmaVirtualAllocation a;
    std::memcpy(&a, &v, sizeof(a));
    return a;
}
}  // namespace

SubAllocatedBuffer::~SubAllocatedBuffer()
{
    destroy();
}

void SubAllocatedBuffer::init(VulkanBase& b, uint32_t elementSize, uint64_t capacity, vk::BufferUsageFlags u,
                              uint64_t maxBytes, const char* what)
{
    destroy();
    base        = &b;
    usage       = u | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    elemSize    = elementSize;
    maxElements = std::max<uint64_t>(maxBytes / elementSize, 1);
    name        = what;
    capacity    = std::clamp<uint64_t>(capacity, 1, maxElements);
    createBlock(capacity);
    current.createBuffer(*base, capacity * elemSize, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
}

void SubAllocatedBuffer::destroy()
{
    destroyBlock();
    ranges.clear();
    current.destroy();
    pendingSource.destroy();
    capacityElements = 0;
}

void SubAllocatedBuffer::createBlock(uint64_t capacity)
{
    VmaVirtualBlockCreateInfo info = {};
    info.size                      = capacity;
    VkResult r                     = vmaCreateVirtualBlock(&info, &block);
    SAIGA_ASSERT(r == VK_SUCCESS, "vmaCreateVirtualBlock failed");
    capacityElements = capacity;
}

void SubAllocatedBuffer::destroyBlock()
{
    if (!block) return;
    vmaClearVirtualBlock(block);
    vmaDestroyVirtualBlock(block);
    block = nullptr;
}

bool SubAllocatedBuffer::tryAllocate(Range& r)
{
    VmaVirtualAllocationCreateInfo info = {};
    info.size                           = r.count;
    info.alignment                      = 1;
    VmaVirtualAllocation allocation     = VK_NULL_HANDLE;
    VkDeviceSize offset                 = 0;
    if (vmaVirtualAllocate(block, &info, &allocation, &offset) != VK_SUCCESS) return false;
    r.allocation = toKey(allocation);
    r.offset     = offset;
    return true;
}

uint64_t SubAllocatedBuffer::allocate(uint64_t key, uint64_t count)
{
    SAIGA_ASSERT(block, "SubAllocatedBuffer: not initialized");
    SAIGA_ASSERT(count > 0);
    SAIGA_ASSERT(ranges.count(key) == 0, "SubAllocatedBuffer: the key is allocated");
    Range r;
    r.count = count;
    if (!tryAllocate(r))
    {
        uint64_t live = count;
        for (auto& [k, x] : ranges) live += x.count;
        grow(live);
        SAIGA_ASSERT(tryAllocate(r), "SubAllocatedBuffer: allocation failed after growing");
    }
    ranges[key] = r;
    return r.offset;
}

void SubAllocatedBuffer::grow(uint64_t required)
{
    auto tooLarge = [&](uint64_t elements)
    {
        SAIGA_EXIT_ERROR("RenderScene: the " + std::string(name) + " need " + std::to_string(elements * elemSize) +
                         " bytes, more than the buffer's limit (maxStorageBufferRange: " +
                         std::to_string(maxElements * elemSize) + " bytes)");
    };
    if (required > maxElements) tooLarge(required);
    uint64_t capacity = std::min(std::max(capacityElements * 2, required), maxElements);
    for (;;)
    {
        // The data on the GPU stays in the buffer that holds it until recordGrowth copies it (an earlier growth
        // without a recordGrowth in between left the intermediate buffer empty)
        if (!pendingSource)
        {
            pendingSource = std::move(current);
            for (auto& [k, r] : ranges)
            {
                r.copyPending = r.resident;
                r.srcOffset   = r.offset;
            }
        }
        else
        {
            current.destroy();
        }
        destroyBlock();
        createBlock(capacity);
        bool ok = true;
        // Re-allocated in the order of the old offsets: a fresh block packs them
        std::vector<Range*> order;
        for (auto& [k, r] : ranges) order.push_back(&r);
        std::sort(order.begin(), order.end(), [](const Range* a, const Range* b) { return a->offset < b->offset; });
        for (Range* r : order) ok = ok && tryAllocate(*r);
        if (ok) break;
        // A fresh block packs the ranges in order: not expected to fail below the limit
        if (capacity == maxElements) tooLarge(capacity + 1);
        capacity = std::min(capacity * 2, maxElements);
    }
    current.createBuffer(*base, capacity * elemSize, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
    growths++;
}

void SubAllocatedBuffer::free(uint64_t key)
{
    auto it = ranges.find(key);
    SAIGA_ASSERT(it != ranges.end(), "SubAllocatedBuffer: the key is not allocated");
    vmaVirtualFree(block, fromKey(it->second.allocation));
    ranges.erase(it);
}

uint64_t SubAllocatedBuffer::offset(uint64_t key) const
{
    auto it = ranges.find(key);
    SAIGA_ASSERT(it != ranges.end(), "SubAllocatedBuffer: the key is not allocated");
    return it->second.offset;
}

void SubAllocatedBuffer::setResident(uint64_t key)
{
    auto it = ranges.find(key);
    SAIGA_ASSERT(it != ranges.end(), "SubAllocatedBuffer: the key is not allocated");
    it->second.resident = true;
}

bool SubAllocatedBuffer::recordGrowth(vk::CommandBuffer cmd)
{
    if (!pendingSource) return false;
    std::vector<vk::BufferCopy> regions;
    for (auto& [k, r] : ranges)
    {
        if (!r.copyPending) continue;
        regions.emplace_back(r.srcOffset * elemSize, r.offset * elemSize, r.count * elemSize);
        r.copyPending = false;
    }
    if (!regions.empty())
    {
        Memory::SafeAccessor acc(*pendingSource.memoryLocation(), *current.memoryLocation());
        cmd.copyBuffer(pendingSource.memoryLocation()->data, current.memoryLocation()->data, regions);
    }
    // Released after the frames that use it (and this copy) have completed
    pendingSource.destroy();
    return !regions.empty();
}

}  // namespace Vulkan
}  // namespace Saiga
