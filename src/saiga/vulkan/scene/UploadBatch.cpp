/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/UploadBatch.h"

#include <algorithm>
#include <cstring>

namespace Saiga
{
namespace Vulkan
{
void UploadBatch::add(Buffer& dst, vk::DeviceSize dstOffset, const void* src, size_t size)
{
    if (size == 0) return;
    SAIGA_ASSERT(dst && dstOffset + size <= dst.size(), "UploadBatch: out of bounds");
    Copy c;
    {
        Memory::SafeAccessor acc(*dst.memoryLocation());
        c.dst = dst.memoryLocation()->data;
    }
    c.dstOffset = dstOffset;
    // 16-byte aligned in the staging buffer
    c.srcOffset = (data.size() + 15) & ~size_t(15);
    c.size      = size;
    data.resize(c.srcOffset + size);
    std::memcpy(data.data() + c.srcOffset, src, size);
    copies.push_back(c);
}

void UploadBatch::record(VulkanBase& base, vk::CommandBuffer cmd, Buffer& staging)
{
    if (copies.empty()) return;
    if (staging.size() < data.size())
    {
        // max(2x, required); the old buffer through deferDelete
        size_t size = std::max<size_t>(data.size(), size_t(staging.size()) * 2);
        staging.createBuffer(base, size, vk::BufferUsageFlagBits::eTransferSrc,
                             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
    std::memcpy(staging.getMappedPointer(), data.data(), data.size());
    vk::Buffer src;
    {
        Memory::SafeAccessor acc(*staging.memoryLocation());
        src = staging.memoryLocation()->data;
    }
    // Grouped by destination, in the order of the queue
    std::stable_sort(copies.begin(), copies.end(),
                     [](const Copy& a, const Copy& b) { return VkBuffer(a.dst) < VkBuffer(b.dst); });
    std::vector<vk::BufferCopy> regions;
    for (size_t i = 0; i < copies.size(); ++i)
    {
        regions.emplace_back(copies[i].srcOffset, copies[i].dstOffset, copies[i].size);
        if (i + 1 == copies.size() || copies[i + 1].dst != copies[i].dst)
        {
            cmd.copyBuffer(src, copies[i].dst, regions);
            regions.clear();
        }
    }
    data.clear();
    copies.clear();
}

}  // namespace Vulkan
}  // namespace Saiga
