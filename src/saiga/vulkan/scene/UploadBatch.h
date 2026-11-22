/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/buffer/Buffer.h"

#include <cstdint>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
/**
 * The uploads of one RenderScene frame (S18): the data is collected on the CPU, copied into the frame slot's host
 * visible staging buffer (grown when too small; the slot's previous frame has completed) and recorded as one
 * vkCmdCopyBuffer per destination. The caller records the barriers around record().
 */
class SAIGA_VULKAN_API UploadBatch
{
   public:
    // Queues size bytes for dst at dstOffset (bytes)
    void add(Buffer& dst, vk::DeviceSize dstOffset, const void* data, size_t size);
    // Copies the data into the staging buffer and records the copies; clears the batch
    void record(VulkanBase& base, vk::CommandBuffer cmd, Buffer& staging);
    // The same into a fixed staging buffer at stagingOffset (S20's terrain uploads: several batches per frame in one
    // staging buffer that never grows; asserted to fit); returns the end offset (16-byte aligned) for the next batch
    size_t recordAt(vk::CommandBuffer cmd, Buffer& staging, size_t stagingOffset);
    bool empty() const { return copies.empty(); }
    size_t bytes() const { return data.size(); }

   private:
    struct Copy
    {
        vk::Buffer dst;
        vk::DeviceSize dstOffset = 0, srcOffset = 0, size = 0;
    };
    std::vector<uint8_t> data;
    std::vector<Copy> copies;
};

}  // namespace Vulkan
}  // namespace Saiga
