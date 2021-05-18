/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/Vertex.h"

#include "Buffer.h"
namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API StagingBuffer : public Buffer
{
   public:
    StagingBuffer()                               = default;
    StagingBuffer(const StagingBuffer& other)     = delete;
    StagingBuffer(StagingBuffer&& other) noexcept = default;

    StagingBuffer& operator=(const StagingBuffer& other) = delete;
    StagingBuffer& operator=(StagingBuffer&& other) noexcept = default;

    ~StagingBuffer() override = default;

    void init(VulkanBase& base, size_t size, const void* data = nullptr)
    {
        createBuffer(base, size, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible);

        if (data) m_memoryLocation->upload(base.device, data, size);
    }
};

}  // namespace Vulkan
}  // namespace Saiga
