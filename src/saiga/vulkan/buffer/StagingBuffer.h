/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "Buffer.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/Vertex.h"
namespace Saiga
{
namespace Vulkan
{
class SAIGA_GLOBAL StagingBuffer : public Buffer
{
   public:
    void init(VulkanBase& base, size_t size, const void* data = nullptr)
    {
        createBuffer(base, size, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eHostCoherent);

        if (data) m_memoryLocation.upload(base.device, data, size);
    }
};

}  // namespace Vulkan
}  // namespace Saiga
