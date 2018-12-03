/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "Buffer.h"
#include "saiga/vulkan/Base.h"

namespace Saiga
{
namespace Vulkan
{
class SAIGA_GLOBAL UniformBuffer : public Buffer
{
   public:
    UniformBuffer()                               = default;
    UniformBuffer(const UniformBuffer& other)     = delete;
    UniformBuffer(UniformBuffer&& other) noexcept = default;

    UniformBuffer& operator=(const UniformBuffer& other) = delete;
    UniformBuffer& operator=(UniformBuffer&& other) noexcept = default;

    void init(VulkanBase& base, const void* data, size_t size)
    {
        createBuffer(base, size, vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible);

        if (data)
        {
            m_memoryLocation.upload(base.device, data);
        }
    }

    vk::DescriptorBufferInfo getDescriptorInfo()
    {
        vk::DescriptorBufferInfo descriptorInfo;
        descriptorInfo.buffer = m_memoryLocation.buffer;
        descriptorInfo.offset = m_memoryLocation.offset;
        descriptorInfo.range  = m_memoryLocation.size;
        return descriptorInfo;
    }
};

}  // namespace Vulkan
}  // namespace Saiga
