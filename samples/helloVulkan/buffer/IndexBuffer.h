/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "Buffer.h"
#include "vulkanBase.h"

namespace Saiga {
namespace Vulkan {




template<class IndexType> struct IndexVKType;
template<> struct IndexVKType<uint16_t>
{ static const vk::IndexType value =  vk::IndexType::eUint16;};
template<> struct IndexVKType<uint32_t>
{ static const vk::IndexType value =  vk::IndexType::eUint32;};



template<typename IndexType = uint32_t>
class SAIGA_GLOBAL IndexBuffer : public Buffer
{
    static_assert(std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value,
                  "Only unsigned integral types allowed!");
    static_assert(sizeof(IndexType)==2 || sizeof(IndexType)==4,
                  "Only 2 and 4 byte index types allowed!");

public:

    using VKType = IndexVKType<IndexType>;

    void init(VulkanBase& base,  std::vector<uint32_t> &indices)
    {
        uint32_t indexBufferSize = indices.size() * sizeof(uint32_t);
        createBuffer(base,indexBufferSize,vk::BufferUsageFlagBits::eIndexBuffer);
        allocateMemory(base);
        DeviceMemory::upload(0,indexBufferSize,indices.data());
        base.device.bindBufferMemory(buffer,memory,0);
    }

    void bind(vk::CommandBuffer &cmd, vk::DeviceSize offset = 0)
    {
        cmd.bindIndexBuffer(buffer, offset, VKType::value);
    }
};

}
}
