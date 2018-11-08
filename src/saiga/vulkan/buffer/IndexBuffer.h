/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/Base.h"

namespace Saiga {
namespace Vulkan {




template<class IndexType> struct IndexVKType;
template<> struct IndexVKType<uint16_t>
{ static const vk::IndexType value =  vk::IndexType::eUint16;};
template<> struct IndexVKType<uint32_t>
{ static const vk::IndexType value =  vk::IndexType::eUint32;};



template<typename IndexType = uint32_t>
class SAIGA_TEMPLATE IndexBuffer : public Buffer
{
    static_assert(std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value,
                  "Only unsigned integral types allowed!");
    static_assert(sizeof(IndexType)==2 || sizeof(IndexType)==4,
                  "Only 2 and 4 byte index types allowed!");

public:

    using VKType = IndexVKType<IndexType>;

    int indexCount = 0;

    void init(
            VulkanBase& base,
            int count,
            vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eHostVisible| vk::MemoryPropertyFlagBits::eHostCoherent
            )
    {
        indexCount = count;
        size_t indexBufferSize = indexCount * sizeof(IndexType);
        createBuffer(base,indexBufferSize,vk::BufferUsageFlagBits::eIndexBuffer|vk::BufferUsageFlagBits::eTransferDst);
        allocateMemoryBuffer(base,flags);
    }


    void init(VulkanBase& base, const std::vector<IndexType> &indices)
    {
        indexCount = indices.size();
        size_t indexBufferSize = indexCount * sizeof(IndexType);
        createBuffer(base,indexBufferSize,vk::BufferUsageFlagBits::eIndexBuffer);
        allocateMemoryBuffer(base);
        DeviceMemory::mappedUpload(0,size,indices.data());
    }

    void initDeviceLocal(VulkanBase& base, const std::vector<IndexType>& indices)
    {
        init(base,indices.size(),vk::MemoryPropertyFlagBits::eDeviceLocal);
        stagedUpload(base,0,indices.size()*sizeof(IndexType),indices.data());
    }


    void bind(vk::CommandBuffer &cmd, vk::DeviceSize offset = 0)
    {
        cmd.bindIndexBuffer(buffer, offset, VKType::value);
    }

    void draw(vk::CommandBuffer &cmd)
    {
        cmd.drawIndexed(indexCount,1,0,0,0);
    }
};

}
}
