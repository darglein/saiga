/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/buffer/Buffer.h"
namespace Saiga
{
namespace Vulkan
{
template <class IndexType>
struct IndexVKType;
template <>
struct IndexVKType<uint16_t>
{
    static const vk::IndexType value = vk::IndexType::eUint16;
};
template <>
struct IndexVKType<uint32_t>
{
    static const vk::IndexType value = vk::IndexType::eUint32;
};



template <typename IndexType = uint32_t>
class SAIGA_TEMPLATE IndexBuffer : public Buffer
{
    static_assert(std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value,
                  "Only unsigned integral types allowed!");
    static_assert(sizeof(IndexType) == 2 || sizeof(IndexType) == 4, "Only 2 and 4 byte index types allowed!");

   public:
    IndexBuffer()                             = default;
    IndexBuffer(const IndexBuffer& other)     = delete;
    IndexBuffer(IndexBuffer&& other) noexcept = default;

    IndexBuffer& operator=(const IndexBuffer& other) = delete;
    IndexBuffer& operator=(IndexBuffer&& other) noexcept = default;

    ~IndexBuffer() override = default;

    using VKType = IndexVKType<IndexType>;

    uint32_t indexCount = 0;

    void init(VulkanBase& base, uint32_t count, const vk::MemoryPropertyFlags& flags)
    {
        indexCount             = count;
        size_t indexBufferSize = indexCount * sizeof(IndexType);
        createBuffer(base, indexBufferSize,
                     vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, flags);
    }


    void init(VulkanBase& base, const std::vector<IndexType>& indices, const vk::MemoryPropertyFlags& flags)
    {
        if (indices.size() > std::numeric_limits<uint32_t>::max())
        {
            std::cerr << "Only 32 bit of indices are supported" << std::endl;
            return;
        }
        indexCount             = static_cast<uint32_t>(indices.size());
        size_t indexBufferSize = indexCount * sizeof(IndexType);

        createBuffer(base, indexBufferSize,
                     vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, flags);

        stagedUpload(base, indices.size() * sizeof(IndexType), indices.data());
        //        m_memoryLocation.upload(base.device, indices.data());
    }

    void bind(vk::CommandBuffer& cmd)
    {
        cmd.bindIndexBuffer(m_memoryLocation->data, m_memoryLocation->offset, VKType::value);
    }

    void draw(vk::CommandBuffer& cmd) { cmd.drawIndexed(indexCount, 1, 0, 0, 0); }
};

}  // namespace Vulkan
}  // namespace Saiga
