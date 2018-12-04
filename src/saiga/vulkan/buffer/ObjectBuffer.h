//
// Created by Peter Eichinger on 2018-12-04.
//

#pragma once

#include "saiga/export.h"

#include "Buffer.h"

namespace Saiga
{
namespace Vulkan
{
/**
 * Buffer that stores the templated object in a buffer in hostVisible | hostCoherent memory.
 * Object can be accessed via -> and * operators.
 * @tparam Obj Type of object
 */
template <typename Obj>
class SAIGA_GLOBAL ObjectBuffer : public Buffer
{
    using pointer         = Obj*;
    using reference       = Obj&;
    using const_reference = const Obj&;

   public:
    void init(VulkanBase& base, const vk::BufferUsageFlags& usage)
    {
        createBuffer(base, sizeof(Obj), usage,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }


    pointer operator->() { return static_cast<pointer>(m_memoryLocation.mappedPointer); }

    reference operator*() { return *static_cast<pointer>(m_memoryLocation.mappedPointer); }

    const_reference operator*() const { return *static_cast<pointer>(m_memoryLocation.mappedPointer); }
};
}  // namespace Vulkan
}  // namespace Saiga
