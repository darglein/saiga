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
class SAIGA_VULKAN_API ObjectBuffer : public Buffer
{
    using pointer         = Obj*;
    using const_pointer   = const Obj*;
    using reference       = Obj&;
    using const_reference = const Obj&;

   public:
    ~ObjectBuffer()
    {
        // Note: Buffer::destroy will be called by the destructor of Buffer
        if (m_memoryLocation) destruct();
    }

    void init(VulkanBase& base, const vk::BufferUsageFlags& usage)
    {
        createBuffer(base, sizeof(Obj), usage,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        // Construct the object in the buffer
        new (get()) Obj();
    }

    void destroy()
    {
        if (m_memoryLocation) destruct();
        Buffer::destroy();
    }

    pointer get() { return static_cast<pointer>(m_memoryLocation->mappedPointer); }
    const_pointer get() const { return static_cast<const_pointer>(m_memoryLocation->mappedPointer); }

    pointer operator->() { return get(); }
    reference operator*() { return *get(); }
    const_reference operator*() const { return *get(); }

   private:
    void destruct() { get()->~Obj(); }
};

/**
 * A spezialization of the above, because this is the most common use-case.
 */
template <typename Obj>
class SAIGA_VULKAN_API CoherentUniformObject : public ObjectBuffer<Obj>
{
   public:
    void init(VulkanBase& base) { ObjectBuffer<Obj>::init(base, vk::BufferUsageFlagBits::eUniformBuffer); }
};

}  // namespace Vulkan
}  // namespace Saiga
