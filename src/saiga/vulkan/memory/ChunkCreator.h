//
// Created by Peter Eichinger on 08.10.18.
//

#pragma once

#include "ChunkType.h"
#include "MemoryLocation.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
/**
 * Class that allocates chunks of memory for different types.
 * Returns shared pointers to the chunks.
 */
class SAIGA_LOCAL ChunkCreator
{
   private:
    bool m_initialized = false;
    vk::PhysicalDevice m_physicalDevice;
    vk::Device m_device;
    std::vector<ChunkType> m_memoryTypes;

    ChunkType& findMemoryType(vk::MemoryPropertyFlags flags);

   public:
    void init(vk::PhysicalDevice _physicalDevice, vk::Device _device);

    vk::MemoryPropertyFlags getEffectiveFlags(vk::MemoryPropertyFlags memoryFlags);

    std::shared_ptr<Chunk> allocate(vk::MemoryPropertyFlags propertyFlags, vk::DeviceSize chunkSize);

    void deallocate(std::shared_ptr<Chunk> chunk);

    void destroy()
    {
        for (auto& type : m_memoryTypes)
        {
            type.destroy();
        }
    }
};

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
