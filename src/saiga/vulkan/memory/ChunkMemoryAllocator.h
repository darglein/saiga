//
// Created by Peter Eichinger on 10.10.18.
//

#pragma once
#include <vulkan/vulkan.hpp>
#include "saiga/export.h"
#include "saiga/vulkan/memory/ChunkAllocator.h"
#include "saiga/util/imath.h"
#include "saiga/vulkan/memory/MemoryLocation.h"
#include "saiga/vulkan/memory/MemoryAllocatorBase.h"
#include <limits>
using namespace Saiga::Vulkan::Memory;

namespace Saiga{
namespace Vulkan{
namespace Memory{

struct SAIGA_GLOBAL ChunkMemoryAllocator : public MemoryAllocatorBase {
private:
    std::string m_logger;
    ChunkAllocator* m_chunkAllocator;
    vk::Device m_device;
    vk::Buffer m_currentBuffer;
    std::vector<vk::Buffer> m_buffers;
    std::shared_ptr<Chunk> m_currentChunk = nullptr;
    vk::DeviceSize m_currentOffset = 0;

    vk::DeviceSize m_chunkSize;
    vk::DeviceSize m_allocateSize;

    vk::BufferCreateInfo m_bufferCreateInfo;


    vk::DeviceSize m_alignment = std::numeric_limits<vk::DeviceSize>::max();

    void createNewBuffer();
public:
    vk::MemoryPropertyFlags flags;
    vk::BufferUsageFlags usageFlags;

    void init(vk::Device _device, ChunkAllocator* chunkAllocator, const vk::MemoryPropertyFlags &_flags,
                    const vk::BufferUsageFlags &usage, vk::DeviceSize chunkSize = 64* 1024* 1024, const std::string& name = "");


    MemoryLocation allocate(vk::DeviceSize size) override;

    void destroy() {
        for (auto& buffer : m_buffers) {
            m_device.destroy(buffer);
        }
    }
};
}
}
}


