//
// Created by Peter Eichinger on 10.10.18.
//

#pragma once
#include "saiga/util/assert.h"

#include "BufferChunkAllocator.h"
#include "ChunkCreator.h"
#include "ImageChunkAllocator.h"
#include "MemoryType.h"
#include "SimpleMemoryAllocator.h"

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <vulkan/vulkan.hpp>

#include <unordered_map>

namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
class SAIGA_GLOBAL VulkanMemory
{
   private:
    using BufferMap        = std::map<BufferType, std::shared_ptr<BaseMemoryAllocator>>;
    using ImageMap         = std::map<ImageType, ImageChunkAllocator>;
    using BufferDefaultMap = std::map<BufferType, vk::DeviceSize>;
    using ImageDefaultMap  = std::map<ImageType, vk::DeviceSize>;
    using BufferIter       = BufferMap::iterator;
    using ImageIter        = ImageMap::iterator;

    vk::PhysicalDevice m_pDevice;
    vk::Device m_device;


    std::vector<vk::MemoryType> memoryTypes;
    const vk::BufferUsageFlags all_buffer_usage = static_cast<vk::BufferUsageFlags>(VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM);
    const vk::ImageUsageFlags all_image_usage   = static_cast<vk::ImageUsageFlags>(VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM);
    const vk::MemoryPropertyFlags all_mem_properties =
        static_cast<vk::MemoryPropertyFlagBits>(VK_MEMORY_PROPERTY_FLAG_BITS_MAX_ENUM);

    const vk::DeviceSize fallback_buffer_chunk_size = 64 * 1024 * 1024;
    const vk::DeviceSize fallback_image_chunk_size  = 256 * 1024 * 1024;


    template <typename T>
    inline bool allocator_valid(const MemoryType<T> allocator_type, const MemoryType<T>& type) const
    {
        return ((allocator_type.usageFlags & type.usageFlags) == type.usageFlags) &&
               (allocator_type.memoryFlags == type.memoryFlags);
    }

    template <typename T>
    inline bool default_valid(const MemoryType<T> allocator_type, const MemoryType<T>& type) const
    {
        return ((allocator_type.usageFlags & type.usageFlags) == type.usageFlags) &&
               ((allocator_type.memoryFlags & type.memoryFlags) == type.memoryFlags);
    }


    BufferDefaultMap default_buffer_chunk_sizes{
        {BufferType{vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,
                    all_mem_properties},
         1024 * 1024},
        {BufferType{all_buffer_usage, all_mem_properties}, fallback_buffer_chunk_size}};

    ImageDefaultMap default_image_chunk_sizes{
        {ImageType{all_image_usage, all_mem_properties}, fallback_image_chunk_size}};


    BufferMap bufferAllocators;
    ImageMap imageAllocators;

    ChunkCreator chunkAllocator;
    FirstFitStrategy strategy;


    template <typename DefaultSizeMap, typename MemoryType>
    inline typename DefaultSizeMap::const_iterator find_default_size(const DefaultSizeMap& defaultSizes,
                                                                     const MemoryType& type)
    {
        const auto sizes_begin = defaultSizes.cbegin();
        const auto sizes_end   = defaultSizes.cend();
        auto found = std::find_if(sizes_begin, sizes_end, [&](typename DefaultSizeMap::const_reference entry) {
            return (default_valid(entry.first, type));
        });

        SAIGA_ASSERT(found != defaultSizes.cend(), "No default size found. At least a fallback size must be added.");
        return found;
    }

    BufferIter createNewBufferAllocator(BufferMap& map, const BufferDefaultMap& defaultSizes, const BufferType& type);

    ImageIter createNewImageAllocator(ImageMap& map, const ImageDefaultMap& defaultSizes, const ImageType& type);

    template <typename Map, typename UsageType>
    inline typename Map::iterator findAllocator(Map& map, const MemoryType<UsageType>& memoryType)
    {
        const auto begin = map.begin();
        const auto end   = map.end();

        return std::find_if(begin, end,
                            [=](typename Map::reference entry) { return allocator_valid(entry.first, memoryType); });
    }

   public:
    void init(vk::PhysicalDevice _pDevice, vk::Device _device);

    BaseMemoryAllocator& getAllocator(const vk::BufferUsageFlags& usage,
                                      const vk::MemoryPropertyFlags& flags = vk::MemoryPropertyFlagBits::eDeviceLocal);

    BaseMemoryAllocator& getImageAllocator(
        const vk::MemoryPropertyFlags& flags = vk::MemoryPropertyFlagBits::eDeviceLocal,
        const vk::ImageUsageFlags& usage     = vk::ImageUsageFlagBits::eSampled);


    void destroy();

    void renderGUI();

   private:
    inline vk::MemoryPropertyFlags getEffectiveFlags(vk::MemoryPropertyFlags flags) const
    {
        for (const auto& memory : memoryTypes)
        {
            if ((memory.propertyFlags & flags) == flags)
            {
                return memory.propertyFlags;
            }
        }
        SAIGA_ASSERT(false, "No valid memory found");
        return vk::MemoryPropertyFlags();
    }
};


}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
