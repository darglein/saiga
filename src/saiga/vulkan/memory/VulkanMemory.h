//
// Created by Peter Eichinger on 10.10.18.
//

#pragma once
#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vulkan/vulkan.hpp>
#include "BufferChunkAllocator.h"
#include "ChunkCreator.h"
#include "ImageChunkAllocator.h"
#include "MemoryType.h"
#include "SimpleMemoryAllocator.h"
#include "saiga/util/assert.h"

namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
class SAIGA_GLOBAL VulkanMemory
{
   private:
    using BufferType       = MemoryType<vk::BufferUsageFlags>;
    using ImageType        = MemoryType<vk::ImageUsageFlags>;
    using BufferMap        = std::unordered_map<BufferType, BufferChunkAllocator>;
    using ImageMap         = std::unordered_map<ImageType, ImageChunkAllocator>;
    using BufferDefaultMap = std::unordered_map<BufferType, vk::DeviceSize>;
    using ImageDefaultMap  = std::unordered_map<ImageType, vk::DeviceSize>;
    using BufferIter       = BufferMap::iterator;
    using ImageIter        = ImageMap::iterator;

    vk::PhysicalDevice m_pDevice;
    vk::Device m_device;
    const vk::BufferUsageFlags all_buffer_usage = static_cast<vk::BufferUsageFlags>(VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM);
    const vk::ImageUsageFlags all_image_usage   = static_cast<vk::ImageUsageFlags>(VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM);
    const vk::MemoryPropertyFlags all_mem_properties =
        static_cast<vk::MemoryPropertyFlagBits>(VK_MEMORY_PROPERTY_FLAG_BITS_MAX_ENUM);



    const vk::DeviceSize fallback_buffer_chunk_size = 64 * 1024 * 1024;
    const vk::DeviceSize fallback_image_chunk_size  = 256 * 1024 * 1024;

    BufferDefaultMap default_buffer_chunk_sizes{
        {{vk::BufferUsageFlagBits::eUniformBuffer,
          vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent},
         1024 * 1024},
        {{all_buffer_usage, all_mem_properties}, fallback_buffer_chunk_size}};

    ImageDefaultMap default_image_chunk_sizes{{{all_image_usage, all_mem_properties}, fallback_image_chunk_size}};


    BufferMap bufferAllocators;
    ImageMap imageAllocators;

   public:
    ChunkCreator chunkAllocator;
    FirstFitStrategy strategy;



    void init(vk::PhysicalDevice _pDevice, vk::Device _device);

    template <typename DefaultSizeMap, typename MemoryType>
    inline typename DefaultSizeMap::const_iterator find_default_size(const DefaultSizeMap& defaultSizes,
                                                                     const MemoryType& type)
    {
        const auto sizes_begin = defaultSizes.cbegin();
        const auto sizes_end   = defaultSizes.cend();
        auto found = std::find_if(sizes_begin, sizes_end, [&](typename DefaultSizeMap::const_reference entry) {
            return (entry.first.valid(type));
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

        return std::find_if(begin, end, [=](typename Map::reference entry) { return entry.first.valid(memoryType); });
    }
    BaseMemoryAllocator& getAllocator(const vk::BufferUsageFlags& usage,
                                      const vk::MemoryPropertyFlags& flags = vk::MemoryPropertyFlagBits::eDeviceLocal);


    BaseMemoryAllocator& getImageAllocator(
        const vk::MemoryPropertyFlags& flags = vk::MemoryPropertyFlagBits::eDeviceLocal,
        const vk::ImageUsageFlags& usage     = vk::ImageUsageFlagBits::eSampled);


    void destroy();
};


}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
