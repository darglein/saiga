//
// Created by Peter Eichinger on 10.10.18.
//

#pragma once
#include "saiga/util/assert.h"

#include "BufferChunkAllocator.h"
#include "ChunkCreator.h"
#include "FallbackAllocator.h"
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
    using BufferMap        = std::map<BufferType, std::unique_ptr<BaseMemoryAllocator>>;
    using ImageMap         = std::map<ImageType, std::unique_ptr<ImageChunkAllocator>>;
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
    inline bool allocator_valid_exact(const MemoryType<T> allocator_type, const MemoryType<T>& type) const
    {
        return ((allocator_type.usageFlags & type.usageFlags) == type.usageFlags) &&
               (allocator_type.memoryFlags == type.memoryFlags);
    }

    template <typename T>
    inline bool allocator_valid_relaxed(const MemoryType<T> allocator_type, const MemoryType<T>& type) const
    {
        return ((allocator_type.usageFlags & type.usageFlags) == type.usageFlags) &&
               ((allocator_type.memoryFlags & type.memoryFlags) == type.memoryFlags);
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
    std::unique_ptr<FallbackAllocator> fallbackAllocator;
    ChunkCreator chunkCreator;
    std::unique_ptr<FitStrategy> strategy;


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

        auto found = std::find_if(
            begin, end, [=](typename Map::reference entry) { return allocator_valid_exact(entry.first, memoryType); });

        if (found == end)
        {
            found = std::find_if(begin, end, [=](typename Map::reference entry) {
                return allocator_valid_relaxed(entry.first, memoryType);
            });
        }

        return found;
    }

    inline vk::MemoryPropertyFlags getEffectiveFlags(const vk::MemoryPropertyFlags& flags) const
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

   public:
    void init(vk::PhysicalDevice _pDevice, vk::Device _device);

    void destroy();

    void renderGUI();

    MemoryLocation allocate(const BufferType& type, vk::DeviceSize size);

    MemoryLocation allocate(const ImageType& type, const vk::Image& image);

    void deallocateBuffer(const BufferType& type, MemoryLocation& location);

    void deallocateImage(const ImageType& type, MemoryLocation& location);

    BaseMemoryAllocator& getAllocator(const BufferType& type);

    ImageChunkAllocator& getImageAllocator(const ImageType& type);
};


}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
