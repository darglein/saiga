//
// Created by Peter Eichinger on 10.10.18.
//

#pragma once
#include "saiga/core/util/assert.h"
#include "saiga/vulkan/Parameters.h"
#include "saiga/vulkan/svulkan.h"

#include "BufferChunkAllocator.h"
#include "Defragger.h"
#include "ImageChunkAllocator.h"
#include "ImageCopyComputeShader.h"
#include "MemoryType.h"
#include "UniqueAllocator.h"

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <vulkan/vulkan.hpp>

#include <unordered_map>
namespace Saiga::Vulkan
{
struct VulkanBase;
}
namespace Saiga::Vulkan::Memory
{
static const vk::BufferUsageFlags all_buffer_usage(VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM);
static const vk::ImageUsageFlags all_image_usage(VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM);

static const vk::MemoryPropertyFlags all_mem_properties(VK_MEMORY_PROPERTY_FLAG_BITS_MAX_ENUM);
static const vk::DeviceSize fallback_buffer_chunk_size = 64 * 1024 * 1024;

static const vk::DeviceSize fallback_image_chunk_size = 32 * 1024 * 1024;

class SAIGA_VULKAN_API VulkanMemory
{
   private:
    uint32_t frame_number = 0;

   public:
    bool enableDefragmentation = false;
    bool enableChunkAllocator  = true;

   private:
    template <bool usage_exact, bool memory_exact, typename T>
    struct allocator_find_functor
    {
        T type;
        explicit allocator_find_functor(T _type) : type(_type) {}
        template <typename MapIter>
        inline bool operator()(const MapIter& iter) const
        {
            bool usage_valid;
            if (usage_exact)
            {
                usage_valid = iter.first.usageFlags == type.usageFlags;
            }
            else
            {
                usage_valid = (iter.first.usageFlags & type.usageFlags) == type.usageFlags;
            }

            bool memory_valid;

            if (memory_exact)
            {
                memory_valid = iter.first.memoryFlags == type.memoryFlags;
            }
            else
            {
                memory_valid = ((iter.first.memoryFlags & type.memoryFlags) == type.memoryFlags);
            }

            return usage_valid && memory_valid;
        }
    };

    struct BufferContainer
    {
        std::unique_ptr<BufferChunkAllocator> allocator;
        std::unique_ptr<BufferDefragger> defragger;
    };

    struct ImageContainer
    {
        std::unique_ptr<ImageChunkAllocator> allocator;
        std::unique_ptr<ImageDefragger> defragger;
    };
    using BufferMap  = std::map<BufferType, BufferContainer>;
    using ImageMap   = std::map<ImageType, ImageContainer>;
    using BufferIter = BufferMap::iterator;
    using ImageIter  = ImageMap::iterator;

    using BufferDefaultMap = std::map<BufferType, vk::DeviceSize>;
    using ImageDefaultMap  = std::map<ImageType, vk::DeviceSize>;


    VulkanBase* base{};
    vk::PhysicalDevice m_pDevice;
    vk::Device m_device;
    Queue* m_queue{};
    uint32_t m_swapchain_images{};
    std::unique_ptr<ImageCopyComputeShader> img_copy_shader;


    std::vector<vk::MemoryType> memoryTypes;


    BufferDefaultMap default_buffer_chunk_sizes{
        {BufferType{vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,
                    all_mem_properties},
         1024 * 1024},
        {BufferType{all_buffer_usage, all_mem_properties}, fallback_buffer_chunk_size}};

    ImageDefaultMap default_image_chunk_sizes{
        {ImageType{all_image_usage, all_mem_properties}, fallback_image_chunk_size}};


    BufferMap bufferAllocators;
    ImageMap imageAllocators;

    std::unique_ptr<UniqueAllocator> fallbackAllocator;
    std::unique_ptr<FitStrategy<BufferMemoryLocation>> strategy;
    std::unique_ptr<FitStrategy<ImageMemoryLocation>> image_strategy;


    template <typename DefaultSizeMap, typename MemoryType>
    inline typename DefaultSizeMap::const_iterator find_default_size(const DefaultSizeMap& defaultSizes,
                                                                     const MemoryType& type)
    {
        const auto sizes_begin = defaultSizes.cbegin();
        const auto sizes_end   = defaultSizes.cend();

        auto found = std::find_if(sizes_begin, sizes_end, allocator_find_functor<false, false, MemoryType>(type));

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

        auto found = std::find_if(begin, end, allocator_find_functor<false, true, MemoryType<UsageType>>(memoryType));

        if (found == end)
        {
            found = std::find_if(begin, end, allocator_find_functor<false, false, MemoryType<UsageType>>(memoryType));
        }

        return found;
    }

    template <typename Map, typename UsageType>
    inline typename Map::iterator find_allocator_exact(Map& map, const MemoryType<UsageType>& memoryType)
    {
        const auto begin = map.begin();
        const auto end   = map.end();

        auto found = std::find_if(begin, end, allocator_find_functor<true, true, MemoryType<UsageType>>(memoryType));

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
    BufferContainer& getAllocator(const BufferType& type);
    BufferContainer& get_allocator_exact(const BufferType& type);
    ImageContainer& getImageAllocator(const ImageType& type);
    ImageContainer& get_image_allocator_exact(const ImageType& type);

   private:
   public:
    void init(VulkanBase* base, uint32_t swapchain_frames, const VulkanParameters& parameters);
    VulkanMemory()                    = default;
    VulkanMemory(const VulkanMemory&) = delete;
    VulkanMemory& operator=(const VulkanMemory&) = delete;

    void destroy();

    void renderGUI();

    BufferMemoryLocation* allocate(const BufferType& type, vk::DeviceSize size);

    ImageMemoryLocation* allocate(const ImageType& type, ImageData& image);

    void deallocateBuffer(const BufferType& type, BufferMemoryLocation* location);

    void deallocateImage(const ImageType& type, ImageMemoryLocation* location);

    void enable_defragmentation(const BufferType& type, bool enable);

    void enable_defragmentation(const ImageType& type, bool enable);

    void start_defrag(const BufferType& type);

    void start_defrag(const ImageType& type);

    void stop_defrag(const BufferType& type);

    void stop_defrag(const ImageType& type);

    bool performTimedDefrag(int64_t time, vk::Semaphore semaphore);

    void update()
    {
        frame_number++;
        for (auto& allocator : bufferAllocators)
        {
            auto* defragger = allocator.second.defragger.get();

            if (defragger)
            {
                defragger->update(frame_number);
            }
        }

        for (auto& allocator : imageAllocators)
        {
            auto* defragger = allocator.second.defragger.get();

            if (defragger)
            {
                defragger->update(frame_number);
            }
        }
    }

    void full_defrag();
};

}  // namespace Saiga::Vulkan::Memory
