#pragma once

#include "saiga/core/image/image.h"
#include "saiga/vulkan/AsyncCommand.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/buffer/StagingBuffer.h"
#include "saiga/vulkan/memory/MemoryType.h"
#include "saiga/vulkan/svulkan.h"

#include <algorithm>
#include <assert.h>
#include <exception>
namespace Saiga
{
namespace Vulkan
{
struct SAIGA_VULKAN_API Texture
{
   protected:
    VulkanBase* base = nullptr;

   public:

    Memory::ImageMemoryLocation* memoryLocation = nullptr;
    Memory::ImageType type;

   public:
    Texture() = default;

    Texture(const Texture& other) = delete;
    Texture& operator=(const Texture& other) = delete;
    Texture(Texture&& other) noexcept
        : base(other.base),
          memoryLocation(other.memoryLocation),
          type(std::move(other.type)),
          width(other.width),
          height(other.height),
          mipLevels(other.mipLevels),
          layerCount(other.layerCount)
    {
        other.memoryLocation = nullptr;
    }

    Texture& operator=(Texture&& other) noexcept
    {
        if (this != &other)
        {
            base                 = other.base;
            memoryLocation       = std::move(other.memoryLocation);
            type                 = other.type;
            width                = other.width;
            height               = other.height;
            mipLevels            = other.mipLevels;
            layerCount           = other.layerCount;
            other.memoryLocation = nullptr;
        }
        return *this;
    }

    virtual ~Texture() { destroy(); }
    uint32_t width, height;
    uint32_t mipLevels;
    uint32_t layerCount;

    void destroy();

    void transitionImageLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout);

    vk::DescriptorImageInfo getDescriptorInfo();
    inline void mark_dynamic()
    {
        if (memoryLocation)
        {
            memoryLocation->mark_dynamic();
        }
    }
};

struct SAIGA_VULKAN_API Texture2D : public Texture
{
    Texture2D()                       = default;
    Texture2D(const Texture2D& other) = delete;
    Texture2D(Texture2D&& other)      = default;

    Texture2D& operator=(const Texture2D& other) = delete;
    Texture2D& operator=(Texture2D&& other) = default;
    ~Texture2D() override                   = default;
    AsyncCommand fromStagingBuffer(VulkanBase& _base, uint32_t width, uint32_t height, vk::Format format,
                                   Saiga::Vulkan::StagingBuffer& stagingBuffer, Queue& queue, CommandPool& pool,
                                   vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled);
    void fromImage(VulkanBase& _base, Image& img, vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled,
                   bool flipY = true);
    void fromImage(VulkanBase& _base, Image& img, Queue& queue, CommandPool& pool,
                   vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled, bool flipY = true);

    void uploadImage(Image& img, bool flipY);
};

}  // namespace Vulkan
}  // namespace Saiga
