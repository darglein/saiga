/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

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

    Texture(const Texture& other)            = delete;
    Texture& operator=(const Texture& other) = delete;
    Texture(Texture&& other) noexcept
        : base(other.base),
          memoryLocation(other.memoryLocation),
          type(std::move(other.type)),
          width(other.width),
          height(other.height),
          mipLevels(other.mipLevels),
          layerCount(other.layerCount),
          format(other.format),
          movable(other.movable)
    {
        other.memoryLocation = nullptr;
    }

    Texture& operator=(Texture&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
            base                 = other.base;
            memoryLocation       = other.memoryLocation;
            type                 = other.type;
            width                = other.width;
            height               = other.height;
            mipLevels            = other.mipLevels;
            layerCount           = other.layerCount;
            format               = other.format;
            movable              = other.movable;
            other.memoryLocation = nullptr;
        }
        return *this;
    }

    virtual ~Texture() { destroy(); }
    uint32_t width      = 0;
    uint32_t height     = 0;
    uint32_t mipLevels  = 1;
    uint32_t layerCount = 1;
    vk::Format format   = vk::Format::eUndefined;

    /**
     * Releases the image. The memory, image, view and sampler are freed once all frames submitted so far (including
     * the one being recorded) have completed, so the texture can be destroyed while a frame in flight still uses it.
     */
    void destroy();

    void transitionImageLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout);

    vk::DescriptorImageInfo getDescriptorInfo();

    /**
     * Allows the defragmenter to move this texture. Only textures whose format supports storage images can be moved
     * (the defragmenter copies them with a compute shader); for the others this does nothing.
     */
    void mark_dynamic();

    // True if the defragmenter can move the texture, see mark_dynamic().
    bool is_movable() const { return movable; }

   protected:
    // The format supports storage images and the image was created with eStorage (needed by the defragmenter).
    bool movable = false;
};

struct SAIGA_VULKAN_API Texture2D : public Texture
{
    Texture2D()                       = default;
    Texture2D(const Texture2D& other) = delete;
    Texture2D(Texture2D&& other)      = default;

    Texture2D& operator=(const Texture2D& other) = delete;
    Texture2D& operator=(Texture2D&& other)      = default;
    ~Texture2D() override                        = default;

    /**
     * Creates the texture and records + submits the copy from the staging buffer (tightly packed rows of the format).
     * The returned command buffer is allocated from pool: the caller must wait for the fence, then free the command
     * buffer (pool.freeCommandBuffer) and destroy the fence, and must keep the staging buffer alive until then.
     */
    AsyncCommand fromStagingBuffer(VulkanBase& _base, uint32_t width, uint32_t height, vk::Format format,
                                   Saiga::Vulkan::StagingBuffer& stagingBuffer, Queue& queue, CommandPool& pool,
                                   vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled);

    /**
     * Creates the texture from the image and uploads it (blocking).
     * 3-channel images are expanded to 4 channels (alpha = 1), because almost no device supports the 3-channel
     * formats as optimal tiled sampled images.
     */
    void fromImage(VulkanBase& _base, const Image& img, vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled,
                   bool flipY = true);
    void fromImage(VulkanBase& _base, const Image& img, Queue& queue, CommandPool& pool,
                   vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled, bool flipY = true);

    // Uploads new content (blocking). The image must have the size and type the texture was created from.
    void uploadImage(const Image& img, bool flipY);

   protected:
    // Creates the image, view and sampler (in eUndefined layout).
    void create(VulkanBase& _base, uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage);

    // Copies the image into a new staging buffer and records the upload (TransferDst -> ShaderReadOnly).
    void recordUpload(vk::CommandBuffer cmd, const Image& img, bool flipY, StagingBuffer& staging);
};

}  // namespace Vulkan
}  // namespace Saiga
