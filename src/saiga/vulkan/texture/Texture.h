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
#include "saiga/vulkan/texture/Sampler.h"

#include <algorithm>
#include <assert.h>
#include <exception>
#include <functional>
#include <vector>
namespace Saiga
{
namespace Vulkan
{
/**
 * Base class of the textures: an image with one view of all mip levels and layers and a default sampler.
 * The layout of the whole image is tracked (memoryLocation->data.layout), change it with transitionImageLayout.
 *
 * Types: Texture2D, Texture2DArray, TextureCube, Texture3D.
 */
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
          depth(other.depth),
          mipLevels(other.mipLevels),
          layerCount(other.layerCount),
          format(other.format),
          samples(other.samples),
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
            depth                = other.depth;
            mipLevels            = other.mipLevels;
            layerCount           = other.layerCount;
            format               = other.format;
            samples              = other.samples;
            movable              = other.movable;
            other.memoryLocation = nullptr;
        }
        return *this;
    }

    virtual ~Texture() { destroy(); }
    uint32_t width                  = 0;
    uint32_t height                 = 0;
    uint32_t depth                  = 1;
    uint32_t mipLevels              = 1;
    uint32_t layerCount             = 1;
    vk::Format format               = vk::Format::eUndefined;
    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;

    vk::Image image() const;
    vk::ImageView view() const;

    // The size of a mip level
    uint32_t mipWidth(uint32_t level) const { return std::max(1u, width >> level); }
    uint32_t mipHeight(uint32_t level) const { return std::max(1u, height >> level); }
    uint32_t mipDepth(uint32_t level) const { return std::max(1u, depth >> level); }

    // The number of mip levels of a full chain (down to 1x1x1)
    static uint32_t fullMipLevels(uint32_t width, uint32_t height, uint32_t depth = 1);

    /**
     * Releases the image. The memory, image, view and sampler are freed once all frames submitted so far (including
     * the one being recorded) have completed, so the texture can be destroyed while a frame in flight still uses it.
     */
    void destroy();

    // Records a layout transition and tracks the new layout (see setImageLayout()).
    void transitionImageLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout);

    /**
     * The texture tracks its image layout (Memory::ImageData::layout). getDescriptorInfo() uses it, and the
     * defragmentation copies a moved texture from it and restores it. Vulkan can not query the layout of an image, so a
     * transition recorded with an own barrier (instead of transitionImageLayout()) must be reported with this function.
     * The tracked layout must be the layout after all submitted command buffers.
     */
    void setImageLayout(vk::ImageLayout layout);
    vk::ImageLayout getImageLayout() const;

    // With the default sampler of the texture
    vk::DescriptorImageInfo getDescriptorInfo();
    vk::DescriptorImageInfo getDescriptorInfo(const Sampler& sampler);

    /**
     * Records the generation of the mip levels 1.. from level 0 with linear blits (all layers). The image must have
     * content in level 0 and is in newLayout afterwards. The format must support linear blits (asserted).
     */
    void recordGenerateMipmaps(vk::CommandBuffer cmd,
                               vk::ImageLayout newLayout = vk::ImageLayout::eShaderReadOnlyOptimal);
    // Blocking version on the main queue. The layout is kept (shader read only if it was a transfer layout).
    void generateMipmaps();

    /**
     * Reads back one mip level of one layer (blocking, on the main queue): tightly packed texels of the format, for 3D
     * textures all depth slices. The layout is kept. Depth-stencil formats return the depth aspect.
     */
    std::vector<uint8_t> download(uint32_t mipLevel = 0, uint32_t layer = 0);
    // The number of bytes download() returns
    size_t mipByteSize(uint32_t mipLevel) const;

    /**
     * Allows the defragmentation to move this texture (see Memory::VulkanMemory::mark_dynamic): it is copied into a
     * defragmentable pool (blocking). A moved texture gets new image, view and sampler handles: use it through a
     * DynamicDescriptorSet, which is updated after a move. Returns false if the texture stays static (the format does
     * not support transfer src).
     */
    bool mark_dynamic();

    // True if the defragmentation can move the texture, see mark_dynamic().
    bool is_movable() const { return movable; }

   protected:
    // The image was created with eTransferSrc and eTransferDst (needed by the defragmentation).
    bool movable = false;

    struct ImageDescription
    {
        vk::ImageType imageType    = vk::ImageType::e2D;
        vk::ImageViewType viewType = vk::ImageViewType::e2D;
        vk::ImageCreateFlags flags = {};
        uint32_t width = 1, height = 1, depth = 1;
        uint32_t mipLevels = 1, layerCount = 1;
        vk::Format format               = vk::Format::eUndefined;
        vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
        vk::ImageUsageFlags usage;
        // Adds eTransferDst and (if supported) eTransferSrc to the usage, for uploads, downloads and mipmaps.
        bool transferUsage                 = true;
        vk::SamplerAddressMode addressMode = vk::SamplerAddressMode::eRepeat;
    };
    // Creates the image, view and default sampler (in eUndefined layout).
    void createImage(VulkanBase& _base, const ImageDescription& description);

    /**
     * Records the copy of tightly packed texels into a region of mip level 0 of one layer. The image must be in
     * eTransferDstOptimal. The staging buffer is created here and must be kept until the commands have completed.
     */
    void recordCopy(vk::CommandBuffer cmd, const void* data, size_t size, StagingBuffer& staging, uint32_t layer,
                    vk::Offset3D offset, vk::Extent3D extent);

    // Blocking upload of the images into the layers [firstLayer, firstLayer + images.size()) of mip level 0.
    void uploadLayers(const std::vector<const Image*>& images, uint32_t firstLayer, bool flipY, bool mipmaps);
    // Records func on the main queue between the transition to eTransferDstOptimal and the transition back (mipmaps:
    // their generation), submits and waits.
    void blockingUpload(const std::function<void(vk::CommandBuffer)>& func, bool mipmaps);
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
     *
     * mipmaps: creates the full mip chain and generates it from the image (see recordGenerateMipmaps).
     */
    void fromImage(VulkanBase& _base, const Image& img, vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled,
                   bool flipY = true, bool mipmaps = false);
    void fromImage(VulkanBase& _base, const Image& img, Queue& queue, CommandPool& pool,
                   vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled, bool flipY = true,
                   bool mipmaps = false);

    // Uploads new content (blocking) and regenerates the mipmaps if the texture has more than one level. The image
    // must have the size and type the texture was created from.
    void uploadImage(const Image& img, bool flipY);

    /**
     * Creates the image, view and sampler without content (in eUndefined layout), e.g. for a render target.
     * eTransferDst is always added to the usage, eTransferSrc if the format supports it.
     *
     * samples > 1 creates a multisampled image with exactly the given usage (no transfers, one mip level).
     */
    void create(VulkanBase& _base, uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage,
                uint32_t mipLevels = 1, vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1);

   protected:
    // Copies the image into a new staging buffer and records the upload (TransferDst -> ShaderReadOnly, with the
    // mipmap generation in between).
    void recordUpload(vk::CommandBuffer cmd, const Image& img, bool flipY, StagingBuffer& staging);
};

/**
 * An array of 2D images of the same size and format (sampler2DArray).
 */
struct SAIGA_VULKAN_API Texture2DArray : public Texture
{
    Texture2DArray()                            = default;
    Texture2DArray(Texture2DArray&&)            = default;
    Texture2DArray& operator=(Texture2DArray&&) = default;
    ~Texture2DArray() override                  = default;

    // Without content (in eUndefined layout).
    void create(VulkanBase& _base, uint32_t width, uint32_t height, uint32_t layers, vk::Format format,
                vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled, uint32_t mipLevels = 1);

    // One layer per image (blocking). All images must have the same size and type, see Texture2D::fromImage.
    void fromImages(VulkanBase& _base, const std::vector<const Image*>& images,
                    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled, bool flipY = true,
                    bool mipmaps = false);

    // Replaces one layer (blocking) and regenerates the mipmaps of all layers.
    void uploadLayer(uint32_t layer, const Image& img, bool flipY);
};

/**
 * A cube map (samplerCube). The faces are the layers +X, -X, +Y, -Y, +Z, -Z. The default sampler clamps to the edge.
 */
struct SAIGA_VULKAN_API TextureCube : public Texture
{
    TextureCube()                         = default;
    TextureCube(TextureCube&&)            = default;
    TextureCube& operator=(TextureCube&&) = default;
    ~TextureCube() override               = default;

    // Without content (in eUndefined layout).
    void create(VulkanBase& _base, uint32_t size, vk::Format format,
                vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled, uint32_t mipLevels = 1);

    // The 6 square faces of the same size and type (blocking), see Texture2D::fromImage.
    void fromImages(VulkanBase& _base, const std::vector<const Image*>& faces,
                    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled, bool flipY = false,
                    bool mipmaps = false);

    // Replaces one face (blocking) and regenerates the mipmaps of all faces.
    void uploadFace(uint32_t face, const Image& img, bool flipY);
};

/**
 * A volume (sampler3D). The data is tightly packed: x is the fastest index, then y, then z.
 */
struct SAIGA_VULKAN_API Texture3D : public Texture
{
    Texture3D()                       = default;
    Texture3D(Texture3D&&)            = default;
    Texture3D& operator=(Texture3D&&) = default;
    ~Texture3D() override             = default;

    // Without content (in eUndefined layout).
    void create(VulkanBase& _base, uint32_t width, uint32_t height, uint32_t depth, vk::Format format,
                vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled, uint32_t mipLevels = 1);

    // Creates the texture and uploads width * height * depth texels of the format (blocking).
    void fromData(VulkanBase& _base, uint32_t width, uint32_t height, uint32_t depth, vk::Format format,
                  const void* data, vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled, bool mipmaps = false);

    // Replaces the whole volume (blocking) and regenerates the mipmaps.
    void upload(const void* data);

    // Replaces the texels of a box in mip level 0 (blocking) and regenerates the mipmaps of the whole volume.
    void uploadRegion(ivec3 offset, ivec3 size, const void* data);
};

}  // namespace Vulkan
}  // namespace Saiga
