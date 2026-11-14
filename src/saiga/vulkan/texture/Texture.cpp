/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Texture.h"

#include "saiga/vulkan/buffer/StagingBuffer.h"

#include "vkImageFormat.h"

#include <cstring>

#include <vulkan/vulkan_format_traits.hpp>

namespace Saiga
{
namespace Vulkan
{
namespace
{
// The type an image is uploaded as: 3-channel formats are almost never supported for optimal tiled sampled images,
// so they are expanded to 4 channels.
ImageType uploadType(ImageType type)
{
    SAIGA_ASSERT(type >= 0 && type < TYPE_UNKNOWN, "Unknown image type");
    return channels(type) == 3 ? getType(4, elementType(type)) : type;
}

// The value of a missing alpha channel: 1.0 for normalized and float formats, 1 for integer formats (as Vulkan
// substitutes it for formats without alpha).
void alphaValue(ImageElementType type, uint8_t* dst)
{
    auto write = [dst](auto v) { std::memcpy(dst, &v, sizeof(v)); };
    switch (type)
    {
        case ImageElementType::IET_CHAR:
            write(int8_t(127));
            break;
        case ImageElementType::IET_UCHAR:
            write(uint8_t(255));
            break;
        case ImageElementType::IET_SHORT:
            write(int16_t(32767));
            break;
        case ImageElementType::IET_USHORT:
            write(uint16_t(65535));
            break;
        case ImageElementType::IET_INT:
            write(int32_t(1));
            break;
        case ImageElementType::IET_UINT:
            write(uint32_t(1));
            break;
        case ImageElementType::IET_FLOAT:
            write(1.0f);
            break;
        case ImageElementType::IET_DOUBLE:
            write(1.0);
            break;
        default:
            SAIGA_ASSERT(false, "Unknown image element type");
    }
}

// Tightly packed (and optionally flipped) copy of the image in the upload type.
std::vector<uint8_t> packImage(const Image& img, bool flipY)
{
    const ImageType dstType = uploadType(img.type);
    const size_t srcPixel   = elementSize(img.type);
    const size_t dstPixel   = elementSize(dstType);
    const size_t dstPitch   = size_t(img.width) * dstPixel;

    uint8_t alpha[8];
    if (dstType != img.type) alphaValue(elementType(img.type), alpha);
    const size_t alphaSize = elementSize(elementType(img.type));

    std::vector<uint8_t> data(dstPitch * img.height);
    for (int y = 0; y < img.height; ++y)
    {
        auto src = static_cast<const uint8_t*>(img.rowPtr(flipY ? img.height - 1 - y : y));
        auto dst = data.data() + y * dstPitch;
        if (dstType == img.type)
        {
            std::memcpy(dst, src, dstPitch);
        }
        else
        {
            for (int x = 0; x < img.width; ++x)
            {
                std::memcpy(dst + x * dstPixel, src + x * srcPixel, srcPixel);
                std::memcpy(dst + x * dstPixel + srcPixel, alpha, alphaSize);
            }
        }
    }
    return data;
}
}  // namespace

void Texture::destroy()
{
    if (!memoryLocation) return;

    // The device is gone after VulkanBase::destroy() (the allocator was destroyed with it).
    if (base && base->device)
    {
        // Frames in flight may still use the image, view and sampler.
        auto b        = base;
        auto t        = type;
        auto location = memoryLocation;
        base->deferDelete([b, t, location]() { b->memory.deallocateImage(t, location); });
    }
    memoryLocation = nullptr;
    movable        = false;
}

void Texture::transitionImageLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout)
{
    SAIGA_ASSERT(memoryLocation && memoryLocation->data);

    memoryLocation->data.transitionImageLayout(cmd, newLayout);
}

vk::Image Texture::image() const
{
    SAIGA_ASSERT(memoryLocation);
    return memoryLocation->data.image;
}

vk::ImageView Texture::view() const
{
    SAIGA_ASSERT(memoryLocation);
    return memoryLocation->data.view;
}

uint32_t Texture::fullMipLevels(uint32_t width, uint32_t height, uint32_t depth)
{
    uint32_t size   = std::max({width, height, depth, 1u});
    uint32_t levels = 1;
    while (size > 1)
    {
        size >>= 1;
        levels++;
    }
    return levels;
}

vk::DescriptorImageInfo Texture::getDescriptorInfo(const Sampler& sampler)
{
    SAIGA_ASSERT(sampler, "Texture::getDescriptorInfo: the sampler was not created");
    auto info    = getDescriptorInfo();
    info.sampler = sampler;
    return info;
}

vk::DescriptorImageInfo Texture::getDescriptorInfo()
{
    SAIGA_ASSERT(memoryLocation && memoryLocation->data && memoryLocation->data.sampler);
    Memory::SafeAccessor acc(*memoryLocation);
    return memoryLocation->data.get_descriptor_info();
}

void Texture::mark_dynamic()
{
    if (!memoryLocation) return;
    if (!movable)
    {
        VLOG(1) << "Texture::mark_dynamic: the format " << vk::to_string(format)
                << " does not support storage images, the texture can not be moved by the defragmenter.";
        return;
    }
    memoryLocation->mark_dynamic();
}

void Texture::createImage(VulkanBase& _base, const ImageDescription& d)
{
    destroy();
    base       = &_base;
    width      = d.width;
    height     = d.height;
    depth      = d.depth;
    mipLevels  = d.mipLevels;
    layerCount = d.layerCount;
    format     = d.format;
    samples    = d.samples;

    SAIGA_ASSERT(width > 0 && height > 0 && depth > 0 && layerCount > 0, "Texture: empty image");
    SAIGA_ASSERT(mipLevels >= 1 && mipLevels <= fullMipLevels(width, height, depth),
                 "Texture: invalid mip level count");

    const bool multisampled = samples != vk::SampleCountFlagBits::e1;
    SAIGA_ASSERT(!multisampled || (d.imageType == vk::ImageType::e2D && mipLevels == 1),
                 "Texture: multisampled images are 2D with one mip level");

    auto features = base->physicalDevice.getFormatProperties(format).optimalTilingFeatures;
    auto check    = [&](vk::FormatFeatureFlags required, const char* what)
    {
        SAIGA_ASSERT((features & required) == required, "Texture: the format " + vk::to_string(format) +
                                                            " does not support " + what +
                                                            " with optimal tiling on this device");
    };

    vk::ImageUsageFlags usage = d.usage;
    // Multisampled images can not be copied from/to buffers.
    const bool transferUsage = d.transferUsage && !multisampled;
    if (transferUsage)
    {
        check(vk::FormatFeatureFlagBits::eTransferDst, "transfer dst (upload)");
        usage |= vk::ImageUsageFlagBits::eTransferDst;
        if (features & vk::FormatFeatureFlagBits::eTransferSrc) usage |= vk::ImageUsageFlagBits::eTransferSrc;
    }
    if (usage & vk::ImageUsageFlagBits::eSampled) check(vk::FormatFeatureFlagBits::eSampledImage, "sampled images");
    if (usage & vk::ImageUsageFlagBits::eStorage) check(vk::FormatFeatureFlagBits::eStorageImage, "storage images");
    if (usage & vk::ImageUsageFlagBits::eColorAttachment)
        check(vk::FormatFeatureFlagBits::eColorAttachment, "color attachments");
    if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment)
        check(vk::FormatFeatureFlagBits::eDepthStencilAttachment, "depth stencil attachments");

    // The defragmenter copies a moved image with a compute shader (sampled -> storage), which needs storage support
    // that is optional for many formats (3-channel, 64 bit, most snorm/16 bit). It only handles plain 2D images.
    // Without it the texture stays static.
    movable = d.imageType == vk::ImageType::e2D && mipLevels == 1 && layerCount == 1 && !multisampled &&
              transferUsage && (features & vk::FormatFeatureFlagBits::eStorageImage) &&
              (features & vk::FormatFeatureFlagBits::eSampledImage) && (usage & vk::ImageUsageFlagBits::eSampled);
    if (movable)
    {
        usage |= vk::ImageUsageFlagBits::eStorage;
    }

    type = Memory::ImageType{usage, vk::MemoryPropertyFlagBits::eDeviceLocal};

    vk::ImageCreateInfo imageCreateInfo;
    imageCreateInfo.flags         = d.flags;
    imageCreateInfo.imageType     = d.imageType;
    imageCreateInfo.format        = format;
    imageCreateInfo.mipLevels     = mipLevels;
    imageCreateInfo.arrayLayers   = layerCount;
    imageCreateInfo.samples       = samples;
    imageCreateInfo.tiling        = vk::ImageTiling::eOptimal;
    imageCreateInfo.sharingMode   = vk::SharingMode::eExclusive;
    imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageCreateInfo.extent        = vk::Extent3D{width, height, depth};
    imageCreateInfo.usage         = usage;

    // A view of both aspects of a depth-stencil image can not be sampled, so the view is the depth aspect.
    auto aspect = Memory::ImageData::aspectFlags(format);
    if (aspect & vk::ImageAspectFlagBits::eDepth) aspect = vk::ImageAspectFlagBits::eDepth;

    vk::ImageViewCreateInfo viewCreateInfo = {};
    viewCreateInfo.viewType                = d.viewType;
    viewCreateInfo.format                  = format;
    viewCreateInfo.subresourceRange        = {aspect, 0, mipLevels, 0, layerCount};

    // Linear filtering is optional (e.g. for integer and many depth and float formats).
    bool linear = bool(features & vk::FormatFeatureFlagBits::eSampledImageFilterLinear);

    vk::SamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.magFilter             = linear ? vk::Filter::eLinear : vk::Filter::eNearest;
    samplerCreateInfo.minFilter             = samplerCreateInfo.magFilter;
    samplerCreateInfo.mipmapMode            = linear ? vk::SamplerMipmapMode::eLinear : vk::SamplerMipmapMode::eNearest;
    samplerCreateInfo.addressModeU          = d.addressMode;
    samplerCreateInfo.addressModeV          = d.addressMode;
    samplerCreateInfo.addressModeW          = d.addressMode;
    samplerCreateInfo.mipLodBias            = 0.0f;
    samplerCreateInfo.compareOp             = vk::CompareOp::eNever;
    samplerCreateInfo.minLod                = 0.0f;
    samplerCreateInfo.maxLod                = float(mipLevels - 1);
    samplerCreateInfo.maxAnisotropy         = 1.0f;
    samplerCreateInfo.anisotropyEnable      = VK_FALSE;
    samplerCreateInfo.borderColor           = vk::BorderColor::eIntOpaqueWhite;

    Memory::ImageData img_data(imageCreateInfo, viewCreateInfo, vk::ImageLayout::eUndefined, samplerCreateInfo);
    memoryLocation = base->memory.allocate(type, img_data);
    SAIGA_ASSERT(memoryLocation && memoryLocation->data, "Texture: image allocation failed");
}

void Texture::recordCopy(vk::CommandBuffer cmd, const void* data, size_t size, StagingBuffer& staging, uint32_t layer,
                         vk::Offset3D offset, vk::Extent3D extent)
{
    SAIGA_ASSERT(memoryLocation && memoryLocation->data.layout == vk::ImageLayout::eTransferDstOptimal);
    SAIGA_ASSERT(layer < layerCount);
    SAIGA_ASSERT(offset.x >= 0 && offset.y >= 0 && offset.z >= 0 && offset.x + extent.width <= width &&
                     offset.y + extent.height <= height && offset.z + extent.depth <= depth,
                 "Texture: the upload region is outside of the image");
    size_t texel = vk::blockSize(format);
    SAIGA_ASSERT(size == size_t(extent.width) * extent.height * extent.depth * texel,
                 "Texture: the data size does not match the region");

    staging.init(*base, size, data);

    vk::BufferImageCopy region = staging.getBufferImageCopy(0);
    // vkCmdCopyBufferToImage: the buffer offset must be a multiple of the texel size.
    SAIGA_ASSERT(region.bufferOffset % texel == 0, "Texture: misaligned staging buffer");
    region.imageSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, layer, 1);
    // The staging data is tightly packed.
    region.bufferRowLength   = 0;
    region.bufferImageHeight = 0;
    region.imageOffset       = offset;
    region.imageExtent       = extent;
    staging.copyTo(cmd, memoryLocation->data.image, vk::ImageLayout::eTransferDstOptimal, region);
}

void Texture::recordGenerateMipmaps(vk::CommandBuffer cmd, vk::ImageLayout newLayout)
{
    using Stage  = vk::PipelineStageFlagBits2;
    using Access = vk::AccessFlagBits2;
    SAIGA_ASSERT(memoryLocation && memoryLocation->data);
    SAIGA_ASSERT(samples == vk::SampleCountFlagBits::e1);

    if (mipLevels > 1)
    {
        auto features = base->physicalDevice.getFormatProperties(format).optimalTilingFeatures;
        auto required = vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst |
                        vk::FormatFeatureFlagBits::eSampledImageFilterLinear;
        SAIGA_ASSERT((features & required) == required,
                     "Texture: the format " + vk::to_string(format) + " does not support linear blits (mipmaps)");
        SAIGA_ASSERT((type.usageFlags & vk::ImageUsageFlagBits::eTransferSrc) &&
                         (type.usageFlags & vk::ImageUsageFlagBits::eTransferDst),
                     "Texture: mipmaps need the transfer src and dst usage");

        // Keeps the content of level 0.
        SAIGA_ASSERT(memoryLocation->data.layout != vk::ImageLayout::eUndefined, "Texture: no content for the mipmaps");
        transitionImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);

        vk::Image img = memoryLocation->data.image;
        // Level by level: the source level becomes TransferSrc, at the end the whole image is TransferSrc.
        auto toSrc = [&](uint32_t level)
        {
            vk::ImageMemoryBarrier2 barrier(
                Stage::eAllTransfer, Access::eTransferWrite, Stage::eAllTransfer, Access::eTransferRead,
                vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal, VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED, img,
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, level, 1, 0, layerCount));
            cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, barrier));
        };

        for (uint32_t level = 1; level < mipLevels; ++level)
        {
            toSrc(level - 1);

            vk::ImageBlit blit;
            blit.srcSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, level - 1, 0, layerCount);
            blit.srcOffsets[1]  = vk::Offset3D(mipWidth(level - 1), mipHeight(level - 1), mipDepth(level - 1));
            blit.dstSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, level, 0, layerCount);
            blit.dstOffsets[1]  = vk::Offset3D(mipWidth(level), mipHeight(level), mipDepth(level));
            cmd.blitImage(img, vk::ImageLayout::eTransferSrcOptimal, img, vk::ImageLayout::eTransferDstOptimal, blit,
                          vk::Filter::eLinear);
        }
        toSrc(mipLevels - 1);
        memoryLocation->data.layout = vk::ImageLayout::eTransferSrcOptimal;
    }
    transitionImageLayout(cmd, newLayout);
}

void Texture::generateMipmaps()
{
    SAIGA_ASSERT(base && memoryLocation);
    auto layout = memoryLocation->data.layout;
    if (layout == vk::ImageLayout::eTransferDstOptimal || layout == vk::ImageLayout::eTransferSrcOptimal)
    {
        layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    vk::CommandBuffer cmd = base->mainQueue.commandPool.createAndBeginOneTimeBuffer();
    recordGenerateMipmaps(cmd, layout);
    cmd.end();
    base->mainQueue.submitAndWait(cmd);
    base->mainQueue.commandPool.freeCommandBuffer(cmd);
}

size_t Texture::mipByteSize(uint32_t mipLevel) const
{
    size_t texel = vk::blockSize(format);
    // The depth aspect of the combined formats is copied as D16 / X8D24 / D32.
    if (format == vk::Format::eD16UnormS8Uint) texel = 2;
    if (format == vk::Format::eD24UnormS8Uint || format == vk::Format::eD32SfloatS8Uint) texel = 4;
    return size_t(mipWidth(mipLevel)) * mipHeight(mipLevel) * mipDepth(mipLevel) * texel;
}

std::vector<uint8_t> Texture::download(uint32_t mipLevel, uint32_t layer)
{
    SAIGA_ASSERT(base && memoryLocation && memoryLocation->data);
    SAIGA_ASSERT(mipLevel < mipLevels && layer < layerCount);
    SAIGA_ASSERT(samples == vk::SampleCountFlagBits::e1, "Texture::download: resolve multisampled images first");
    SAIGA_ASSERT(type.usageFlags & vk::ImageUsageFlagBits::eTransferSrc,
                 "Texture::download: the image was created without the transfer src usage");
    auto layout = memoryLocation->data.layout;
    SAIGA_ASSERT(layout != vk::ImageLayout::eUndefined, "Texture::download: the image has no content");

    auto aspect = Memory::ImageData::aspectFlags(format);
    if (aspect & vk::ImageAspectFlagBits::eDepth) aspect = vk::ImageAspectFlagBits::eDepth;

    size_t size = mipByteSize(mipLevel);
    StagingBuffer staging;
    staging.init(*base, size);

    vk::CommandBuffer cmd = base->mainQueue.commandPool.createAndBeginOneTimeBuffer();
    transitionImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);

    vk::BufferImageCopy region = staging.getBufferImageCopy(0);
    region.imageSubresource    = vk::ImageSubresourceLayers(aspect, mipLevel, layer, 1);
    region.imageExtent         = vk::Extent3D(mipWidth(mipLevel), mipHeight(mipLevel), mipDepth(mipLevel));
    cmd.copyImageToBuffer(memoryLocation->data.image, vk::ImageLayout::eTransferSrcOptimal, staging.createInfo().buffer,
                          region);

    // The host reads the staging buffer after the wait.
    staging.barrier(cmd, vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferWrite,
                    vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
    transitionImageLayout(cmd, layout);
    cmd.end();
    base->mainQueue.submitAndWait(cmd);
    base->mainQueue.commandPool.freeCommandBuffer(cmd);

    std::vector<uint8_t> result(size);
    staging.download(result.data(), size);
    staging.destroyImmediately();
    return result;
}

void Texture::blockingUpload(const std::function<void(vk::CommandBuffer)>& func, bool mipmaps)
{
    SAIGA_ASSERT(base && memoryLocation, "Texture: the texture was not created");
    vk::CommandBuffer cmd = base->mainQueue.commandPool.createAndBeginOneTimeBuffer();
    // The transition waits for all previous uses of the image on the queue. It keeps the content, which matters for
    // partial uploads.
    transitionImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
    func(cmd);
    if (mipmaps && mipLevels > 1)
        recordGenerateMipmaps(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    else
        transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    cmd.end();
    base->mainQueue.submitAndWait(cmd);
    base->mainQueue.commandPool.freeCommandBuffer(cmd);
}

void Texture::uploadLayers(const std::vector<const Image*>& images, uint32_t firstLayer, bool flipY, bool mipmaps)
{
    SAIGA_ASSERT(firstLayer + images.size() <= layerCount, "Texture: too many layers");
    // One staging buffer per layer, alive until the copies have completed.
    std::vector<StagingBuffer> staging(images.size());
    blockingUpload(
        [&](vk::CommandBuffer cmd)
        {
            for (size_t i = 0; i < images.size(); ++i)
            {
                const Image& img = *images[i];
                SAIGA_ASSERT(img.width == (int)width && img.height == (int)height,
                             "Texture: the image size must match the texture");
                SAIGA_ASSERT(getvkFormat(uploadType(img.type)) == format,
                             "Texture: the image type must match the texture");
                auto data = packImage(img, flipY);
                recordCopy(cmd, data.data(), data.size(), staging[i], firstLayer + uint32_t(i), vk::Offset3D(0, 0, 0),
                           vk::Extent3D(width, height, 1));
            }
        },
        mipmaps);
    // The copies have completed, no frame uses the staging buffers.
    for (auto& s : staging) s.destroyImmediately();
}

void Texture2D::create(VulkanBase& _base, uint32_t _width, uint32_t _height, vk::Format _format,
                       vk::ImageUsageFlags usage, uint32_t _mipLevels, vk::SampleCountFlagBits _samples)
{
    ImageDescription d;
    d.width     = _width;
    d.height    = _height;
    d.format    = _format;
    d.usage     = usage;
    d.mipLevels = _mipLevels;
    d.samples   = _samples;
    createImage(_base, d);
}

void Texture2D::recordUpload(vk::CommandBuffer cmd, const Image& img, bool flipY, StagingBuffer& staging)
{
    SAIGA_ASSERT(img.width == (int)width && img.height == (int)height, "Texture2D: the size must match the texture");
    SAIGA_ASSERT(getvkFormat(uploadType(img.type)) == format, "Texture2D: the image type must match the texture");

    auto data = packImage(img, flipY);

    transitionImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
    recordCopy(cmd, data.data(), data.size(), staging, 0, vk::Offset3D(0, 0, 0), vk::Extent3D(width, height, 1));
    // Generates the mip levels if there are any.
    recordGenerateMipmaps(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Texture2D::fromImage(VulkanBase& _base, const Image& img, vk::ImageUsageFlags usage, bool flipY, bool mipmaps)
{
    fromImage(_base, img, _base.mainQueue, _base.mainQueue.commandPool, usage, flipY, mipmaps);
}

void Texture2D::fromImage(VulkanBase& _base, const Image& img, Queue& queue, CommandPool& pool,
                          vk::ImageUsageFlags usage, bool flipY, bool mipmaps)
{
    create(_base, img.width, img.height, getvkFormat(uploadType(img.type)), usage,
           mipmaps ? fullMipLevels(img.width, img.height) : 1);

    StagingBuffer staging;
    vk::CommandBuffer cmd = pool.createAndBeginOneTimeBuffer();
    recordUpload(cmd, img, flipY, staging);
    cmd.end();
    queue.submitAndWait(cmd);
    pool.freeCommandBuffer(cmd);
    // The copy has completed, no frame uses the staging buffer.
    staging.destroyImmediately();
}

void Texture2D::uploadImage(const Image& img, bool flipY)
{
    SAIGA_ASSERT(base && memoryLocation, "Texture2D::uploadImage: the texture was not created");
    StagingBuffer staging;
    vk::CommandBuffer cmd = base->mainQueue.commandPool.createAndBeginOneTimeBuffer();
    // The transition to TransferDst waits for all previous uses of the image on the queue.
    recordUpload(cmd, img, flipY, staging);
    cmd.end();
    base->mainQueue.submitAndWait(cmd);
    base->mainQueue.commandPool.freeCommandBuffer(cmd);
    staging.destroyImmediately();
}

AsyncCommand Texture2D::fromStagingBuffer(VulkanBase& _base, uint32_t _width, uint32_t _height, vk::Format _format,
                                          Saiga::Vulkan::StagingBuffer& stagingBuffer, Queue& queue, CommandPool& pool,
                                          vk::ImageUsageFlags usage)
{
    create(_base, _width, _height, _format, usage);

    vk::CommandBuffer cmd = pool.createAndBeginOneTimeBuffer();

    transitionImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);

    vk::BufferImageCopy bufferCopyRegion             = stagingBuffer.getBufferImageCopy(0);
    bufferCopyRegion.imageSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
    bufferCopyRegion.imageSubresource.mipLevel       = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount     = 1;
    bufferCopyRegion.imageExtent                     = vk::Extent3D{width, height, 1U};

    stagingBuffer.copyTo(cmd, memoryLocation->data.image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegion);

    transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

    cmd.end();

    auto fence = queue.submit(cmd);

    return AsyncCommand{cmd, fence};
}

void Texture2DArray::create(VulkanBase& _base, uint32_t _width, uint32_t _height, uint32_t layers, vk::Format _format,
                            vk::ImageUsageFlags usage, uint32_t _mipLevels)
{
    ImageDescription d;
    d.viewType   = vk::ImageViewType::e2DArray;
    d.width      = _width;
    d.height     = _height;
    d.layerCount = layers;
    d.format     = _format;
    d.usage      = usage;
    d.mipLevels  = _mipLevels;
    createImage(_base, d);
}

void Texture2DArray::fromImages(VulkanBase& _base, const std::vector<const Image*>& images, vk::ImageUsageFlags usage,
                                bool flipY, bool mipmaps)
{
    SAIGA_ASSERT(!images.empty() && images.front(), "Texture2DArray: no images");
    const Image& first = *images.front();
    create(_base, first.width, first.height, uint32_t(images.size()), getvkFormat(uploadType(first.type)), usage,
           mipmaps ? fullMipLevels(first.width, first.height) : 1);
    uploadLayers(images, 0, flipY, true);
}

void Texture2DArray::uploadLayer(uint32_t layer, const Image& img, bool flipY)
{
    uploadLayers({&img}, layer, flipY, true);
}

void TextureCube::create(VulkanBase& _base, uint32_t size, vk::Format _format, vk::ImageUsageFlags usage,
                         uint32_t _mipLevels)
{
    ImageDescription d;
    d.viewType    = vk::ImageViewType::eCube;
    d.flags       = vk::ImageCreateFlagBits::eCubeCompatible;
    d.width       = size;
    d.height      = size;
    d.layerCount  = 6;
    d.format      = _format;
    d.usage       = usage;
    d.mipLevels   = _mipLevels;
    d.addressMode = vk::SamplerAddressMode::eClampToEdge;
    createImage(_base, d);
}

void TextureCube::fromImages(VulkanBase& _base, const std::vector<const Image*>& faces, vk::ImageUsageFlags usage,
                             bool flipY, bool mipmaps)
{
    SAIGA_ASSERT(faces.size() == 6 && faces.front(), "TextureCube: 6 faces are required");
    const Image& first = *faces.front();
    SAIGA_ASSERT(first.width == first.height, "TextureCube: the faces must be square");
    create(_base, first.width, getvkFormat(uploadType(first.type)), usage,
           mipmaps ? fullMipLevels(first.width, first.height) : 1);
    uploadLayers(faces, 0, flipY, true);
}

void TextureCube::uploadFace(uint32_t face, const Image& img, bool flipY)
{
    uploadLayers({&img}, face, flipY, true);
}

void Texture3D::create(VulkanBase& _base, uint32_t _width, uint32_t _height, uint32_t _depth, vk::Format _format,
                       vk::ImageUsageFlags usage, uint32_t _mipLevels)
{
    ImageDescription d;
    d.imageType = vk::ImageType::e3D;
    d.viewType  = vk::ImageViewType::e3D;
    d.width     = _width;
    d.height    = _height;
    d.depth     = _depth;
    d.format    = _format;
    d.usage     = usage;
    d.mipLevels = _mipLevels;
    createImage(_base, d);
}

void Texture3D::fromData(VulkanBase& _base, uint32_t _width, uint32_t _height, uint32_t _depth, vk::Format _format,
                         const void* data, vk::ImageUsageFlags usage, bool mipmaps)
{
    create(_base, _width, _height, _depth, _format, usage, mipmaps ? fullMipLevels(_width, _height, _depth) : 1);
    upload(data);
}

void Texture3D::upload(const void* data)
{
    uploadRegion(ivec3(0, 0, 0), ivec3(width, height, depth), data);
}

void Texture3D::uploadRegion(ivec3 offset, ivec3 size, const void* data)
{
    SAIGA_ASSERT(data && size.x() > 0 && size.y() > 0 && size.z() > 0);
    size_t bytes = size_t(size.x()) * size.y() * size.z() * vk::blockSize(format);
    StagingBuffer staging;
    blockingUpload(
        [&](vk::CommandBuffer cmd)
        {
            recordCopy(cmd, data, bytes, staging, 0, vk::Offset3D(offset.x(), offset.y(), offset.z()),
                       vk::Extent3D(size.x(), size.y(), size.z()));
        },
        true);
    staging.destroyImmediately();
}

}  // namespace Vulkan
}  // namespace Saiga
