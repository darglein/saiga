/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Texture.h"

#include "saiga/vulkan/buffer/StagingBuffer.h"

#include "vkImageFormat.h"

#include <cstring>

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

void Texture2D::create(VulkanBase& _base, uint32_t _width, uint32_t _height, vk::Format _format,
                       vk::ImageUsageFlags usage)
{
    destroy();

    base       = &_base;
    mipLevels  = 1;
    layerCount = 1;
    width      = _width;
    height     = _height;
    format     = _format;

    SAIGA_ASSERT(width > 0 && height > 0, "Texture2D: empty image");

    auto features = base->physicalDevice.getFormatProperties(format).optimalTilingFeatures;
    auto check    = [&](vk::FormatFeatureFlags required, const char* what)
    {
        SAIGA_ASSERT((features & required) == required, "Texture2D: the format " + vk::to_string(format) +
                                                            " does not support " + what +
                                                            " with optimal tiling on this device");
    };
    check(vk::FormatFeatureFlagBits::eTransferDst, "transfer dst (upload)");
    if (usage & vk::ImageUsageFlagBits::eSampled) check(vk::FormatFeatureFlagBits::eSampledImage, "sampled images");
    if (usage & vk::ImageUsageFlagBits::eStorage) check(vk::FormatFeatureFlagBits::eStorageImage, "storage images");
    if (usage & vk::ImageUsageFlagBits::eColorAttachment)
        check(vk::FormatFeatureFlagBits::eColorAttachment, "color attachments");

    vk::ImageUsageFlags finalUsageFlags = usage | vk::ImageUsageFlagBits::eTransferDst;
    if (features & vk::FormatFeatureFlagBits::eTransferSrc)
    {
        finalUsageFlags |= vk::ImageUsageFlagBits::eTransferSrc;
    }
    // The defragmenter copies a moved image with a compute shader (sampled -> storage), which needs storage support
    // that is optional for many formats (3-channel, 64 bit, most snorm/16 bit). Without it the texture stays static.
    movable = (features & vk::FormatFeatureFlagBits::eStorageImage) &&
              (features & vk::FormatFeatureFlagBits::eSampledImage) && (usage & vk::ImageUsageFlagBits::eSampled);
    if (movable)
    {
        finalUsageFlags |= vk::ImageUsageFlagBits::eStorage;
    }

    type = Memory::ImageType{finalUsageFlags, vk::MemoryPropertyFlagBits::eDeviceLocal};

    vk::ImageCreateInfo imageCreateInfo;
    imageCreateInfo.imageType     = vk::ImageType::e2D;
    imageCreateInfo.format        = format;
    imageCreateInfo.mipLevels     = mipLevels;
    imageCreateInfo.arrayLayers   = layerCount;
    imageCreateInfo.samples       = vk::SampleCountFlagBits::e1;
    imageCreateInfo.tiling        = vk::ImageTiling::eOptimal;
    imageCreateInfo.sharingMode   = vk::SharingMode::eExclusive;
    imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageCreateInfo.extent        = vk::Extent3D{width, height, 1U};
    imageCreateInfo.usage         = finalUsageFlags;

    vk::ImageViewCreateInfo viewCreateInfo = {};
    viewCreateInfo.viewType                = vk::ImageViewType::e2D;
    viewCreateInfo.format                  = format;
    viewCreateInfo.subresourceRange        = {vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, layerCount};

    vk::SamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.magFilter             = vk::Filter::eLinear;
    samplerCreateInfo.minFilter             = vk::Filter::eLinear;
    samplerCreateInfo.mipmapMode            = vk::SamplerMipmapMode::eLinear;
    samplerCreateInfo.addressModeU          = vk::SamplerAddressMode::eRepeat;
    samplerCreateInfo.addressModeV          = vk::SamplerAddressMode::eRepeat;
    samplerCreateInfo.addressModeW          = vk::SamplerAddressMode::eRepeat;
    samplerCreateInfo.mipLodBias            = 0.0f;
    samplerCreateInfo.compareOp             = vk::CompareOp::eNever;
    samplerCreateInfo.minLod                = 0.0f;
    samplerCreateInfo.maxLod                = float(mipLevels - 1);
    samplerCreateInfo.maxAnisotropy         = 1.0f;
    samplerCreateInfo.anisotropyEnable      = VK_FALSE;
    samplerCreateInfo.borderColor           = vk::BorderColor::eIntOpaqueWhite;

    Memory::ImageData img_data(imageCreateInfo, viewCreateInfo, vk::ImageLayout::eUndefined, samplerCreateInfo);
    memoryLocation = base->memory.allocate(type, img_data);
    SAIGA_ASSERT(memoryLocation && memoryLocation->data, "Texture2D: image allocation failed");
}

void Texture2D::recordUpload(vk::CommandBuffer cmd, const Image& img, bool flipY, StagingBuffer& staging)
{
    SAIGA_ASSERT(img.width == (int)width && img.height == (int)height, "Texture2D: the size must match the texture");
    SAIGA_ASSERT(getvkFormat(uploadType(img.type)) == format, "Texture2D: the image type must match the texture");

    auto data = packImage(img, flipY);
    staging.init(*base, data.size(), data.data());

    vk::BufferImageCopy bufferCopyRegion = staging.getBufferImageCopy(0);
    // vkCmdCopyBufferToImage: the buffer offset must be a multiple of the texel size.
    SAIGA_ASSERT(bufferCopyRegion.bufferOffset % elementSize(uploadType(img.type)) == 0,
                 "Texture2D: misaligned staging buffer");
    bufferCopyRegion.imageSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
    bufferCopyRegion.imageSubresource.mipLevel       = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount     = 1;
    // The staging data is tightly packed.
    bufferCopyRegion.bufferRowLength   = 0;
    bufferCopyRegion.bufferImageHeight = 0;
    bufferCopyRegion.imageExtent       = vk::Extent3D{width, height, 1U};

    transitionImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
    staging.copyTo(cmd, memoryLocation->data.image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegion);
    transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Texture2D::fromImage(VulkanBase& _base, const Image& img, vk::ImageUsageFlags usage, bool flipY)
{
    fromImage(_base, img, _base.mainQueue, _base.mainQueue.commandPool, usage, flipY);
}

void Texture2D::fromImage(VulkanBase& _base, const Image& img, Queue& queue, CommandPool& pool,
                          vk::ImageUsageFlags usage, bool flipY)
{
    create(_base, img.width, img.height, getvkFormat(uploadType(img.type)), usage);

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


}  // namespace Vulkan
}  // namespace Saiga
