#include "Texture.h"
#include "saiga/vulkan/buffer/StagingBuffer.h"

#include "vkImageFormat.h"

namespace Saiga
{
namespace Vulkan
{
static const vk::ImageUsageFlags necessary_flags =
    vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eStorage;

void Texture::destroy()
{
    if (!base)
    {
        return;
    }


    if (memoryLocation)
    {
        base->memory.deallocateImage(type, memoryLocation);
        memoryLocation = nullptr;
    }
}

void Texture::transitionImageLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout)
{
    SAIGA_ASSERT(memoryLocation->data);

    memoryLocation->data.transitionImageLayout(cmd, newLayout);
}

vk::DescriptorImageInfo Texture::getDescriptorInfo()
{
    SAIGA_ASSERT(memoryLocation->data && memoryLocation->data.sampler);
    Memory::SafeAccessor acc(*memoryLocation);
    return memoryLocation->data.get_descriptor_info();
}

void Texture2D::fromImage(VulkanBase& base, Image& img, vk::ImageUsageFlags usage, bool flipY)
{
    fromImage(base, img, base.mainQueue, base.mainQueue.commandPool, usage, flipY);
}

void Texture2D::uploadImage(Image& img, bool flipY)
{
    vk::CommandBuffer cmd = base->mainQueue.commandPool.createAndBeginOneTimeBuffer();

    transitionImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);

    StagingBuffer staging;

    if (flipY)
    {
        std::vector<char> data(img.pitchBytes * img.h);
        for (int i = 0; i < img.h; ++i)
        {
            memcpy(&data[i * img.pitchBytes], img.rowPtr(img.h - i - 1), img.pitchBytes);
        }

        staging.init(*base, data.size(), data.data());
    }
    else
    {
        staging.init(*base, img.size(), img.data());
    }



    vk::BufferImageCopy bufferCopyRegion             = staging.getBufferImageCopy(0);
    bufferCopyRegion.imageSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
    bufferCopyRegion.imageSubresource.mipLevel       = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount     = 1;
    bufferCopyRegion.imageExtent.width               = width;
    bufferCopyRegion.imageExtent.height              = height;
    bufferCopyRegion.imageExtent.depth               = 1;

    staging.copyTo(cmd, memoryLocation->data.image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegion);

    transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

    cmd.end();
    base->mainQueue.submitAndWait(cmd);

    staging.destroy();
}

void Texture2D::fromImage(VulkanBase& _base, Image& img, Queue& queue, CommandPool& pool, vk::ImageUsageFlags usage,
                          bool flipY)
{
    destroy();

    this->base = &_base;

    mipLevels = 1;
    width     = img.width;
    height    = img.height;

    vk::Format format = getvkFormat(img.type);



    auto finalUsageFlags = usage | necessary_flags;

    type = Memory::ImageType{finalUsageFlags, vk::MemoryPropertyFlagBits::eDeviceLocal};

    // Create optimal tiled target image
    vk::ImageCreateInfo imageCreateInfo;
    imageCreateInfo.imageType     = vk::ImageType::e2D;
    imageCreateInfo.format        = format;
    imageCreateInfo.mipLevels     = mipLevels;
    imageCreateInfo.arrayLayers   = 1;
    imageCreateInfo.samples       = vk::SampleCountFlagBits::e1;
    imageCreateInfo.tiling        = vk::ImageTiling::eOptimal;
    imageCreateInfo.sharingMode   = vk::SharingMode::eExclusive;
    imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageCreateInfo.extent        = vk::Extent3D{width, height, 1U};
    imageCreateInfo.usage         = finalUsageFlags;


    vk::ImageViewCreateInfo viewCreateInfo = {};
    viewCreateInfo.viewType                = vk::ImageViewType::e2D;
    viewCreateInfo.format                  = format;
    viewCreateInfo.subresourceRange        = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

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
    // Max level-of-detail should match mip level count
    samplerCreateInfo.maxLod = 0.0f;
    // Only enable anisotropic filtering if enabled on the devicec
    samplerCreateInfo.maxAnisotropy    = 16;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.borderColor      = vk::BorderColor::eIntOpaqueWhite;

    Memory::ImageData img_data(imageCreateInfo, viewCreateInfo, vk::ImageLayout::eUndefined, samplerCreateInfo);
    // image                         = base->device.createImage(imageCreateInfo);
    // SAIGA_ASSERT(image);


    // VLOG(3) << "Creating image synched: " << image;

    memoryLocation = base->memory.allocate(type, img_data);

    // memoryLocation->data = std::move(img_data);

    // memoryLocation->data.create(base->device);

    // base->device.bindImageMemory(image, memoryLocation->memory, memoryLocation->offset);

    vk::CommandBuffer cmd = pool.createAndBeginOneTimeBuffer();

    transitionImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);



    StagingBuffer staging;

    if (flipY)
    {
        std::vector<char> data(img.pitchBytes * img.h);
        for (int i = 0; i < img.h; ++i)
        {
            memcpy(&data[i * img.pitchBytes], img.rowPtr(img.h - i - 1), img.pitchBytes);
        }

        staging.init(*base, data.size(), data.data());
    }
    else
    {
        staging.init(*base, img.size(), img.data());
    }

    vk::BufferImageCopy bufferCopyRegion             = staging.getBufferImageCopy(0);
    bufferCopyRegion.imageSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
    bufferCopyRegion.imageSubresource.mipLevel       = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount     = 1;
    bufferCopyRegion.imageExtent.width               = width;
    bufferCopyRegion.imageExtent.height              = height;
    bufferCopyRegion.imageExtent.depth               = 1;

    staging.copyTo(cmd, memoryLocation->data.image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegion);

    transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

    cmd.end();
    queue.submitAndWait(cmd);
    pool.freeCommandBuffer(cmd);


    staging.destroy();


    // vk::ImageViewCreateInfo viewCreateInfo = {};
    // viewCreateInfo.viewType                = vk::ImageViewType::e2D;
    // viewCreateInfo.format                  = format;
    // viewCreateInfo.subresourceRange        = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    // viewCreateInfo.image                   = image;
    // imageView                              = base->device.createImageView(viewCreateInfo);
    // SAIGA_ASSERT(imageView);
}

AsyncCommand Texture2D::fromStagingBuffer(VulkanBase& base, uint32_t width, uint32_t height, vk::Format format,
                                          Saiga::Vulkan::StagingBuffer& stagingBuffer, Queue& queue, CommandPool& pool,
                                          vk::ImageUsageFlags usage)
{
    destroy();
    this->base = &base;

    mipLevels = 1;
    //    width = img.width;
    //    height = img.height;

    //    vk::Format format = getvkFormat(img.type);



    auto finalUsageFlags = usage | necessary_flags;

    type = Memory::ImageType{finalUsageFlags, vk::MemoryPropertyFlagBits::eDeviceLocal};

    // imageLayout = vk::ImageLayout::eUndefined;
    // Create optimal tiled target image
    vk::ImageCreateInfo imageCreateInfo;
    imageCreateInfo.imageType     = vk::ImageType::e2D;
    imageCreateInfo.format        = format;
    imageCreateInfo.mipLevels     = mipLevels;
    imageCreateInfo.arrayLayers   = 1;
    imageCreateInfo.samples       = vk::SampleCountFlagBits::e1;
    imageCreateInfo.tiling        = vk::ImageTiling::eOptimal;
    imageCreateInfo.sharingMode   = vk::SharingMode::eExclusive;
    imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageCreateInfo.extent        = vk::Extent3D{width, height, 1U};
    imageCreateInfo.usage         = finalUsageFlags;
    // image                         = base.device.createImage(imageCreateInfo);



    vk::ImageViewCreateInfo viewCreateInfo = {};
    viewCreateInfo.viewType                = vk::ImageViewType::e2D;
    viewCreateInfo.format                  = format;
    viewCreateInfo.subresourceRange        = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    // imageView                              = base.device.createImageView(viewCreateInfo);


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
    // Max level-of-detail should match mip level count
    samplerCreateInfo.maxLod = 0.0f;
    // Only enable anisotropic filtering if enabled on the devicec
    samplerCreateInfo.maxAnisotropy    = 16;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.borderColor      = vk::BorderColor::eIntOpaqueWhite;
    Memory::ImageData img_data(imageCreateInfo, viewCreateInfo, vk::ImageLayout::eUndefined, samplerCreateInfo);

    memoryLocation = base.memory.allocate(type, img_data);
    // base.device.bindImageMemory(image, memoryLocation->memory, memoryLocation->offset);

    vk::CommandBuffer cmd = pool.createAndBeginOneTimeBuffer();

    transitionImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);


    vk::BufferImageCopy bufferCopyRegion             = stagingBuffer.getBufferImageCopy(0);
    bufferCopyRegion.imageSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
    bufferCopyRegion.imageSubresource.mipLevel       = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount     = 1;
    bufferCopyRegion.imageExtent.width               = width;
    bufferCopyRegion.imageExtent.height              = height;
    bufferCopyRegion.imageExtent.depth               = 1;

    //    StagingBuffer staging;
    //
    //    staging.init(base,img.size(),img.data());

    stagingBuffer.copyTo(cmd, memoryLocation->data.image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegion);

    transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

    cmd.end();

    //    queue.submitAndWait(cmd);
    //    pool.freeCommandBuffer(cmd);



    // vk::ImageViewCreateInfo viewCreateInfo = {};
    // viewCreateInfo.viewType                = vk::ImageViewType::e2D;
    // viewCreateInfo.format                  = format;
    // viewCreateInfo.subresourceRange        = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    // viewCreateInfo.image                   = image;
    // imageView                              = base.device.createImageView(viewCreateInfo);

    // Create a defaultsampler

    auto fence = queue.submit(cmd);
    return AsyncCommand{cmd, fence};
}


}  // namespace Vulkan
}  // namespace Saiga
