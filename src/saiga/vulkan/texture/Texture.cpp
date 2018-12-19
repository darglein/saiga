#include "Texture.h"
#include "saiga/vulkan/buffer/StagingBuffer.h"

#include "vkImageFormat.h"

namespace Saiga
{
namespace Vulkan
{
void Texture::destroy()
{
    if (!base)
    {
        return;
    }
    if (image)
    {
        LOG(INFO) << "Destroying image: " << image;
        base->device.destroyImage(image);
        base->device.destroyImageView(imageView);
        base->device.destroySampler(sampler);
        image = nullptr;
    }

    if (memoryLocation)
    {
        base->memory.deallocateImage(type, memoryLocation);
    }
}

void Texture::transitionImageLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout)
{
    vk::ImageMemoryBarrier barrier          = {};
    barrier.oldLayout                       = imageLayout;
    barrier.newLayout                       = newLayout;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                           = image;
    barrier.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;
    //        barrier.srcAccessMask = 0; // TODO
    //        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite; // TODO


    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (imageLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage      = vk::PipelineStageFlagBits::eHost;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (imageLayout == vk::ImageLayout::eTransferDstOptimal &&
             newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage      = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eAllCommands;
    }
    else
    {
        //            throw std::invalid_argument("unsupported layout transition!");
        barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        sourceStage           = vk::PipelineStageFlagBits::eAllCommands;
        destinationStage      = vk::PipelineStageFlagBits::eAllCommands;
    }



    cmd.pipelineBarrier(sourceStage, destinationStage, vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &barrier);


    imageLayout = newLayout;
}

vk::DescriptorImageInfo Texture::getDescriptorInfo()
{
    SAIGA_ASSERT(image);
    vk::DescriptorImageInfo descriptorInfo;
    descriptorInfo.imageLayout = imageLayout;
    descriptorInfo.imageView   = imageView;
    descriptorInfo.sampler     = sampler;
    SAIGA_ASSERT(imageView && sampler);
    return descriptorInfo;
}

void Texture2D::fromImage(VulkanBase& base, Image& img, vk::ImageUsageFlags usage, bool flipY)
{
    fromImage(base, img, base.transferQueue, base.commandPool, usage, flipY);
}

void Texture2D::uploadImage(Image& img, bool flipY)
{
    vk::CommandBuffer cmd = base->createAndBeginTransferCommand();

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

    staging.copyTo(cmd, image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegion);

    transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

    base->endTransferWait(cmd);

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



    auto finalUsageFlags = usage | vk::ImageUsageFlagBits::eTransferDst;

    type = Memory::ImageType{finalUsageFlags, vk::MemoryPropertyFlagBits::eDeviceLocal};

    imageLayout = vk::ImageLayout::eUndefined;
    // Create optimal tiled target image
    vk::ImageCreateInfo imageCreateInfo;
    imageCreateInfo.imageType     = vk::ImageType::e2D;
    imageCreateInfo.format        = format;
    imageCreateInfo.mipLevels     = mipLevels;
    imageCreateInfo.arrayLayers   = 1;
    imageCreateInfo.samples       = vk::SampleCountFlagBits::e1;
    imageCreateInfo.tiling        = vk::ImageTiling::eOptimal;
    imageCreateInfo.sharingMode   = vk::SharingMode::eExclusive;
    imageCreateInfo.initialLayout = imageLayout;
    imageCreateInfo.extent        = vk::Extent3D{width, height, 1U};
    imageCreateInfo.usage         = finalUsageFlags;
    image                         = base->device.createImage(imageCreateInfo);
    SAIGA_ASSERT(image);


    LOG(INFO) << "Creating image synched: " << image;



    memoryLocation = base->memory.allocate(type, image);
    base->device.bindImageMemory(image, memoryLocation.memory, memoryLocation.offset);

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

    staging.copyTo(cmd, image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegion);

    transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

    cmd.end();
    queue.submitAndWait(cmd);
    pool.freeCommandBuffer(cmd);


    staging.destroy();


    vk::ImageViewCreateInfo viewCreateInfo = {};
    viewCreateInfo.viewType                = vk::ImageViewType::e2D;
    viewCreateInfo.format                  = format;
    viewCreateInfo.subresourceRange        = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    viewCreateInfo.image                   = image;
    imageView                              = base->device.createImageView(viewCreateInfo);
    SAIGA_ASSERT(imageView);

    // Create a defaultsampler
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

    sampler = base->device.createSampler(samplerCreateInfo);
    SAIGA_ASSERT(sampler);
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



    auto finalUsageFlags = usage | vk::ImageUsageFlagBits::eTransferDst;

    type = Memory::ImageType{finalUsageFlags, vk::MemoryPropertyFlagBits::eDeviceLocal};

    imageLayout = vk::ImageLayout::eUndefined;
    // Create optimal tiled target image
    vk::ImageCreateInfo imageCreateInfo;
    imageCreateInfo.imageType     = vk::ImageType::e2D;
    imageCreateInfo.format        = format;
    imageCreateInfo.mipLevels     = mipLevels;
    imageCreateInfo.arrayLayers   = 1;
    imageCreateInfo.samples       = vk::SampleCountFlagBits::e1;
    imageCreateInfo.tiling        = vk::ImageTiling::eOptimal;
    imageCreateInfo.sharingMode   = vk::SharingMode::eExclusive;
    imageCreateInfo.initialLayout = imageLayout;
    imageCreateInfo.extent        = vk::Extent3D{width, height, 1U};
    imageCreateInfo.usage         = finalUsageFlags;
    image                         = base.device.createImage(imageCreateInfo);

    LOG(INFO) << "Creating image: " << image << " from " << stagingBuffer;
    SAIGA_ASSERT(image);

    memoryLocation = base.memory.allocate(type, image);
    base.device.bindImageMemory(image, memoryLocation.memory, memoryLocation.offset);

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

    stagingBuffer.copyTo(cmd, image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegion);

    transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);

    cmd.end();

    //    queue.submitAndWait(cmd);
    //    pool.freeCommandBuffer(cmd);



    vk::ImageViewCreateInfo viewCreateInfo = {};
    viewCreateInfo.viewType                = vk::ImageViewType::e2D;
    viewCreateInfo.format                  = format;
    viewCreateInfo.subresourceRange        = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    viewCreateInfo.image                   = image;
    imageView                              = base.device.createImageView(viewCreateInfo);
    SAIGA_ASSERT(imageView);

    // Create a defaultsampler
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

    sampler = base.device.createSampler(samplerCreateInfo);
    SAIGA_ASSERT(sampler);

    auto fence = queue.submit(cmd);
    return AsyncCommand{cmd, fence};
}


}  // namespace Vulkan
}  // namespace Saiga
