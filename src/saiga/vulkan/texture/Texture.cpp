#include "Texture.h"
#include "vkImageFormat.h"

#include "saiga/vulkan/buffer/StagingBuffer.h"

namespace Saiga{
namespace Vulkan{


//Texture::~Texture()
//{
////    cout << "destroy texture" << endl;
//    destroy();
//}

void Texture::destroy(VulkanBase& base)
{
    if(image)
    {
        base.device.destroyImage(image);
        base.device.destroyImageView(imageView);
        base.device.destroySampler(sampler);
        image = nullptr;
    }

    if (memoryLocation) {
        base.memory.getImageAllocator(vk::MemoryPropertyFlagBits::eDeviceLocal).deallocate(memoryLocation);
    }
}

void Texture::transitionImageLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout)
{
    vk::ImageMemoryBarrier barrier = {};
    barrier.oldLayout = imageLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    //        barrier.srcAccessMask = 0; // TODO
    //        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite; // TODO


    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (imageLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eHost ;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (imageLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eAllCommands;
    } else {
        //            throw std::invalid_argument("unsupported layout transition!");
        barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        sourceStage = vk::PipelineStageFlagBits::eAllCommands;
        destinationStage = vk::PipelineStageFlagBits::eAllCommands;
    }



    cmd.pipelineBarrier(sourceStage,destinationStage,vk::DependencyFlags(),0,nullptr,0,nullptr,1,&barrier);


    imageLayout = newLayout;
}

vk::DescriptorImageInfo Texture::getDescriptorInfo()
{
    SAIGA_ASSERT(image);
    vk::DescriptorImageInfo descriptorInfo;
    descriptorInfo.imageLayout = imageLayout;
    descriptorInfo.imageView = imageView;
    descriptorInfo.sampler = sampler;
    SAIGA_ASSERT(imageView && sampler);
    return descriptorInfo;

}

void Texture2D::fromImage(VulkanBase& base, Image &img, vk::ImageUsageFlags usage)
{
    destroy(base);

    mipLevels = 1;
    width = img.width;
    height = img.height;

    //    cout << img.type << endl;
    vk::Format format = getvkFormat(img.type);




    auto finalUsageFlags = usage | vk::ImageUsageFlagBits::eTransferDst;

    imageLayout = vk::ImageLayout::eUndefined;
    // Create optimal tiled target image
    vk::ImageCreateInfo imageCreateInfo;
    imageCreateInfo.imageType = vk::ImageType::e2D;
    imageCreateInfo.format = format;
    imageCreateInfo.mipLevels = mipLevels;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
    imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
    imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
    imageCreateInfo.initialLayout = imageLayout;
    imageCreateInfo.extent = vk::Extent3D{ width, height, 1U };
    imageCreateInfo.usage = finalUsageFlags;
    image = base.device.createImage(imageCreateInfo);
    SAIGA_ASSERT(image);




    auto memReqs = base.device.getImageMemoryRequirements(image);
    memoryLocation = base.memory.getImageAllocator(vk::MemoryPropertyFlagBits::eDeviceLocal).allocate(memReqs.size);
//    base.memory.getAllocator(finalUsageFlags,vk::MemoryPropertyFlagBits::eDeviceLocal).allocate(memReqs.size);
//    DeviceMemory::allocateMemory(base,memReqs,vk::MemoryPropertyFlagBits::eDeviceLocal);

    base.device.bindImageMemory(image,memoryLocation.memory, memoryLocation.offset);

    vk::CommandBuffer cmd = base.createAndBeginTransferCommand();

    transitionImageLayout(cmd,vk::ImageLayout::eTransferDstOptimal);


    vk::BufferImageCopy bufferCopyRegion = {};
    bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = width;
    bufferCopyRegion.imageExtent.height = height;
    bufferCopyRegion.imageExtent.depth = 1;
    bufferCopyRegion.bufferOffset = 0;

    StagingBuffer staging;

    staging.init(base,img.size(),img.data());

    cmd.copyBufferToImage(staging.m_memoryLocation.buffer,image,vk::ImageLayout::eTransferDstOptimal,bufferCopyRegion);

    transitionImageLayout(cmd,vk::ImageLayout::eShaderReadOnlyOptimal);

    base.endTransferWait(cmd);



    staging.destroy();


    vk::ImageViewCreateInfo viewCreateInfo = {};
    viewCreateInfo.viewType = vk::ImageViewType::e2D;
    viewCreateInfo.format = format;
    viewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    viewCreateInfo.image = image;
    imageView = base.device.createImageView(viewCreateInfo);
    SAIGA_ASSERT(imageView);

    // Create a defaultsampler
    vk::SamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.magFilter =  vk::Filter::eLinear;
    samplerCreateInfo.minFilter = vk::Filter::eLinear;
    samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerCreateInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerCreateInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.compareOp = vk::CompareOp::eNever;
    samplerCreateInfo.minLod = 0.0f;
    // Max level-of-detail should match mip level count
    samplerCreateInfo.maxLod = 0.0f;
    // Only enable anisotropic filtering if enabled on the devicec
    samplerCreateInfo.maxAnisotropy = 16;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.borderColor = vk::BorderColor::eIntOpaqueWhite;
    //    VK_CHECK_RESULT(vkCreateSampler(device->device, &samplerCreateInfo, nullptr, &sampler));
    sampler = base.device.createSampler(samplerCreateInfo);
    SAIGA_ASSERT(sampler);

//    cout << "texture created." << endl;
}

void Texture2D::uploadImage(VulkanBase &base, Image &img)
{

    vk::CommandBuffer cmd = base.createAndBeginTransferCommand();

    transitionImageLayout(cmd,vk::ImageLayout::eTransferDstOptimal);


    vk::BufferImageCopy bufferCopyRegion = {};
    bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = width;
    bufferCopyRegion.imageExtent.height = height;
    bufferCopyRegion.imageExtent.depth = 1;
    bufferCopyRegion.bufferOffset = 0;

    StagingBuffer staging;

    staging.init(base,img.size(),img.data());

    cmd.copyBufferToImage(staging.m_memoryLocation.buffer,image,vk::ImageLayout::eTransferDstOptimal,bufferCopyRegion);

    transitionImageLayout(cmd,vk::ImageLayout::eShaderReadOnlyOptimal);

    base.endTransferWait(cmd);



    staging.destroy();
}


}
}
