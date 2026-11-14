//
// Created by Peter Eichinger on 2019-02-14.
//

#include "ImageCopyComputeShader.h"

#include "saiga/vulkan/pipeline/ComputePipeline.h"

#include "memory.h"

namespace Saiga::Vulkan::Memory
{
void ImageCopyComputeShader::init(VulkanBase* _base)
{
    base = _base;

    pipeline = new ComputePipeline;

    pipeline->init(*_base, 1);
    pipeline->addDescriptorSetLayout(
        {{0, {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}},
         {1, {1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute}}});

    pipeline->addPushConstantRange(vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ivec2)});

    bool valid = pipeline->shaderPipeline.loadCompute(_base->device, "vulkan/img_copy.comp");

    if (!valid)
    {
        LOG(ERROR) << "Couldn't load image copy compute shader";
        return;
    }

    pipeline->create();

    // The shader uses texelFetch, so the sampler state does not matter. Own sampler: the source may have none.
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.magFilter    = vk::Filter::eNearest;
    samplerInfo.minFilter    = vk::Filter::eNearest;
    samplerInfo.mipmapMode   = vk::SamplerMipmapMode::eNearest;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.maxLod       = 0.0f;
    sampler                  = base->device.createSampler(samplerInfo);

    initialized = true;
}

ImageCopyComputeShader::~ImageCopyComputeShader()
{
    destroy();
}

void ImageCopyComputeShader::destroy()
{
    if (pipeline)
    {
        // Destroys its Vulkan objects deferred.
        pipeline->destroy();
        delete pipeline;
        pipeline = nullptr;
    }
    if (sampler)
    {
        // Only used by the defragmentation copies, which are complete when this is called (VulkanMemory::destroy).
        base->device.destroy(sampler);
        sampler = nullptr;
    }
    initialized = false;
}

bool ImageCopyComputeShader::can_copy(const ImageData& data)
{
    const auto& info = data.image_create_info;
    const auto usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
    return data.image && data.view && info.imageType == vk::ImageType::e2D && info.mipLevels == 1 &&
           info.arrayLayers == 1 && info.samples == vk::SampleCountFlagBits::e1 &&
           info.tiling == vk::ImageTiling::eOptimal && info.format == vk::Format::eR8G8B8A8Unorm &&
           data.view_create_info.format == vk::Format::eR8G8B8A8Unorm && (info.usage & usage) == usage &&
           (data.layout == vk::ImageLayout::eShaderReadOnlyOptimal || data.layout == vk::ImageLayout::eGeneral);
}

std::optional<vk::DescriptorSet> ImageCopyComputeShader::copy_image(vk::CommandBuffer cmd, ImageMemoryLocation* target,
                                                                    ImageMemoryLocation* source)
{
    std::scoped_lock lock(mutex);
    if (!initialized || !can_copy(source->data) || !target->data.view)
    {
        return std::nullopt;
    }

    if (!pipeline->bind(cmd))
    {
        return std::nullopt;
    }

    const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    const auto finalLayout = source->data.layout;

    // The new target: undefined -> general for the storage image writes.
    vk::ImageMemoryBarrier2 toGeneral;
    toGeneral.srcStageMask        = vk::PipelineStageFlagBits2::eNone;
    toGeneral.srcAccessMask       = vk::AccessFlagBits2::eNone;
    toGeneral.dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader;
    toGeneral.dstAccessMask       = vk::AccessFlagBits2::eShaderStorageWrite;
    toGeneral.oldLayout           = vk::ImageLayout::eUndefined;
    toGeneral.newLayout           = vk::ImageLayout::eGeneral;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.image               = target->data.image;
    toGeneral.subresourceRange    = range;
    vk::DependencyInfo toGeneralDep;
    toGeneralDep.imageMemoryBarrierCount = 1;
    toGeneralDep.pImageMemoryBarriers    = &toGeneral;
    cmd.pipelineBarrier2(toGeneralDep);
    target->data.layout = vk::ImageLayout::eGeneral;

    // The source is read in its current layout. Its earlier writes are visible through the semaphore wait of the
    // submission (frame timeline).
    auto descriptorSet = pipeline->createRawDescriptorSet();

    vk::DescriptorImageInfo target_info(nullptr, target->data.view, vk::ImageLayout::eGeneral);
    vk::DescriptorImageInfo source_info(sampler, source->data.view, source->data.layout);
    base->device.updateDescriptorSets(
        {
            vk::WriteDescriptorSet(descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageImage, &target_info, nullptr,
                                   nullptr),
            vk::WriteDescriptorSet(descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &source_info,
                                   nullptr, nullptr),
        },
        nullptr);

    pipeline->bindRawDescriptorSet(cmd, descriptorSet);

    const auto extent = source->data.image_create_info.extent;
    ivec2 size{(int)extent.width, (int)extent.height};
    pipeline->pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(ivec2), &size, 0);

    cmd.dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);

    // Make the storage writes available and visible to all later commands (also of later submissions, i.e. the
    // frames which use the image after the swap) and transition to the layout of the source.
    vk::ImageMemoryBarrier2 toFinal;
    toFinal.srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader;
    toFinal.srcAccessMask       = vk::AccessFlagBits2::eShaderStorageWrite;
    toFinal.dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
    toFinal.dstAccessMask       = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
    toFinal.oldLayout           = vk::ImageLayout::eGeneral;
    toFinal.newLayout           = finalLayout;
    toFinal.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toFinal.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toFinal.image               = target->data.image;
    toFinal.subresourceRange    = range;
    vk::DependencyInfo toFinalDep;
    toFinalDep.imageMemoryBarrierCount = 1;
    toFinalDep.pImageMemoryBarriers    = &toFinal;
    cmd.pipelineBarrier2(toFinalDep);
    target->data.layout = finalLayout;

    return descriptorSet;
}
}  // namespace Saiga::Vulkan::Memory
