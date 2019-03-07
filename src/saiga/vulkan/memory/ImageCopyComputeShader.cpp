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

    pipeline->addPushConstantRange(vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(glm::ivec2)});

    bool valid = pipeline->shaderPipeline.loadCompute(_base->device, "vulkan/img_copy.comp");

    if (!valid)
    {
        LOG(ERROR) << "Couldn't load image copy compute shader";
        return;
    }

    pipeline->create();

    initialized = true;
}



void ImageCopyComputeShader::destroy()
{
    if (pipeline)
    {
        pipeline->destroy();
        delete pipeline;
        pipeline = nullptr;
    }
}

bool ImageCopyComputeShader::copy_image(ImageMemoryLocation* target, ImageMemoryLocation* source)
{
    auto cmd = base->computeQueue->commandPool.createAndBeginOneTimeBuffer();

    target->data.transitionImageLayout(cmd, vk::ImageLayout::eGeneral);

    cmd.end();
    base->computeQueue->submitAndWait(cmd);

    base->computeQueue->commandPool.freeCommandBuffer(cmd);

    auto descriptorSet = pipeline->createDescriptorSet();

    auto source_info = source->data.get_descriptor_info();
    auto target_info = target->data.get_descriptor_info();
    base->device.updateDescriptorSets(
        {
            vk::WriteDescriptorSet(descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageImage, &target_info, nullptr,
                                   nullptr),
            vk::WriteDescriptorSet(descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &source_info,
                                   nullptr, nullptr),
        },
        nullptr);

    cmd = base->computeQueue->commandPool.createAndBeginOneTimeBuffer();

    if (!pipeline->bind(cmd))
    {
        return false;
    }


    pipeline->bindDescriptorSet(cmd, descriptorSet);

    const auto extent = source->data.image_create_info.extent;
    int countX        = extent.width / 8 + 1;
    int countY        = extent.height / 8 + 1;

    glm::ivec2 size{extent.width, extent.height};
    pipeline->pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(glm::ivec2), &size, 0);

    cmd.dispatch(countX, countY, 1);

    target->data.transitionImageLayout(cmd, source->data.layout);
    cmd.end();
    base->computeQueue->submitAndWait(cmd);

    base->computeQueue->commandPool.freeCommandBuffer(cmd);
    return true;
}
}  // namespace Saiga::Vulkan::Memory
