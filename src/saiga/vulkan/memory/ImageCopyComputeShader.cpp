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
    pipeline->addDescriptorSetLayout({{0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
                                      {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}});


    pipeline->shaderPipeline.loadCompute(_base->device, "vulkan/img_copy.comp");
    pipeline->create();
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

void ImageCopyComputeShader::copy_image(ImageMemoryLocation* target, ImageMemoryLocation* source)
{
    auto descriptorSet = pipeline->createDescriptorSet();

    // vk::DescriptorBufferInfo descriptorInfo = compute.storageBuffer.createInfo();
    // auto iinfo                              = compute.storageTexture.getDescriptorInfo();
    // device.updateDescriptorSets(
    //    {
    //        vk::WriteDescriptorSet(descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
    //        &descriptorInfo,
    //                               nullptr),
    //        vk::WriteDescriptorSet(descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageImage, &iinfo, nullptr,
    //        nullptr),
    //    },
    //    nullptr);
    //
    //
    //// compute.queue.create(device, vulkanDevice->queueFamilyIndices.compute);
    // compute.commandBuffer = base.computeQueue->commandPool.allocateCommandBuffer();
    //
    //{
    //    // Build the command buffer
    //    compute.commandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    //    if (computePipeline.bind(compute.commandBuffer))
    //    {
    //        computePipeline.bindDescriptorSets(compute.commandBuffer, descriptorSet);
    //        // Dispatch 1 block
    //        compute.commandBuffer.dispatch(1, 1, 1);
    //        compute.commandBuffer.end();
    //    }
    //}
    //
    //
    // base.computeQueue->submitAndWait(compute.commandBuffer);
}
}  // namespace Saiga::Vulkan::Memory
