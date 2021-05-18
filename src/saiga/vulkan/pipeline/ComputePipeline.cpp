/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ComputePipeline.h"

#include "saiga/vulkan/Vertex.h"
#include "saiga/vulkan/VulkanInitializers.hpp"

namespace Saiga
{
namespace Vulkan
{
ComputePipeline:: ComputePipeline() : PipelineBase(vk::PipelineBindPoint::eCompute) {}


void ComputePipeline::create()
{
    createPipelineLayout();
    vk::ComputePipelineCreateInfo pipelineCreateInfo(vk::PipelineCreateFlags(), vk::PipelineShaderStageCreateInfo(),
                                                     pipelineLayout, vk::Pipeline(), 0);
    shaderPipeline.addToPipeline(pipelineCreateInfo);
    pipeline = device.createComputePipeline(base->pipelineCache, pipelineCreateInfo);
    SAIGA_ASSERT(pipeline);
}

void ComputePipeline::reload()
{
    shaderPipeline.reload();
    if (!shaderPipeline.valid()) return;
    reloadCounter = 3;
}

bool ComputePipeline::checkShader()
{
    if (autoReload)
    {
        if (shaderPipeline.autoReload())
        {
            if (shaderPipeline.valid()) reloadCounter = 4;
        }
    }


    if (reloadCounter > 0)
    {
        reloadCounter--;

        if (reloadCounter == 0)
        {
            vkDestroyPipeline(device, pipeline, nullptr);
            pipeline = nullptr;

            vk::ComputePipelineCreateInfo pipelineCreateInfo(
                vk::PipelineCreateFlags(), vk::PipelineShaderStageCreateInfo(), pipelineLayout, vk::Pipeline(), 0);
            shaderPipeline.addToPipeline(pipelineCreateInfo);

            pipeline = device.createComputePipeline(base->pipelineCache, pipelineCreateInfo);
            SAIGA_ASSERT(pipeline);

            return true;
        }

        return false;
    }


    return true;
}



}  // namespace Vulkan
}  // namespace Saiga
