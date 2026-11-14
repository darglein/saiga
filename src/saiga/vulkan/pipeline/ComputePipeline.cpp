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
ComputePipeline::ComputePipeline() : PipelineBase(vk::PipelineBindPoint::eCompute) {}


void ComputePipeline::create()
{
    createPipelineLayout();
    SAIGA_ASSERT(shaderPipeline.valid(), "Shader compilation failed");
    replacePipeline(createPipeline());
}

vk::Pipeline ComputePipeline::createPipeline()
{
    vk::ComputePipelineCreateInfo pipelineCreateInfo(vk::PipelineCreateFlags(), vk::PipelineShaderStageCreateInfo(),
                                                     pipelineLayout, vk::Pipeline(), 0);
    shaderPipeline.addToPipeline(pipelineCreateInfo);
    auto result = device.createComputePipeline(base->pipelineCache, pipelineCreateInfo);
    SAIGA_ASSERT(result.result == vk::Result::eSuccess && result.value);
    return result.value;
}

void ComputePipeline::reload()
{
    shaderPipeline.reload();
    if (!shaderPipeline.valid()) return;
    replacePipeline(createPipeline());
}

bool ComputePipeline::checkShader()
{
    if (autoReload && shaderPipeline.autoReload() && shaderPipeline.valid())
    {
        replacePipeline(createPipeline());
    }
    return pipeline;
}

}  // namespace Vulkan
}  // namespace Saiga
