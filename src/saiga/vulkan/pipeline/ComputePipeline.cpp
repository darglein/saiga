/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ComputePipeline.h"
#include "saiga/vulkan/VulkanInitializers.hpp"
#include "saiga/vulkan/Vertex.h"

namespace Saiga {
namespace Vulkan {

void ComputePipelineInfo::setShader(Saiga::Vulkan::ShaderModule& shader)
{
    shaderStage = shader.createPipelineInfo();
}

vk::ComputePipelineCreateInfo ComputePipelineInfo::createCreateInfo(vk::PipelineLayout pipelineLayout)
{
    vk::ComputePipelineCreateInfo pipelineCreateInfo(
                vk::PipelineCreateFlags(),
                shaderStage,
                pipelineLayout,
                vk::Pipeline(),
                0
                );
    return pipelineCreateInfo;
}




void ComputePipeline::preparePipelines(ComputePipelineInfo &pipelineInfo, VkPipelineCache pipelineCache)
{ 
    createPipelineLayout();
    pipelineInfo.setShader(shader);
    auto pipelineCreateInfo= pipelineInfo.createCreateInfo(pipelineLayout);
    pipeline = device.createComputePipeline(pipelineCache,pipelineCreateInfo);
    SAIGA_ASSERT(pipeline);
}



}
}
