/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Pipeline.h"
#include "saiga/vulkan/VulkanInitializers.hpp"
#include "saiga/vulkan/Vertex.h"

namespace Saiga {
namespace Vulkan {



Pipeline::Pipeline()
    : PipelineBase(vk::PipelineBindPoint::eGraphics)
{

}

void Pipeline::create(vk::RenderPass renderPass, PipelineInfo pipelineInfo)
{
    SAIGA_ASSERT(isInitialized());
    createPipelineLayout();
    pipelineInfo.addShaders(shaderPipeline);
    auto pipelineCreateInfo= pipelineInfo.createCreateInfo(pipelineLayout,renderPass);
    pipeline = device.createGraphicsPipeline(base->pipelineCache,pipelineCreateInfo);
    SAIGA_ASSERT(pipeline);
    shaderPipeline.destroy(device);
}

}
}
