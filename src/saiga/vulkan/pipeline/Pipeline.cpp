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




void Pipeline::preparePipelines(PipelineInfo& pipelineInfo, VkPipelineCache pipelineCache, vk::RenderPass renderPass)
{
    createPipelineLayout();
    pipelineInfo.addShaders(shaderPipeline);
    auto pipelineCreateInfo= pipelineInfo.createCreateInfo(pipelineLayout,renderPass);
    pipeline = device.createGraphicsPipeline(pipelineCache,pipelineCreateInfo);
    SAIGA_ASSERT(pipeline);
}

}
}
