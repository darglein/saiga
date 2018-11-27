/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Pipeline.h"
#include "saiga/vulkan/Vertex.h"
#include "saiga/vulkan/VulkanInitializers.hpp"

namespace Saiga
{
namespace Vulkan
{
Pipeline::Pipeline() : PipelineBase(vk::PipelineBindPoint::eGraphics) {}


void Pipeline::create(vk::RenderPass renderPass, PipelineInfo pipelineInfo)
{
    SAIGA_ASSERT(isInitialized());

    this->renderPass   = renderPass;
    this->pipelineInfo = pipelineInfo;

    createPipelineLayout();
    //    pipelineInfo.addShaders(shaderPipeline);
    auto pipelineCreateInfo = pipelineInfo.createCreateInfo(pipelineLayout, renderPass);
    shaderPipeline.addToPipeline(pipelineCreateInfo);
    pipeline = device.createGraphicsPipeline(base->pipelineCache, pipelineCreateInfo);
    SAIGA_ASSERT(pipeline);
    shaderPipeline.destroy();
}

void Pipeline::reload()
{
    shaderPipeline.reload();
    if (!shaderPipeline.valid()) return;
    reloadCounter = 3;
}

bool Pipeline::checkShader()
{
    if (reloadCounter > 0)
    {
        reloadCounter--;

        if (reloadCounter == 0)
        {
            cout << "recreating pipeline" << endl;
            vkDestroyPipeline(device, pipeline, nullptr);
            pipeline = nullptr;

            auto pipelineCreateInfo = pipelineInfo.createCreateInfo(pipelineLayout, renderPass);
            shaderPipeline.addToPipeline(pipelineCreateInfo);
            pipeline = device.createGraphicsPipeline(base->pipelineCache, pipelineCreateInfo);
            SAIGA_ASSERT(pipeline);

            shaderPipeline.destroy();
            return true;
        }

        return false;
    }


    return true;
}

}  // namespace Vulkan
}  // namespace Saiga
