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

void Pipeline::bind(vk::CommandBuffer cmd)
{
    if (reloadFence && fenceAdded)
    {
        //        auto res = device.getFenceStatus(reloadFence);
        auto res = device.getEventStatus(reloadFence);

        if (res == vk::Result::eSuccess)
        {
            vkDestroyPipeline(device, pipeline, nullptr);
            pipeline = nullptr;

            auto pipelineCreateInfo = pipelineInfo.createCreateInfo(pipelineLayout, renderPass);
            pipeline                = device.createGraphicsPipeline(base->pipelineCache, pipelineCreateInfo);
            SAIGA_ASSERT(pipeline);

            device.destroyEvent(reloadFence);
            reloadFence = nullptr;
        }
        shaderPipeline.destroy(device);
    }

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    if (reloadFence && fenceAdded)
    {
        //        cmd.setEvent(event);
    }
}

void Pipeline::create(vk::RenderPass renderPass, PipelineInfo pipelineInfo)
{
    SAIGA_ASSERT(isInitialized());

    this->renderPass   = renderPass;
    this->pipelineInfo = pipelineInfo;

    createPipelineLayout();
    pipelineInfo.addShaders(shaderPipeline);
    auto pipelineCreateInfo = pipelineInfo.createCreateInfo(pipelineLayout, renderPass);
    pipeline                = device.createGraphicsPipeline(base->pipelineCache, pipelineCreateInfo);
    SAIGA_ASSERT(pipeline);
    shaderPipeline.destroy(device);
}

void Pipeline::reload()
{
    shaderPipeline.reload();
    pipelineInfo.addShaders(shaderPipeline);

    reloadFence = device.createEvent({});
    fenceAdded  = false;
}

}  // namespace Vulkan
}  // namespace Saiga
