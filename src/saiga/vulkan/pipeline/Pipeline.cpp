/**
 * Copyright (c) 2021 Darius Rückert
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


void Pipeline::create(const RenderingFormats& _formats, PipelineInfo _pipelineInfo)
{
    SAIGA_ASSERT(isInitialized());

    formats      = _formats;
    pipelineInfo = _pipelineInfo;

    createPipelineLayout();
    SAIGA_ASSERT(shaderPipeline.valid(), "Shader compilation failed");
    replacePipeline(createPipeline());
}

vk::Pipeline Pipeline::createPipeline()
{
    auto pipelineCreateInfo = pipelineInfo.createCreateInfo(pipelineLayout, formats);
    shaderPipeline.addToPipeline(pipelineCreateInfo);
    auto result = device.createGraphicsPipeline(base->pipelineCache, pipelineCreateInfo);
    SAIGA_ASSERT(result.result == vk::Result::eSuccess && result.value);
    return result.value;
}

void Pipeline::reload()
{
    shaderPipeline.reload();
    if (!shaderPipeline.valid()) return;
    replacePipeline(createPipeline());
}

bool Pipeline::checkShader()
{
    if (autoReload && shaderPipeline.autoReload() && shaderPipeline.valid())
    {
        // The old pipeline is destroyed when the frames in flight are done with it.
        replacePipeline(createPipeline());
    }
    return pipeline;
}

}  // namespace Vulkan
}  // namespace Saiga
