/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ComputePipeline.h"

#include "saiga/vulkan/Vertex.h"
#include "saiga/vulkan/VulkanInitializers.hpp"

#include "PipelineKey.h"

namespace Saiga
{
namespace Vulkan
{
ComputePipeline::ComputePipeline() : PipelineBase(vk::PipelineBindPoint::eCompute) {}


void ComputePipeline::create()
{
    createPipelineLayout();
    SAIGA_ASSERT(shaderPipeline.valid(), "Shader compilation failed");
    setPipeline(base->reusePipelines ? reuseKey() : std::string(),
                [this](vk::PipelineLayout layout) { return createPipeline(layout); });
}

std::string ComputePipeline::reuseKey()
{
    vk::ComputePipelineCreateInfo ci(vk::PipelineCreateFlags(), vk::PipelineShaderStageCreateInfo(), pipelineLayout,
                                     vk::Pipeline(), 0);
    shaderPipeline.addToPipeline(ci);
    PipelineKeyWriter k;
    k.bytes = "C";
    k.none(ci.pNext);
    k.f(ci.flags);
    k.stage(ci.stage, shaderPipeline);
    writeLayoutKey(k.bytes);
    return k.valid ? k.bytes : std::string();
}

vk::Pipeline ComputePipeline::createPipeline(vk::PipelineLayout layout)
{
    vk::ComputePipelineCreateInfo pipelineCreateInfo(vk::PipelineCreateFlags(), vk::PipelineShaderStageCreateInfo(),
                                                     layout, vk::Pipeline(), 0);
    shaderPipeline.addToPipeline(pipelineCreateInfo);
    auto result = device.createComputePipeline(base->pipelineCache, pipelineCreateInfo);
    SAIGA_ASSERT(result.result == vk::Result::eSuccess && result.value);
    return result.value;
}

void ComputePipeline::reload()
{
    shaderPipeline.reload();
    if (!shaderPipeline.valid()) return;
    replacePipeline(createPipeline(pipelineLayout));
}

bool ComputePipeline::checkShader()
{
    if (autoReload && shaderPipeline.autoReload() && shaderPipeline.valid())
    {
        replacePipeline(createPipeline(pipelineLayout));
    }
    return pipeline;
}

}  // namespace Vulkan
}  // namespace Saiga
