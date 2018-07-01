/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Shader.h"
#include "saiga/vulkan/vulkanHelper.h"

#include "GLSL.h"

namespace Saiga {
namespace Vulkan {

void ShaderPipeline::load(vk::Device device, std::vector<std::string> shaders)
{
    for(auto p : shaders)
    {
        ShaderModule module;
        module.load(device,p);
        modules.push_back(module);
    }
}


void ShaderPipeline::loadGLSL(vk::Device device, std::vector<std::pair<std::string, vk::ShaderStageFlagBits> > shaders)
{
    for(auto p : shaders)
    {
        ShaderModule module;
        module.loadGLSL(device,p.second,p.first);
        modules.push_back(module);
    }
}

void ShaderPipeline::addToPipeline(VkGraphicsPipelineCreateInfo &pipelineCreateInfo)
{
    createPipelineInfo();

    pipelineCreateInfo.stageCount = pipelineInfo.size();
    pipelineCreateInfo.pStages = ( const VkPipelineShaderStageCreateInfo*)pipelineInfo.data();

}


void ShaderPipeline::createPipelineInfo()
{
    pipelineInfo.clear();
    for(auto& s : modules)
    {
        pipelineInfo.push_back(s.createPipelineInfo());
    }
}




}
}
