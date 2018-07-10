/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ShaderPipeline.h"

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


void ShaderPipeline::loadGLSL(vk::Device device, std::vector<std::tuple<std::string, vk::ShaderStageFlagBits, std::string> > shaders)
{
    for(auto p : shaders)
    {
        ShaderModule module;
//        module.loadGLSL(device,p.second,p.first);
        module.loadGLSL(device,std::get<1>(p),std::get<0>(p),std::get<2>(p));
        modules.push_back(module);
    }
}

void ShaderPipeline::destroy(vk::Device device)
{
    for(auto& s : modules)
    {
        s.destroy(device);
    }
    modules.clear();
    pipelineInfo.clear();
}

void ShaderPipeline::addToPipeline(vk::GraphicsPipelineCreateInfo& pipelineCreateInfo)
{
    createPipelineInfo();

    pipelineCreateInfo.stageCount = pipelineInfo.size();
    pipelineCreateInfo.pStages = pipelineInfo.data();

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
