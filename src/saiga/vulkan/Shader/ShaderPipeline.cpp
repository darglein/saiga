/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ShaderPipeline.h"

#include "GLSL.h"

namespace Saiga
{
namespace Vulkan
{
bool ShaderPipelineBase::load(vk::Device device, std::vector<std::string> shaders)
{
    destroy();
    bool valid = true;
    for (auto& p : shaders)
    {
        modules.emplace_back();
        valid &= modules.back().load(device, p);
    }
    return valid;
}


bool ShaderPipelineBase::loadGLSL(vk::Device device,
                                  std::vector<std::tuple<std::string, vk::ShaderStageFlagBits, std::string> > shaders)
{
    destroy();
    bool valid = true;
    for (auto& p : shaders)
    {
        modules.emplace_back();
        valid &= modules.back().loadGLSL(device, std::get<1>(p), std::get<0>(p), std::get<2>(p));
    }
    return valid;
}

bool ShaderPipelineBase::loadCompute(vk::Device device, std::string shader, std::string injection)
{
    destroy();
    ShaderModule module;
    bool valid = module.load(device, shader, injection);
    if (valid)
    {
        modules.push_back(std::move(module));
    }
    return valid;
}

void ShaderPipelineBase::destroy()
{
    // Vulkan allows destroying shader modules while pipelines created from them are in use, so this does not have
    // to be deferred.
    modules.clear();
    pipelineInfo.clear();
}

bool ShaderPipelineBase::reload()
{
    bool r = false;
    for (auto& s : modules)
    {
        r |= s.reload();
    }
    return r;
}

bool ShaderPipelineBase::autoReload()
{
    bool r = false;
    for (auto& s : modules)
    {
        r |= s.autoReload();
    }
    return r;
}

bool ShaderPipelineBase::valid()
{
    if (modules.empty()) return false;

    for (auto& s : modules)
    {
        if (!s.valid()) return false;
    }
    return true;
}



void ShaderPipelineBase::createPipelineInfo()
{
    pipelineInfo.clear();
    for (auto& s : modules)
    {
        pipelineInfo.push_back(s.createPipelineInfo());
    }
}

void GraphicsShaderPipeline::addToPipeline(vk::GraphicsPipelineCreateInfo& pipelineCreateInfo)
{
    createPipelineInfo();

    pipelineCreateInfo.stageCount = pipelineInfo.size();
    pipelineCreateInfo.pStages    = pipelineInfo.data();
}

void ComputeShaderPipeline::addToPipeline(vk::ComputePipelineCreateInfo& pipelineCreateInfo)
{
    SAIGA_ASSERT(!modules.empty());
    createPipelineInfo();
    pipelineCreateInfo.stage = pipelineInfo.front();
}



}  // namespace Vulkan
}  // namespace Saiga
