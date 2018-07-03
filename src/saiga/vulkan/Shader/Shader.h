/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Shader/ShaderModule.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL ShaderPipeline
{
public:
    std::vector<ShaderModule> modules;

    void addToPipeline(VkGraphicsPipelineCreateInfo& pipelineCreateInfo);

    void load(vk::Device device, std::vector<std::string> shaders);
    void loadGLSL(vk::Device device, std::vector<std::pair<std::string, vk::ShaderStageFlagBits> > shaders);

    void destroy(vk::Device device);
private:
    std::vector<vk::PipelineShaderStageCreateInfo> pipelineInfo;


    void createPipelineInfo();
};

}
}
