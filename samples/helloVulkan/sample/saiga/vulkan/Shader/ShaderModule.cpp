/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ShaderModule.h"

#include "GLSL.h"

namespace Saiga {
namespace Vulkan {

void ShaderModule::create(vk::Device device, vk::ShaderStageFlagBits _stage, std::vector<uint32_t> spirvCode)
{
    stage = _stage;

    vk::ShaderModuleCreateInfo moduleCreateInfo{};
//    moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleCreateInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
    moduleCreateInfo.pCode = spirvCode.data();

//    vkCreateShaderModule(device, &moduleCreateInfo, NULL, &module);
    device.createShaderModule(&moduleCreateInfo, NULL, &module);
}

void ShaderModule::destroy(vk::Device device)
{
    if(module)
    {
        vkDestroyShaderModule(device, module, nullptr);
        module = nullptr;
    }
}

vk::PipelineShaderStageCreateInfo ShaderModule::createPipelineInfo()
{
    vk::PipelineShaderStageCreateInfo info;

    info.pSpecializationInfo = NULL;
    info.stage = stage;
    info.pName = "main";
    info.module = module;

    return info;
}




}
}
