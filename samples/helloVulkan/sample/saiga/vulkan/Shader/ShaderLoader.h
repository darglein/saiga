/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once


#include "saiga/vulkan/vulkan.h"
#include "saiga/vulkan/Shader/ShaderModule.h"
#include "saiga/vulkan/Shader/Shader.h"

namespace Saiga {
namespace Vulkan {

class ShaderLoader
{
public:
    void init(VkDevice _device) { device = _device; }
    void destroy();

    vk::PipelineShaderStageCreateInfo loadShader(std::string fileName, vk::ShaderStageFlagBits stage);
    vk::PipelineShaderStageCreateInfo loadShaderGLSL(std::string fileName, vk::ShaderStageFlagBits stage);
private:
    VkDevice device;
    std::vector<ShaderModule> shaderModules;

    // Load a SPIR-V shader (binary)
    ShaderModule loadModule(const char *fileName, vk::ShaderStageFlagBits stage);

    // Load a GLSL shader (text)
    // Note: GLSL support requires vendor-specific extensions to be enabled and is not a core-feature of Vulkan
    ShaderModule loadModuleGLSL(const char *fileName, vk::ShaderStageFlagBits stage);

};

extern ShaderLoader shaderLoader;


}
}
