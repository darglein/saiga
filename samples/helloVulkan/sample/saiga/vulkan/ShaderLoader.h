/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once


#include "saiga/vulkan/vulkan.h"

namespace Saiga {
namespace Vulkan {

class ShaderLoader
{
public:
    void init(VkDevice _device) { device = _device; }
    void destroy();

    VkPipelineShaderStageCreateInfo loadShader(std::string fileName, VkShaderStageFlagBits stage);
private:
    VkDevice device;
    std::vector<VkShaderModule> shaderModules;

    // Load a SPIR-V shader (binary)
    VkShaderModule loadShader(const char *fileName);

    // Load a GLSL shader (text)
    // Note: GLSL support requires vendor-specific extensions to be enabled and is not a core-feature of Vulkan
    VkShaderModule loadShaderGLSL(const char *fileName, VkShaderStageFlagBits stage);

};

extern ShaderLoader shaderLoader;


}
}
