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

    // Load a SPIR-V shader (binary)
    VkShaderModule loadShader(const char *fileName, VkDevice device);

    // Load a GLSL shader (text)
    // Note: GLSL support requires vendor-specific extensions to be enabled and is not a core-feature of Vulkan
    VkShaderModule loadShaderGLSL(const char *fileName, VkDevice device, VkShaderStageFlagBits stage);

    VkPipelineShaderStageCreateInfo loadShader(VkDevice device, std::string fileName, VkShaderStageFlagBits stage);
private:
    std::vector<VkShaderModule> shaderModules;
};

extern ShaderLoader shaderLoader;


}
}
