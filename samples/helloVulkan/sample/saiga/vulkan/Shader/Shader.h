/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Shader/ShaderModule.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL VShader
{
public:
    std::vector<ShaderModule> modules;

    void destroy();

    std::vector<vk::PipelineShaderStageCreateInfo> createPipelineInfo();
};

}
}
