/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/vulkanBase.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL Shader
{
public:

vk::PipelineShaderStageCreateInfo shaderStages[2];
    void init(VulkanBase& base);
};

}
}
