/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/vulkan.h"

namespace Saiga {
namespace Vulkan {

/**
 * This function parses GLSL code, checks for any syntax errors, and converts it to SPIR-V.
 *
 * It uses the official Khronos GLSL libary glslang.
 * See: https://github.com/KhronosGroup/glslang
 *
 * TODO: glslang init and cleanup only once
 */

std::vector<uint32_t> GLSLtoSPIRV(
        const std::string& shaderString,
        const vk::ShaderStageFlagBits shader_type
        );

}
}
