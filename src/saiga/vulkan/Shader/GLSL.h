/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/config.h"
#include "saiga/vulkan/svulkan.h"

namespace Saiga
{
namespace Vulkan
{
namespace GLSLANG
{
/**
 * This function parses GLSL code, checks for any syntax errors, and converts it to SPIR-V.
 *
 * It uses the official Khronos GLSL libary glslang.
 * See: https://github.com/KhronosGroup/glslang
 *
 */

SAIGA_VULKAN_API std::vector<uint32_t> loadGLSL(const std::string& file, const vk::ShaderStageFlagBits shader_type,
                                            const std::string& injection = {});

SAIGA_VULKAN_API std::vector<uint32_t> loadSPIRV(const std::string& file);

SAIGA_VULKAN_API void addInjectionAfterVersion(std::string& shaderString, const std::string& injection);



/**
 * Create a SPIRV shader from a GLSL string.
 * Usefull for embedding GLSL code in a .cpp file.
 *
 * Note: #include is not supported in this mode.
 *
 */
SAIGA_VULKAN_API std::vector<uint32_t> createFromString(const std::string& shaderString,
                                                    const vk::ShaderStageFlagBits shader_type);


SAIGA_VULKAN_API void init();
SAIGA_VULKAN_API void quit();


}  // namespace GLSLANG
}  // namespace Vulkan
}  // namespace Saiga
