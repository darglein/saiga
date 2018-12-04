/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/config.h"
#include "saiga/util/fileChecker.h"
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

SAIGA_GLOBAL std::vector<uint32_t> loadGLSL(const std::string& file, const vk::ShaderStageFlagBits shader_type,
                                            const std::string& injection = {});

SAIGA_GLOBAL std::vector<uint32_t> loadSPIRV(const std::string& file);

SAIGA_GLOBAL void addInjectionAfterVersion(std::string& shaderString, const std::string& injection);



/**
 * Create a SPIRV shader from a GLSL string.
 * Usefull for embedding GLSL code in a .cpp file.
 *
 * Note: #include is not supported in this mode.
 *
 */
SAIGA_GLOBAL std::vector<uint32_t> createFromString(const std::string& shaderString,
                                                    const vk::ShaderStageFlagBits shader_type);


SAIGA_GLOBAL void init();
SAIGA_GLOBAL void quit();

extern SAIGA_GLOBAL FileChecker shaderPathes;

}  // namespace GLSLANG
}  // namespace Vulkan
}  // namespace Saiga
