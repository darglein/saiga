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
 * Returns an empty vector if the file is not found or does not compile.
 * includedFiles (optional) receives the full names of all files included by the shader.
 */
SAIGA_VULKAN_API std::vector<uint32_t> loadGLSL(const std::string& file, const vk::ShaderStageFlagBits shader_type,
                                                const std::string& injection            = {},
                                                std::vector<std::string>* includedFiles = nullptr);

// Returns an empty vector if the file is not found or invalid.
SAIGA_VULKAN_API std::vector<uint32_t> loadSPIRV(const std::string& file);

// Inserts the injection in a new line after the '#version' line. Returns false if there is no '#version' line.
SAIGA_VULKAN_API bool addInjectionAfterVersion(std::string& shaderString, const std::string& injection);



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
