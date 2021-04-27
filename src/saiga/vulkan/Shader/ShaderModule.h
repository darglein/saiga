/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/core/util/FileSystem.h"
#include "saiga/vulkan/svulkan.h"


namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API ShaderModule
{
   public:
    /**
     * One of the following:
     *
     * vk::ShaderStageFlagBits::eVertex
     * vk::ShaderStageFlagBits::eTessellationControl
     * vk::ShaderStageFlagBits::eTessellationEvaluation
     * vk::ShaderStageFlagBits::eGeometry
     * vk::ShaderStageFlagBits::eFragment
     * vk::ShaderStageFlagBits::eCompute
     * vk::ShaderStageFlagBits::eAllGraphics
     * vk::ShaderStageFlagBits::eAll
     */
    vk::ShaderStageFlagBits stage;
    vk::ShaderModule module = nullptr;


    /**
     * Create the shader module from data provided in host memory.
     * If the shader is stored in an external file use the "load" functions below.
     */
    void createSPIRV(vk::Device device, vk::ShaderStageFlagBits _stage, const void* data, size_t size);
    void createSPIRV(vk::Device device, vk::ShaderStageFlagBits _stage, const std::vector<uint32_t>& data);
    void createGLSL(vk::Device device, vk::ShaderStageFlagBits _stage, const std::string& data);

    /**
     * Load and create the shader module from a file.
     * The "load" function without a stage parameter picks the correct,
     * flag from the file ending. See "ShaderHelper" for more details.
     *
     * Code injections only work for GLSL shaders.
     *
     * Returns true if the validation succeeds.
     */
    bool load(vk::Device device, const std::string& file, const std::string& injection = {});
    bool loadSPIRV(vk::Device device, vk::ShaderStageFlagBits _stage, const std::string& file);
    bool loadGLSL(vk::Device device, vk::ShaderStageFlagBits _stage, const std::string& file,
                  const std::string& injection = {});



    /**
     * Create the module info block used in pipepline creation.
     */
    vk::PipelineShaderStageCreateInfo createPipelineInfo();

    /**
     * Destroys the vulkan object.
     */
    void destroy();
    void reload();
    bool valid();

    /**
     * Checks if the file was changed for reloading.
     * Returns true if it was reloaded.
     */
    bool autoReload();

   private:
    // These variables are only required for the reloading.
    vk::Device device;
    std::string file;
    std::string injection;
#ifdef SAIGA_USE_FILESYSTEM
    std::filesystem::file_time_type lastWrite;
#endif
};

}  // namespace Vulkan
}  // namespace Saiga
