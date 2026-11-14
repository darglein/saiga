/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>


namespace Saiga
{
namespace Vulkan
{
/**
 * Owns a vk::ShaderModule (move-only). The module is destroyed in destroy() or the destructor.
 *
 * Note: Vulkan allows destroying a shader module while pipelines created from it are still in use, so no deferred
 * deletion is necessary.
 */
class SAIGA_VULKAN_API ShaderModule
{
   public:
    ShaderModule() = default;
    ~ShaderModule() { destroy(); }

    ShaderModule(const ShaderModule&)            = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ShaderModule(ShaderModule&& other) noexcept;
    ShaderModule& operator=(ShaderModule&& other) noexcept;

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
    vk::ShaderStageFlagBits stage = vk::ShaderStageFlagBits::eVertex;
    vk::ShaderModule module       = nullptr;


    /**
     * Create the shader module from data provided in host memory.
     * If the shader is stored in an external file use the "load" functions below.
     * Modules created this way are not reloaded.
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

    /**
     * Recompiles a module loaded from a GLSL file. On failure the current module is kept.
     * Returns true if the module was recompiled. Modules created from memory or SPIR-V are not reloaded.
     */
    bool reload();
    bool valid();

    /**
     * Checks if the file (or one of its includes) was changed and reloads it.
     * The file times are checked at most every 0.5 s, so this is cheap enough to call on every bind.
     * Returns true if it was reloaded successfully.
     */
    bool autoReload();

   private:
    // These variables are only required for the reloading.
    vk::Device device;
    std::string file;
    std::string injection;
    bool isSpirv = false;
    // The file followed by all included files and their last write times.
    std::vector<std::string> dependencies;
    std::vector<std::filesystem::file_time_type> lastWrites;
    std::chrono::steady_clock::time_point lastCheck;

    void createModule(vk::Device device, vk::ShaderStageFlagBits _stage, const void* data, size_t size);
    std::vector<std::filesystem::file_time_type> getWriteTimes() const;
};

}  // namespace Vulkan
}  // namespace Saiga
