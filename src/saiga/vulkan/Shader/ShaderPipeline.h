/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Shader/ShaderModule.h"

namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API ShaderPipelineBase
{
   public:
    ShaderPipelineBase() = default;
    ~ShaderPipelineBase() { destroy(); }

    // Owns the shader modules
    ShaderPipelineBase(const ShaderPipelineBase&)                = delete;
    ShaderPipelineBase& operator=(const ShaderPipelineBase&)     = delete;
    ShaderPipelineBase(ShaderPipelineBase&&) noexcept            = default;
    ShaderPipelineBase& operator=(ShaderPipelineBase&&) noexcept = default;

    // The load functions replace the current modules.

    bool load(vk::Device device, std::vector<std::string> shaders);
    bool loadGLSL(vk::Device device,
                  std::vector<std::tuple<std::string, vk::ShaderStageFlagBits, std::string> > shaders);

    bool loadCompute(vk::Device device, std::string shader, std::string injection = {});


    // Destroys the shader modules. Pipelines created from them stay valid.
    void destroy();

    // Recompiles all modules loaded from GLSL files (a module that fails to compile keeps its previous version).
    // Returns true if any module was recompiled.
    bool reload();

    // Returns if any of the shader modules was reloaded successfully. Cheap, see ShaderModule::autoReload().
    bool autoReload();

    /**
     * Checks if at least 1 shader module is loaded and
     * all shader modules are valid.
     */
    bool valid();

   protected:
    std::vector<vk::PipelineShaderStageCreateInfo> pipelineInfo;
    std::vector<ShaderModule> modules;

    void createPipelineInfo();
};

class SAIGA_VULKAN_API GraphicsShaderPipeline : public ShaderPipelineBase
{
   public:
    void addToPipeline(vk::GraphicsPipelineCreateInfo& pipelineCreateInfo);
};

class SAIGA_VULKAN_API ComputeShaderPipeline : public ShaderPipelineBase
{
   public:
    void addToPipeline(vk::ComputePipelineCreateInfo& pipelineCreateInfo);
};

}  // namespace Vulkan
}  // namespace Saiga
