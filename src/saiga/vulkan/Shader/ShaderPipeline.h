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
    ~ShaderPipelineBase() { destroy(); }

    bool load(vk::Device device, std::vector<std::string> shaders);
    bool loadGLSL(vk::Device device,
                  std::vector<std::tuple<std::string, vk::ShaderStageFlagBits, std::string> > shaders);

    bool loadCompute(vk::Device device, std::string shader, std::string injection = {});


    void destroy();
    void reload();

    // Returns if any of the shader modules was reloaded
    bool autoReload();

    /**
     * Checks if at least 1 shader module is loaded and
     * all shader modules are valid.
     */
    bool valid();

   protected:
    vk::Device device;
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
