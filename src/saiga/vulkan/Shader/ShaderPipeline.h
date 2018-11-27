/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Shader/ShaderModule.h"

namespace Saiga
{
namespace Vulkan
{
class SAIGA_GLOBAL ShaderPipelineBase
{
   public:
    ~ShaderPipelineBase() { destroy(); }

    void load(vk::Device device, std::vector<std::string> shaders);
    void loadGLSL(vk::Device device,
                  std::vector<std::tuple<std::string, vk::ShaderStageFlagBits, std::string> > shaders);

    void loadCompute(vk::Device device, std::string shader, std::string injection = {});


    void destroy();
    void reload();

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

class SAIGA_GLOBAL GraphicsShaderPipeline : public ShaderPipelineBase
{
   public:
    void addToPipeline(vk::GraphicsPipelineCreateInfo& pipelineCreateInfo);
};

class SAIGA_GLOBAL ComputeShaderPipeline : public ShaderPipelineBase
{
   public:
    void addToPipeline(vk::ComputePipelineCreateInfo& pipelineCreateInfo);
};

}  // namespace Vulkan
}  // namespace Saiga
