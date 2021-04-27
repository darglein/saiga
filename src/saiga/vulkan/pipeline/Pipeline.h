/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/FrameSync.h"
#include "saiga/vulkan/Shader/ShaderPipeline.h"
#include "saiga/vulkan/pipeline/PipelineBase.h"
#include "saiga/vulkan/pipeline/PipelineInfo.h"

namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API Pipeline : public PipelineBase
{
   public:
    Saiga::Vulkan::GraphicsShaderPipeline shaderPipeline;

    Pipeline();


    void create(vk::RenderPass renderPass, PipelineInfo pipelineInfo = PipelineInfo());
    void reload();

    bool autoReload = true;

   protected:
    vk::RenderPass renderPass;
    PipelineInfo pipelineInfo;
    int reloadCounter = 0;

    virtual bool checkShader() override;
};


}  // namespace Vulkan
}  // namespace Saiga
