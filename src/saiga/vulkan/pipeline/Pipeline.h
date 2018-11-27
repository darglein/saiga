/**
 * Copyright (c) 2017 Darius Rückert
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
class SAIGA_GLOBAL Pipeline : public PipelineBase
{
   public:
    Saiga::Vulkan::ShaderPipeline shaderPipeline;

    Pipeline();


    void create(vk::RenderPass renderPass, PipelineInfo pipelineInfo = PipelineInfo());
    void reload();

   protected:
    bool fenceAdded = false;
    vk::Event reloadFence;
    vk::RenderPass renderPass;
    PipelineInfo pipelineInfo;
    int reloadCounter = 0;

    virtual bool checkShader(vk::CommandBuffer cmd) override;
};


}  // namespace Vulkan
}  // namespace Saiga
