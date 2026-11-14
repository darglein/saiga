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


    // formats: the attachments of the dynamic rendering pass (vkCmdBeginRendering) the pipeline is used in.
    void create(const RenderingFormats& formats, PipelineInfo pipelineInfo = PipelineInfo());
    void reload();

    // Destroys the pipeline (deferred, see PipelineBase::destroy) and the shader modules.
    void destroy()
    {
        shaderPipeline.destroy();
        PipelineBase::destroy();
    }

    bool autoReload = true;

   protected:
    RenderingFormats formats;
    PipelineInfo pipelineInfo;

    vk::Pipeline createPipeline();

    virtual bool checkShader() override;
};


}  // namespace Vulkan
}  // namespace Saiga
