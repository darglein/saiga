/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Shader/ShaderPipeline.h"
#include "saiga/vulkan/pipeline/PipelineBase.h"


namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API ComputePipeline : public PipelineBase
{
   public:
    ComputePipeline();
    void create();


    ComputeShaderPipeline shaderPipeline;

    void reload();

    // Destroys the pipeline (deferred, see PipelineBase::destroy) and the shader modules.
    void destroy()
    {
        shaderPipeline.destroy();
        PipelineBase::destroy();
    }

    bool autoReload = true;

   protected:
    virtual bool checkShader() override;

    // With the given layout (pipelineLayout, or the pool's creation layout)
    vk::Pipeline createPipeline(vk::PipelineLayout layout);
    // The content key of the pipeline reuse (empty: none)
    std::string reuseKey();
};


}  // namespace Vulkan
}  // namespace Saiga
