/**
 * Copyright (c) 2017 Darius Rückert
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

    bool autoReload = false;

   protected:
    virtual bool checkShader() override;

    int reloadCounter = 0;
};


}  // namespace Vulkan
}  // namespace Saiga
