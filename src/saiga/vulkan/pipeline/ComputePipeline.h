/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/pipeline/PipelineBase.h"
#include "saiga/vulkan/Shader/ShaderPipeline.h"


namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL ComputePipelineInfo
{
public:
    void setShader(Saiga::Vulkan::ShaderModule& shader);
    vk::ComputePipelineCreateInfo createCreateInfo(vk::PipelineLayout pipelineLayout);
private:

    vk::PipelineShaderStageCreateInfo shaderStage;
};


class SAIGA_GLOBAL ComputePipeline : public PipelineBase
{
public:
    ComputePipeline();

    Saiga::Vulkan::ShaderModule shader;


    void create(ComputePipelineInfo pipelineInfo = ComputePipelineInfo());
};


}
}
