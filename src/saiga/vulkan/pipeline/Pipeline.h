/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/pipeline/PipelineBase.h"
#include "saiga/vulkan/Shader/ShaderPipeline.h"
#include "saiga/vulkan/pipeline/PipelineInfo.h"

namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL Pipeline : public PipelineBase
{
public:

    Saiga::Vulkan::ShaderPipeline shaderPipeline;


    void preparePipelines(PipelineInfo &pipelineInfo, VkPipelineCache pipelineCache, vk::RenderPass renderPass);
};


}
}
