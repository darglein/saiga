/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/vulkan.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL ShaderModule
{
public:
    /**
     * One of the following:
     *
     * vk::ShaderStageFlagBits::eVertex
     * vk::ShaderStageFlagBits::eTessellationControl
     * vk::ShaderStageFlagBits::eTessellationEvaluation
     * vk::ShaderStageFlagBits::eGeometry
     * vk::ShaderStageFlagBits::eFragment
     * vk::ShaderStageFlagBits::eCompute
     * vk::ShaderStageFlagBits::eAllGraphics
     * vk::ShaderStageFlagBits::eAll
     */
    vk::ShaderStageFlagBits stage;
    vk::ShaderModule module = nullptr;


    void create(vk::Device device, vk::ShaderStageFlagBits _stage, std::vector<uint32_t> spirvCode);
    void destroy(vk::Device device);

    vk::PipelineShaderStageCreateInfo createPipelineInfo();
};

}
}
