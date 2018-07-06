/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"

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


    /**
     * Create the shader module from data provided in host memory.
     * If the shader is stored in an external file use the "load" functions below.
     */
    void createSPIRV(vk::Device device, vk::ShaderStageFlagBits _stage, const void* data, size_t size);
    void createSPIRV(vk::Device device, vk::ShaderStageFlagBits _stage, const std::vector<uint32_t>& data);
    void createGLSL (vk::Device device, vk::ShaderStageFlagBits _stage, const std::string& data);

    /**
     * Load and create the shader module from a file.
     * The "load" function without a stage parameter picks the correct,
     * flag from the file ending. See "ShaderHelper" for more details.
     *
     * Code injections only work for GLSL shaders.
     */
    void load     (vk::Device device, const std::string& file, const std::string& injection = {});
    void loadSPIRV(vk::Device device, vk::ShaderStageFlagBits _stage, const std::string& file);
    void loadGLSL (vk::Device device, vk::ShaderStageFlagBits _stage, const std::string& file,  const std::string& injection = {});



    /**
     * Create the module info block used in pipepline creation.
     */
    vk::PipelineShaderStageCreateInfo createPipelineInfo();

    /**
     * Destroys the vulkan object.
     */
    void destroy(vk::Device device);
};

}
}
