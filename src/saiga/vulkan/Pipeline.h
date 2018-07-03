/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/Shader/Shader.h"


namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL Pipeline
{
public:
protected:
    vk::Device device;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;

    std::vector<vk::DescriptorSetLayout> descriptorSetLayout;
    std::vector<vk::DescriptorSet>       descriptorSet;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    Saiga::Vulkan::ShaderPipeline shaderPipeline;

    void destroy();


    void createDescriptorSetLayout(std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings);

    void createPipelineLayout();
};


}
}
