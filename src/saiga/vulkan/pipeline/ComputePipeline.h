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

class SAIGA_GLOBAL ComputePipelineInfo
{
public:
    void setShader(Saiga::Vulkan::ShaderModule& shader);
    vk::ComputePipelineCreateInfo createCreateInfo(vk::PipelineLayout pipelineLayout);
private:

    vk::PipelineShaderStageCreateInfo shaderStage;
};


class SAIGA_GLOBAL ComputePipeline
{
public:
    vk::Device device;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;

    std::vector<vk::DescriptorSetLayout> descriptorSetLayout;


    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    Saiga::Vulkan::ShaderModule shader;

    void destroy();


    void createDescriptorSetLayout(std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings);

    void createPipelineLayout(std::vector<vk::PushConstantRange> pushConstantRanges);

    void createDescriptorPool(int maxDescriptorSets, std::vector<vk::DescriptorPoolSize> poolSizes);
    void preparePipelines(ComputePipelineInfo &pipelineInfo, VkPipelineCache pipelineCache);
};


}
}
