/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/Base.h"

namespace Saiga {
namespace Vulkan {

/**
 * Base class for both a graphics and compute pipeline
 */

class SAIGA_GLOBAL PipelineBase
{
public:

    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
    std::vector<vk::PushConstantRange> pushConstantRanges;

    void init(VulkanBase& base , uint32_t numDescriptorSetLayouts);
    void destroy();


    void setDescriptorSetLayout(std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings, uint32_t id = 0);
    void addPushConstantRange(vk::PushConstantRange pcr);

    vk::DescriptorSet createDescriptorSet(uint32_t id = 0);



protected:
    VulkanBase* base = nullptr;
    vk::Device device;
    bool isInitialized() { return base; }
    void createPipelineLayout();
};


}
}
