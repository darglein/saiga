/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/geometry/triangle_mesh.h"
#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/Device.h"
#include "saiga/vulkan/VulkanBuffer.hpp"
#include "saiga/vulkan/VulkanAsset.h"

namespace Saiga {
namespace Vulkan {



class SAIGA_GLOBAL AssetRenderer
{
public:
    void init();
    void destroy();


    void bind(VkCommandBuffer cmd);

    void prepareUniformBuffers(vks::VulkanDevice* vulkanDevice);
    void preparePipelines(VkDevice device, VkPipelineCache pipelineCache, VkRenderPass renderPass);
    void setupLayoutsAndDescriptors(VkDevice device);
    void updateUniformBuffers(glm::mat4 view, glm::mat4 proj);
private:
    VkDevice device;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSet;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    struct UBOVS {
        glm::mat4 projection;
        glm::mat4 modelview;
        glm::vec4 lightPos;
    } uboVS;

    vks::Buffer uniformBufferVS;
};



}
}
