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
#include "saiga/vulkan/pipeline/Pipeline.h"


namespace Saiga {
namespace Vulkan {



class SAIGA_GLOBAL AssetRenderer : public Pipeline
{
public:
    void destroy();

    void bind(vk::CommandBuffer cmd);


    void pushModel(VkCommandBuffer cmd, mat4 model);
    void updateUniformBuffers(glm::mat4 view, glm::mat4 proj);
    void updateUniformBuffers(vk::CommandBuffer cmd, glm::mat4 view, glm::mat4 proj);

    void init(vks::VulkanDevice* vulkanDevice, VkPipelineCache pipelineCache, VkRenderPass renderPass);

    void prepareUniformBuffers(vks::VulkanDevice* vulkanDevice);
//    void preparePipelines(VkPipelineCache pipelineCache, VkRenderPass renderPass);
    void setupLayoutsAndDescriptors();
private:
    struct UBOVS {
        glm::mat4 projection;
        glm::mat4 modelview;
        glm::vec4 lightPos;
    } uboVS;

    vks::Buffer uniformBufferVS;
    vks::Buffer uniformBufferVS2;
    std::vector<vk::DescriptorSet>       descriptorSet;
};



}
}
