/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/geometry/triangle_mesh.h"
#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/VulkanBuffer.hpp"
#include "saiga/vulkan/VulkanAsset.h"
#include "saiga/vulkan/pipeline/Pipeline.h"


namespace Saiga {
namespace Vulkan {



class SAIGA_GLOBAL PointCloudRenderer : public Pipeline
{
public:
    using VertexType = VertexNC;
    void destroy();

    void bind(vk::CommandBuffer cmd);


    void pushModel(VkCommandBuffer cmd, mat4 model);
    void updateUniformBuffers(glm::mat4 view, glm::mat4 proj);
    void updateUniformBuffers(vk::CommandBuffer cmd, glm::mat4 view, glm::mat4 proj);

    void init(Saiga::Vulkan::VulkanBase& vulkanDevice, VkRenderPass renderPass, float pointSize);

    void prepareUniformBuffers(Saiga::Vulkan::VulkanBase* vulkanDevice);
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
    vk::DescriptorSet       descriptorSet;
};



}
}
