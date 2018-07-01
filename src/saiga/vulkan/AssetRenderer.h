/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/geometry/triangle_mesh.h"
#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/base/VulkanDevice.hpp"
#include "saiga/vulkan/base/VulkanBuffer.hpp"
#include "saiga/vulkan/base/VulkanModel.hpp"
namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL Asset
{
public:
       Saiga::TriangleMesh<Saiga::VertexNC, uint32_t> mesh;


    VkDevice device = nullptr;
    vks::Buffer vertices;
    vks::Buffer indices;
    uint32_t indexCount = 0;
    uint32_t vertexCount = 0;

    void load(const std::string& file, vks::VulkanDevice *device, VkQueue copyQueue);
};



class SAIGA_GLOBAL AssetRenderer
{
public:
//    static vks::VertexLayout vertexLayout;
    static int asdf;
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
