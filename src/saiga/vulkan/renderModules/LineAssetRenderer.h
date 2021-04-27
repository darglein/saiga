/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/core/geometry/triangle_mesh.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/VulkanAsset.h"
#include "saiga/vulkan/VulkanBuffer.hpp"
#include "saiga/vulkan/buffer/UniformBuffer.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/svulkan.h"

namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API LineAssetRenderer : public Pipeline
{
   public:
    using VertexType = VertexNC;
    ~LineAssetRenderer() { destroy(); }
    void destroy();

    SAIGA_WARN_UNUSED_RESULT bool bind(vk::CommandBuffer cmd);


    void pushModel(VkCommandBuffer cmd, mat4 model, vec4 color = make_vec4(1));

    void updateUniformBuffers(vk::CommandBuffer cmd, mat4 view, mat4 proj);

    void init(Saiga::Vulkan::VulkanBase& vulkanDevice, VkRenderPass renderPass, float lineWidth);

    void prepareUniformBuffers(Saiga::Vulkan::VulkanBase* vulkanDevice);
    //    void preparePipelines(VkPipelineCache pipelineCache, VkRenderPass renderPass);
    void setupLayoutsAndDescriptors();

   private:
    struct UBOVS
    {
        mat4 projection;
        mat4 modelview;
        vec4 lightPos;
    } uboVS;

    struct PC
    {
        mat4 model;
        vec4 color;
    } pc;

    UniformBuffer uniformBufferVS;
    StaticDescriptorSet descriptorSet;
};



}  // namespace Vulkan
}  // namespace Saiga
