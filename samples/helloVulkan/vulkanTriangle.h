/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/all.h"

namespace Saiga {



class SAIGA_GLOBAL VulkanWindow : public Vulkan::Application
{
public:
        glm::mat4 MVP;


    VulkanWindow();
    ~VulkanWindow();

    virtual void update(vk::CommandBuffer& cmd) override;
    virtual void render(vk::CommandBuffer& cmd) override;
private:





    Vulkan::VertexBuffer<Vulkan::Vertex> vertexBuffer;
    Vulkan::IndexBuffer<uint32_t> indexBuffer;
    Vulkan::Shader shader;
    Vulkan::UniformBuffer uniformBuffer;


    std::vector<vk::DescriptorSetLayout> desc_layout;
    vk::DescriptorPool desc_pool;
    std::vector<vk::DescriptorSet> desc_set;






    vk::PipelineLayout pipeline_layout;
    vk::Pipeline pipeline;
    vk::PipelineCache pipelineCache;

};

}

