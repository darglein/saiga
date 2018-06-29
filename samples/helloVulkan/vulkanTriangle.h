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
    VulkanWindow();
    ~VulkanWindow();

    virtual void update() override;
    virtual void render(vk::CommandBuffer& cmd) override;
private:





    Vulkan::VertexBuffer vertexBuffer;
    Vulkan::IndexBuffer indexBuffer;
    Vulkan::Shader shader;
    Vulkan::UniformBuffer uniformBuffer;


    std::vector<vk::DescriptorSetLayout> desc_layout;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorPool desc_pool;
    std::vector<vk::DescriptorSet> desc_set;






    vk::Pipeline pipeline;
    vk::PipelineCache pipelineCache;

};

}

