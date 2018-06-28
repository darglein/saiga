/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "window.h"
#include "swapChain.h"
#include "vulkan.h"
#include "depthBuffer.h"
#include "VertexBuffer.h"


namespace Saiga {


struct LayerPropertiesEx
{
    VkLayerProperties properties;
    std::vector<VkExtensionProperties> instance_extensions;
    std::vector<VkExtensionProperties> device_extensions;
};


class SAIGA_GLOBAL VulkanWindow : public Vulkan::VulkanBase
{
public:
    VulkanWindow();
    ~VulkanWindow();
private:
    std::string name = "test";
    std::vector<LayerPropertiesEx> layerProperties;


    Vulkan::Window window;



//    vk::Queue graphics_queue;
//    vk::Queue present_queue;

    vk::CommandPool cmd_pool;
    vk::CommandBuffer cmd;
//    vk::SwapchainKHR swap_chain;

    Vulkan::SwapChain* swapChain;
    Vulkan::DepthBuffer depthBuffer;
    Vulkan::VertexBuffer vertexBuffer;
//    std::vector<vk::Image> swapChainImages;
//    std::vector<vk::ImageView> swapChainImagesViews;

//    vk::Format format;



    std::vector<vk::DescriptorSetLayout> desc_layout;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorPool desc_pool;
    std::vector<vk::DescriptorSet> desc_set;
    uint32_t current_buffer;
    vk::RenderPass render_pass;
    vk::PipelineShaderStageCreateInfo shaderStages[2];

    std::vector<vk::Framebuffer> framebuffers;



    vk::Pipeline pipeline;
    vk::PipelineCache pipelineCache;
    // ======= Window =======


    glm::mat4 Projection;
    glm::mat4 View;
    glm::mat4 Model;
    glm::mat4 Clip;
    glm::mat4 MVP;
    vk::Buffer uniformbuf;
    vk::DeviceMemory uniformmem;
    vk::DescriptorBufferInfo uniformbuffer_info;



//    vk::SurfaceKHR surface;



//    int graphics_queue_family_index = -1;
//    int present_queue_family_index = -1;

    void init_global_layer_properties();
    void init_instance();
    void init_swapchain_extension();
//    void init_depth_buffer();
    void init_uniform_buffer();
//    void createWindow();
//    void createDevice();
//    void init_vertex_buffer();
};

}

