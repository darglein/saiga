/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#define VK_USE_PLATFORM_XCB_KHR
#include "saiga/config.h"
#include "saiga/util/glm.h"

#include "vulkan/vk_sdk_platform.h"
#include "vulkan/vulkan.hpp"
#include "xcb/xcb.h"

namespace Saiga {


struct LayerPropertiesEx
{
    VkLayerProperties properties;
    std::vector<VkExtensionProperties> instance_extensions;
    std::vector<VkExtensionProperties> device_extensions;
};


class SAIGA_GLOBAL VulkanWindow
{
public:
    VulkanWindow();
    ~VulkanWindow();
private:
    std::string name = "test";
    std::vector<LayerPropertiesEx> layerProperties;


    vk::Instance inst;
    std::vector<vk::PhysicalDevice> physicalDevices;
    vk::PhysicalDevice physicalDevice;
    vk::PhysicalDeviceProperties gpu_props;
    vk::PhysicalDeviceMemoryProperties memory_properties;

    vk::Device device;

    std::vector<vk::QueueFamilyProperties> queueFamilyProperties;
    vk::CommandPool cmd_pool;
    vk::CommandBuffer cmd;
    vk::SwapchainKHR swap_chain;

    std::vector<vk::Image> swapChainImages;
    std::vector<vk::ImageView> swapChainImagesViews;

    vk::Format format;

    //depth image
    vk::Format depthFormat;
    vk::Image depthimage;
    vk::DeviceMemory depthmem;
    vk::ImageView depthview;

    std::vector<vk::DescriptorSetLayout> desc_layout;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorPool desc_pool;
    std::vector<vk::DescriptorSet> desc_set;
    uint32_t current_buffer;
    vk::RenderPass render_pass;
    vk::PipelineShaderStageCreateInfo shaderStages[2];

    std::vector<vk::Framebuffer> framebuffers;


    vk::Buffer vertexbuf;
    vk::DeviceMemory vertexmem;
    vk::DescriptorBufferInfo vertexbuffer_info;

    vk::VertexInputBindingDescription vi_binding;
    vk::VertexInputAttributeDescription vi_attribs[2];

    vk::Pipeline pipeline;

    // ======= Window =======


    glm::mat4 Projection;
    glm::mat4 View;
    glm::mat4 Model;
    glm::mat4 Clip;
    glm::mat4 MVP;
    vk::Buffer uniformbuf;
    vk::DeviceMemory uniformmem;
    vk::DescriptorBufferInfo uniformbuffer_info;


    int width = 50;
    int height = 50;

    vk::SurfaceKHR surface;

    xcb_connection_t *connection;
    xcb_screen_t* screen;
    xcb_window_t window;
    xcb_intern_atom_reply_t *atom_wm_delete_window;



    int graphics_queue_family_index = -1;
    int present_queue_family_index = -1;

    void init_global_layer_properties();
    void init_instance();
    void init_physical_device();
    void init_swapchain_extension();
    void init_depth_buffer();
    void init_uniform_buffer();
    void createWindow();
    void createDevice();
    void init_vertex_buffer();
};

}

