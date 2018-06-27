/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#define VK_USE_PLATFORM_XCB_KHR
#include "saiga/config.h"
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
    vk::Device device;

    std::vector<vk::QueueFamilyProperties> queueFamilyProperties;
    vk::CommandPool cmd_pool;
    vk::CommandBuffer cmd;
    vk::SwapchainKHR swap_chain;

    std::vector<vk::Image> swapChainImages;
    std::vector<vk::ImageView> swapChainImagesViews;

    vk::Format format;
    // ======= Window =======

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
    void createWindow();
    void createDevice();
};

}
