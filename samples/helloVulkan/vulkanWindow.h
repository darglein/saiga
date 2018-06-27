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
    vk::Device device;
    vk::CommandPool cmd_pool;
    vk::CommandBuffer cmd;

    // ======= Window =======

    int width = 50;
    int height = 50;

    xcb_connection_t *connection;
    xcb_screen_t* screen;
    xcb_window_t window;
    xcb_intern_atom_reply_t *atom_wm_delete_window;
};

}
