/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "vulkanWindow.h"
#include "saiga/util/assert.h"
namespace Saiga {


VulkanWindow::VulkanWindow()
{


    // ======================= Layers =======================


    vk::Result res;


        init_global_layer_properties();
    init_instance();
    init_physical_device();

    createWindow();

    init_swapchain_extension();


    createDevice();



    {

        vk::SurfaceCapabilitiesKHR surfCapabilities;

        res = physicalDevice.getSurfaceCapabilitiesKHR(surface,&surfCapabilities);
        SAIGA_ASSERT(res == vk::Result::eSuccess);

        std::vector<vk::PresentModeKHR> presentModes = physicalDevice.getSurfacePresentModesKHR(surface);

        vk::Extent2D swapchainExtent(width,height);


        // The FIFO present mode is guaranteed by the spec to be supported
        vk::PresentModeKHR swapchainPresentMode = vk::PresentModeKHR::eFifo;

        // Determine the number of VkImage's to use in the swap chain.
        // We need to acquire only 1 presentable image at at time.
        // Asking for minImageCount images ensures that we can acquire
        // 1 presentable image as long as we present it before attempting
        // to acquire another.
        uint32_t desiredNumberOfSwapChainImages = surfCapabilities.minImageCount;

        vk::SurfaceTransformFlagBitsKHR preTransform;
        if (surfCapabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity) {
            preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
        } else {
            preTransform = surfCapabilities.currentTransform;
        }

        // Find a supported composite alpha mode - one of these is guaranteed to be set
        vk::CompositeAlphaFlagBitsKHR compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        //        vk::CompositeAlphaFlagBitsKHR compositeAlphaFlags[4] =
        //        {
        //            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        //            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        //            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        //            VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
        //        };
        //        for (uint32_t i = 0; i < sizeof(compositeAlphaFlags); i++) {
        //            if (surfCapabilities.supportedCompositeAlpha & compositeAlphaFlags[i]) {
        //                compositeAlpha = compositeAlphaFlags[i];
        //                break;
        //            }
        //        }



        vk::SwapchainCreateInfoKHR swapchain_ci = {};
        swapchain_ci.surface = surface;
        swapchain_ci.minImageCount = desiredNumberOfSwapChainImages;
        swapchain_ci.imageFormat = format;
        swapchain_ci.imageExtent.width = swapchainExtent.width;
        swapchain_ci.imageExtent.height = swapchainExtent.height;
        swapchain_ci.preTransform = preTransform;
        swapchain_ci.compositeAlpha = compositeAlpha;
        swapchain_ci.imageArrayLayers = 1;
        swapchain_ci.presentMode = swapchainPresentMode;
        //        swapchain_ci.oldSwapchain = VK_NULL_HANDLE;
        swapchain_ci.clipped = true;
        swapchain_ci.imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
        swapchain_ci.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
        swapchain_ci.imageSharingMode = vk::SharingMode::eExclusive;
        swapchain_ci.queueFamilyIndexCount = 0;
        swapchain_ci.pQueueFamilyIndices = NULL;

        SAIGA_ASSERT(graphics_queue_family_index == present_queue_family_index);

        res = device.createSwapchainKHR(&swapchain_ci,nullptr,&swap_chain);
        SAIGA_ASSERT(res == vk::Result::eSuccess);


        swapChainImages = device.getSwapchainImagesKHR(swap_chain);
        swapChainImagesViews.resize(swapChainImages.size());


        for (uint32_t i = 0; i < swapChainImages.size(); i++)
        {
            vk::ImageViewCreateInfo color_image_view = {};
//            color_image_view.flags = 0;
            color_image_view.image = swapChainImages[i];
            color_image_view.viewType = vk::ImageViewType::e2D;
            color_image_view.format = format;
            color_image_view.components.r = vk::ComponentSwizzle::eR;
            color_image_view.components.g = vk::ComponentSwizzle::eG;
            color_image_view.components.b = vk::ComponentSwizzle::eB;
            color_image_view.components.a = vk::ComponentSwizzle::eA;
            color_image_view.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            color_image_view.subresourceRange.baseMipLevel = 0;
            color_image_view.subresourceRange.levelCount = 1;
            color_image_view.subresourceRange.baseArrayLayer = 0;
            color_image_view.subresourceRange.layerCount = 1;

//            res = vkCreateImageView(info.device, &color_image_view, NULL, &info.buffers[i].view);
            res = device.createImageView(&color_image_view,nullptr,&swapChainImagesViews[i]);
            SAIGA_ASSERT(res == vk::Result::eSuccess);
//            assert(res == VK_SUCCESS);
        }
    }


    // ======================= Command Buffer =======================

    {
        vk::CommandPoolCreateInfo cmd_pool_info = {};
        cmd_pool_info.queueFamilyIndex = graphics_queue_family_index;

        res = device.createCommandPool(&cmd_pool_info, nullptr, &cmd_pool);
        SAIGA_ASSERT(res == vk::Result::eSuccess);

        vk::CommandBufferAllocateInfo cmd_info = {};
        cmd_info.commandPool = cmd_pool;
        cmd_info.level = vk::CommandBufferLevel::ePrimary;
        cmd_info.commandBufferCount = 1;

        res = device.allocateCommandBuffers(&cmd_info,&cmd);
        SAIGA_ASSERT(res == vk::Result::eSuccess);

    }



}

VulkanWindow::~VulkanWindow()
{
    for(vk::ImageView& iv : swapChainImagesViews)
    {
        device.destroyImageView(iv);
    }
    device.destroySwapchainKHR(swap_chain);

    device.freeCommandBuffers(cmd_pool,cmd);
    device.destroyCommandPool(cmd_pool);
    device.destroy();
    inst.destroy();
}



}
