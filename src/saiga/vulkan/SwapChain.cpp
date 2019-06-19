#include "SwapChain.h"

#include "saiga/core/util/easylogging++.h"

#include "VulkanInitializers.hpp"
#include "VulkanTools.h"

void VulkanSwapChain::initSurface(VkSurfaceKHR _surface)
{
    surface = _surface;

    // Get available queue family properties
    uint32_t queueCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, NULL);
    assert(queueCount >= 1);

    std::vector<VkQueueFamilyProperties> queueProps(queueCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, queueProps.data());

    // Iterate over each queue to learn whether it supports presenting:
    // Find a queue with present support
    // Will be used to present the swap chain images to the windowing system
    std::vector<VkBool32> supportsPresent(queueCount);
    for (uint32_t i = 0; i < queueCount; i++)
    {
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &supportsPresent[i]);
    }

    // Search for a graphics and a present queue in the array of queue
    // families, try to find one that supports both
    uint32_t graphicsQueueNodeIndex = UINT32_MAX;
    uint32_t presentQueueNodeIndex  = UINT32_MAX;
    for (uint32_t i = 0; i < queueCount; i++)
    {
        if ((queueProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
        {
            if (graphicsQueueNodeIndex == UINT32_MAX)
            {
                graphicsQueueNodeIndex = i;
            }

            if (supportsPresent[i] == VK_TRUE)
            {
                graphicsQueueNodeIndex = i;
                presentQueueNodeIndex  = i;
                break;
            }
        }
    }
    if (presentQueueNodeIndex == UINT32_MAX)
    {
        // If there's no queue that supports both present and graphics
        // try to find a separate present queue
        for (uint32_t i = 0; i < queueCount; ++i)
        {
            if (supportsPresent[i] == VK_TRUE)
            {
                presentQueueNodeIndex = i;
                break;
            }
        }
    }


    // Exit if either a graphics or a presenting queue hasn't been found
    if (graphicsQueueNodeIndex == UINT32_MAX || presentQueueNodeIndex == UINT32_MAX)
    {
        SAIGA_EXIT_ERROR("Could not find a graphics and/or presenting queue!");
    }

    // todo : Add support for separate graphics and presenting queue
    if (graphicsQueueNodeIndex != presentQueueNodeIndex)
    {
        SAIGA_EXIT_ERROR("Separate graphics and presenting queues are not supported yet!");
    }

    queueNodeIndex = graphicsQueueNodeIndex;

    // Get list of supported surface formats
    uint32_t formatCount;
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, NULL));
    assert(formatCount > 0);

    std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, surfaceFormats.data()));

    // If the surface format list only includes one entry with VK_FORMAT_UNDEFINED,
    // there is no preferered format, so we assume VK_FORMAT_B8G8R8A8_UNORM
    if ((formatCount == 1) && (surfaceFormats[0].format == VK_FORMAT_UNDEFINED))
    {
        colorFormat = VK_FORMAT_B8G8R8A8_UNORM;
        colorSpace  = surfaceFormats[0].colorSpace;
    }
    else
    {
        // iterate over the list of available surface format and
        // check for the presence of VK_FORMAT_B8G8R8A8_UNORM
        bool found_B8G8R8A8_UNORM = false;
        for (auto&& surfaceFormat : surfaceFormats)
        {
            if (surfaceFormat.format == VK_FORMAT_B8G8R8A8_UNORM)
            {
                colorFormat          = surfaceFormat.format;
                colorSpace           = surfaceFormat.colorSpace;
                found_B8G8R8A8_UNORM = true;
                break;
            }
        }

        // in case VK_FORMAT_B8G8R8A8_UNORM is not available
        // select the first available color format
        if (!found_B8G8R8A8_UNORM)
        {
            colorFormat = surfaceFormats[0].format;
            colorSpace  = surfaceFormats[0].colorSpace;
        }
    }
}

void VulkanSwapChain::connect(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device)
{
    this->instance       = instance;
    this->physicalDevice = physicalDevice;
    this->device         = device;
    //		GET_INSTANCE_PROC_ADDR(instance, GetPhysicalDeviceSurfaceSupportKHR);
    //		GET_INSTANCE_PROC_ADDR(instance, GetPhysicalDeviceSurfaceCapabilitiesKHR);
    //		GET_INSTANCE_PROC_ADDR(instance, GetPhysicalDeviceSurfaceFormatsKHR);
    //		GET_INSTANCE_PROC_ADDR(instance, GetPhysicalDeviceSurfacePresentModesKHR);
    //		GET_DEVICE_PROC_ADDR(device, CreateSwapchainKHR);
    //		GET_DEVICE_PROC_ADDR(device, DestroySwapchainKHR);
    //		GET_DEVICE_PROC_ADDR(device, GetSwapchainImagesKHR);
    //		GET_DEVICE_PROC_ADDR(device, AcquireNextImageKHR);
    //		GET_DEVICE_PROC_ADDR(device, QueuePresentKHR);
}

void VulkanSwapChain::create(int* width, int* height, bool vsync)
{
    if (swapChain)
    {
        cleanup();
    }

    // Get physical device surface properties and formats
    VkSurfaceCapabilitiesKHR surfCaps;
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfCaps));

    // Get available present modes
    uint32_t presentModeCount;
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, NULL));
    assert(presentModeCount > 0);

    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    VK_CHECK_RESULT(
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data()));

    VkExtent2D swapchainExtent = {};
    // If width (and height) equals the special value 0xFFFFFFFF, the size of the surface will be set by the swapchain
    if (surfCaps.currentExtent.width == (uint32_t)-1)
    {
        // If the surface size is undefined, the size is set to
        // the size of the images requested.
        swapchainExtent.width  = *width;
        swapchainExtent.height = *height;
    }
    else
    {
        // If the surface size is defined, the swap chain size must match
        swapchainExtent = surfCaps.currentExtent;
        *width          = surfCaps.currentExtent.width;
        *height         = surfCaps.currentExtent.height;
    }

    VkPresentModeKHR swapchainPresentMode;

    if (vsync)
    {
        // this is always supported.
        swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    }
    else
    {
        swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        for (size_t i = 0; i < presentModeCount; i++)
        {
            if (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                swapchainPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
                break;
            }
        }
    }



    // Determine the number of images
    uint32_t desiredNumberOfSwapchainImages = 3;  // surfCaps.minImageCount + 1;
    if ((surfCaps.maxImageCount > 0) && (desiredNumberOfSwapchainImages > surfCaps.maxImageCount))
    {
        desiredNumberOfSwapchainImages = surfCaps.maxImageCount;
    }

    // Find the transformation of the surface
    VkSurfaceTransformFlagsKHR preTransform;
    if (surfCaps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
    {
        // We prefer a non-rotated transform
        preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    }
    else
    {
        preTransform = surfCaps.currentTransform;
    }

    // Find a supported composite alpha format (not all devices support alpha opaque)
    VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    // Simply select the first composite alpha format available
    std::vector<VkCompositeAlphaFlagBitsKHR> compositeAlphaFlags = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
    };
    for (auto& compositeAlphaFlag : compositeAlphaFlags)
    {
        if (surfCaps.supportedCompositeAlpha & compositeAlphaFlag)
        {
            compositeAlpha = compositeAlphaFlag;
            break;
        };
    }

    //    uint32_t queueFamilyIndices[] = {0,0};


    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType                    = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.pNext                    = NULL;
    createInfo.surface                  = surface;
    createInfo.minImageCount            = desiredNumberOfSwapchainImages;
    createInfo.imageFormat              = colorFormat;
    createInfo.imageColorSpace          = colorSpace;
    createInfo.imageExtent              = {swapchainExtent.width, swapchainExtent.height};
    createInfo.imageUsage               = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.preTransform             = (VkSurfaceTransformFlagBitsKHR)preTransform;
    createInfo.imageArrayLayers         = 1;
    createInfo.imageSharingMode         = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount    = 0;
    createInfo.pQueueFamilyIndices      = nullptr;
    createInfo.presentMode              = swapchainPresentMode;
    createInfo.oldSwapchain             = VK_NULL_HANDLE;
    // Setting clipped to VK_TRUE allows the implementation to discard rendering outside of the surface area
    createInfo.clipped        = VK_TRUE;
    createInfo.compositeAlpha = compositeAlpha;

    // Enable transfer source on swap chain images if supported
    if (surfCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
    {
        createInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }

    // Enable transfer destination on swap chain images if supported
    if (surfCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
    {
        createInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;


#if 0
    std::cout << "Swap Chain: surfaceFormat: " << createInfo.imageFormat << " " << createInfo.imageColorSpace << std::endl;
    std::cout << "Swap Chain: presentMode: " << createInfo.presentMode << std::endl;
    std::cout << "Swap Chain: compositeAlpha: " << createInfo.compositeAlpha << std::endl;
    std::cout << "Swap Chain: minImageCount: " << createInfo.minImageCount << std::endl;
    std::cout << "Swap Chain: imageUsage: " << createInfo.imageUsage << std::endl;
    std::cout << "Swap Chain: imageSharingMode: " << createInfo.imageSharingMode << std::endl;
    std::cout << "Swap Chain: preTransform: " << createInfo.preTransform << std::endl;
    std::cout << "Swap Chain: queueFamilyIndexCount: " << createInfo.queueFamilyIndexCount << std::endl;
    std::cout << "Swap Chain: imageExtent: " << createInfo.imageExtent.width << "x" << createInfo.imageExtent.height << std::endl;
#endif



    VK_CHECK_RESULT(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain));

    // If an existing swap chain is re-created, destroy the old swap chain
    // This also cleans up all the presentable images

    VK_CHECK_RESULT(vkGetSwapchainImagesKHR(device, swapChain, &imageCount, NULL));
    images.resize(imageCount);
    VK_CHECK_RESULT(vkGetSwapchainImagesKHR(device, swapChain, &imageCount, images.data()));

    // Get the swap chain buffers containing the image and imageview
    buffers.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; i++)
    {
        VkImageViewCreateInfo colorAttachmentView       = {};
        colorAttachmentView.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        colorAttachmentView.pNext                       = NULL;
        colorAttachmentView.format                      = colorFormat;
        colorAttachmentView.components                  = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                          VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        colorAttachmentView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        colorAttachmentView.subresourceRange.baseMipLevel   = 0;
        colorAttachmentView.subresourceRange.levelCount     = 1;
        colorAttachmentView.subresourceRange.baseArrayLayer = 0;
        colorAttachmentView.subresourceRange.layerCount     = 1;
        colorAttachmentView.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        colorAttachmentView.flags                           = 0;

        buffers[i].image = images[i];

        colorAttachmentView.image = buffers[i].image;

        VK_CHECK_RESULT(vkCreateImageView(device, &colorAttachmentView, nullptr, &buffers[i].view));
    }
}

VkResult VulkanSwapChain::acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t* imageIndex)
{
    // By setting timeout to UINT64_MAX we will always wait until the next image has been acquired or an actual error is
    // thrown With that we don't have to handle VK_NOT_READY
    //		return fpAcquireNextImageKHR(device, swapChain, UINT64_MAX, presentCompleteSemaphore, (VkFence)nullptr,
    // imageIndex);
    return vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, presentCompleteSemaphore, (VkFence) nullptr,
                                 imageIndex);
}

VkResult VulkanSwapChain::queuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore)
{
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType            = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext            = NULL;
    presentInfo.swapchainCount   = 1;
    presentInfo.pSwapchains      = &swapChain;
    presentInfo.pImageIndices    = &imageIndex;
    // Check if a wait semaphore has been specified to wait for before presenting the image
    if (waitSemaphore != VK_NULL_HANDLE)
    {
        presentInfo.pWaitSemaphores    = &waitSemaphore;
        presentInfo.waitSemaphoreCount = 1;
    }
    //		return fpQueuePresentKHR(queue, &presentInfo);
    return vkQueuePresentKHR(queue, &presentInfo);
}

void VulkanSwapChain::cleanup()
{
    VLOG(3) << "Destroying Swap Chain";
    if (swapChain != VK_NULL_HANDLE)
    {
        for (uint32_t i = 0; i < imageCount; i++)
        {
            vkDestroyImageView(device, buffers[i].view, nullptr);
        }
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }
    swapChain = VK_NULL_HANDLE;
}

void VulkanSwapChain::destroySurface()
{
    VLOG(3) << "Destroying Surface";
    if (surface != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(instance, surface, nullptr);
    }
    surface = VK_NULL_HANDLE;
}
