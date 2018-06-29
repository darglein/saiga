/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "swapChain.h"

namespace Saiga {
namespace Vulkan {

SwapChain::SwapChain()
{

}

SwapChain::~SwapChain()
{
    if (swapChain)
    {
        for (auto& buffer : buffers)
        {
            device.destroyImageView(buffer.view);
        }
    }
    if (surface)
    {
        device.destroySwapchainKHR(swapChain);
        instance.destroySurfaceKHR(surface);
    }
}

void SwapChain::create(vk::Instance instance, vk::PhysicalDevice physicalDevice, vk::Device device)
{
    this->instance = instance;
    this->physicalDevice = physicalDevice;
    this->device = device;
}

void SwapChain::setSurface(vk::SurfaceKHR _surface)

{
    surface = _surface;


    assert(surface);

    // Get available queue family properties
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties;
    queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    // Iterate over each queue to learn whether it supports presenting:
    // Find a queue with present support
    // Will be used to present the swap chain images to the windowing system
    std::vector<vk::Bool32> supportsPresent(queueFamilyProperties.size());
    for (size_t i = 0; i < queueFamilyProperties.size(); i++)
    {
        supportsPresent[i] = physicalDevice.getSurfaceSupportKHR(i, surface);
    }

    // Search for a graphics and a present queue in the array of queue families, try to find one that supports both
    uint32_t graphicsQueueNodeIndex = UINT32_MAX;
    uint32_t presentQueueNodeIndex = UINT32_MAX;
    for (size_t i = 0; i < queueFamilyProperties.size(); i++)
    {
        if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics)
        {
            if (graphicsQueueNodeIndex == UINT32_MAX)
            {
                graphicsQueueNodeIndex = i;
            }

            if (supportsPresent[i] == VK_TRUE)
            {
                graphicsQueueNodeIndex = i;
                presentQueueNodeIndex = i;
                break;
            }
        }
    }
    if (presentQueueNodeIndex == UINT32_MAX)
    {
        // If there's no queue that supports both present and graphics try to find a separate present queue
        for (size_t i = 0; i < queueFamilyProperties.size(); i++)
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
        //vkTools::exitFatal("Could not find a graphics and/or presenting queue!", "Fatal error");
        SAIGA_ASSERT(0);
    }

    // todo : Add support for separate graphics and presenting queue
    if (graphicsQueueNodeIndex != presentQueueNodeIndex)
    {
        //vkTools::exitFatal("Separate graphics and presenting queues are not supported yet!", "Fatal error");
        SAIGA_ASSERT(0);
    }

    queueNodeIndex = graphicsQueueNodeIndex;

    // Get list of supported surface formats
    std::vector<vk::SurfaceFormatKHR> surfaceFormats;
    surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface);

    // If the surface format list only includes one entry with VK_FORMAT_UNDEFINED,
    // there is no preferered format, so we assume VK_FORMAT_B8G8R8A8_UNORM
    if ((surfaceFormats.size() == 1) && (surfaceFormats[0].format == vk::Format::eUndefined))
    {
        colorFormat = vk::Format::eB8G8R8A8Unorm;
    }
    else
    {
        // Always select the first available color format
        // If you need a specific format (e.g. SRGB) you'd need to
        // iterate over the list of available surface format and
        // check for it's presence
        colorFormat = surfaceFormats[0].format;
    }
    colorSpace = surfaceFormats[0].colorSpace;

    // Find a suitable depth (stencil) format that is supported by the device
    std::vector<vk::Format> depthFormats = {
        vk::Format::eD32SfloatS8Uint,
        vk::Format::eD24UnormS8Uint,
        vk::Format::eD16UnormS8Uint,
        vk::Format::eD32Sfloat,
        vk::Format::eD16Unorm
    };

    for (auto& format : depthFormats) {
        vk::FormatProperties formatProps = physicalDevice.getFormatProperties(format);
        if (formatProps.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
            depthFormat = format;
            break;
        }
    }

}

void SwapChain::create(uint32_t *width, uint32_t *height, bool vsync)
{
    vk::SwapchainKHR oldSwapchain = swapChain;

    // Get physical device surface properties and formats
    vk::SurfaceCapabilitiesKHR surfCaps = physicalDevice.getSurfaceCapabilitiesKHR(surface);

    // Get available present modes
    std::vector<vk::PresentModeKHR> presentModes;
    presentModes = physicalDevice.getSurfacePresentModesKHR(surface);
    assert(presentModes.size() > 0);

    vk::Extent2D swapchainExtent;
    // If width (and height) equals the special value 0xFFFFFFFF, the size of the surface will be set by the swapchain
    if (surfCaps.currentExtent.width == (uint32_t)-1)
    {
        // If the surface size is undefined, the size is set to
        // the size of the images requested.
        swapchainExtent.width = *width;
        swapchainExtent.height = *height;
    }
    else
    {
        // If the surface size is defined, the swap chain size must match
        swapchainExtent = surfCaps.currentExtent;
        *width = surfCaps.currentExtent.width;
        *height = surfCaps.currentExtent.height;
    }


    // Select a present mode for the swapchain

    // The VK_PRESENT_MODE_FIFO_KHR mode must always be present as per spec
    // This mode waits for the vertical blank ("v-sync")
    vk::PresentModeKHR swapchainPresentMode = vk::PresentModeKHR::eFifo;

    // If v-sync is not requested, try to find a mailbox mode
    // It's the lowest latency non-tearing present mode available
    if (!vsync)
    {
        for (auto presentMode : presentModes)
        {
            if (presentMode == vk::PresentModeKHR::eMailbox)
            {
                swapchainPresentMode = vk::PresentModeKHR::eMailbox;
                break;
            }
            if ((swapchainPresentMode != vk::PresentModeKHR::eMailbox) && (presentMode == vk::PresentModeKHR::eImmediate))
            {
                swapchainPresentMode = vk::PresentModeKHR::eImmediate;
            }
        }
    }

    // Determine the number of images
    uint32_t desiredNumberOfSwapchainImages = surfCaps.minImageCount + 1;
    if ((surfCaps.maxImageCount > 0) && (desiredNumberOfSwapchainImages > surfCaps.maxImageCount))
    {
        desiredNumberOfSwapchainImages = surfCaps.maxImageCount;
    }

    // Find the transformation of the surface, prefer a non-rotated transform
    vk::SurfaceTransformFlagBitsKHR preTransform;
    if (surfCaps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity)
    {
        preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    }
    else
    {
        preTransform = surfCaps.currentTransform;
    }

    vk::SwapchainCreateInfoKHR swapchainCI;
    swapchainCI.surface = surface;
    swapchainCI.minImageCount = desiredNumberOfSwapchainImages;
    swapchainCI.imageFormat = colorFormat;
    swapchainCI.imageColorSpace = colorSpace;
    swapchainCI.imageExtent = swapchainExtent;
    swapchainCI.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    swapchainCI.preTransform = preTransform;
    swapchainCI.imageArrayLayers = 1;
    swapchainCI.imageSharingMode = vk::SharingMode::eExclusive;
    swapchainCI.queueFamilyIndexCount = 0;
    swapchainCI.pQueueFamilyIndices = NULL;
    swapchainCI.presentMode = swapchainPresentMode;
    swapchainCI.oldSwapchain = oldSwapchain;
    swapchainCI.clipped = VK_TRUE;
    swapchainCI.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;

    swapChain = device.createSwapchainKHR(swapchainCI);

    // If an existing sawp chain is re-created, destroy the old swap chain
    // This also cleans up all the presentable images
    if (oldSwapchain)
    {
        for (auto& buffer : buffers)
        {
            device.destroyImageView(buffer.view);
        }
        device.destroySwapchainKHR(oldSwapchain);
    }

    // Get the swap chain images
    images = device.getSwapchainImagesKHR(swapChain);

    // Get the swap chain buffers containing the image and imageview
    buffers.resize(images.size());
    for (size_t i = 0; i < buffers.size(); i++)
    {
        vk::ImageViewCreateInfo colorAttachmentView;
        colorAttachmentView.format = colorFormat;
        colorAttachmentView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        colorAttachmentView.subresourceRange.baseMipLevel = 0;
        colorAttachmentView.subresourceRange.levelCount = 1;
        colorAttachmentView.subresourceRange.baseArrayLayer = 0;
        colorAttachmentView.subresourceRange.layerCount = 1;
        colorAttachmentView.viewType = vk::ImageViewType::e2D;

        buffers[i].image = images[i];

        colorAttachmentView.image = buffers[i].image;

        buffers[i].view = device.createImageView(colorAttachmentView);
    }
}

void SwapChain::acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t &imageIndex)
{
    auto resultValue = device.acquireNextImageKHR(swapChain, UINT64_MAX, presentCompleteSemaphore, vk::Fence());
    imageIndex = resultValue.value;
}

void SwapChain::queuePresent(vk::Queue queue, uint32_t imageIndex, vk::Semaphore waitSemaphore)
{
    vk::PresentInfoKHR presentInfo;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapChain;
    presentInfo.pImageIndices = &imageIndex;
    if (waitSemaphore)
    {
        presentInfo.pWaitSemaphores = &waitSemaphore;
        presentInfo.waitSemaphoreCount = 1;
    }
    queue.presentKHR(presentInfo);
}



}
}
