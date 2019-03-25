/*
 * Class wrapping access to the swap chain
 *
 * A swap chain is a collection of framebuffers used for rendering and presentation to the windowing system
 *
 * Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once

#include "saiga/vulkan/svulkan.h"


struct SAIGA_VULKAN_API SwapChainBuffer
{
    VkImage image;
    VkImageView view;
};

class SAIGA_VULKAN_API VulkanSwapChain
{
   private:
    VkInstance instance;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkSurfaceKHR surface;

   public:
    VkFormat colorFormat;
    VkColorSpaceKHR colorSpace;
    /** @brief Handle to the current swap chain, required for recreation */
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    uint32_t imageCount      = 0;
    std::vector<VkImage> images;
    std::vector<SwapChainBuffer> buffers;
    /** @brief Queue family index of the detected graphics and presenting device queue */
    uint32_t queueNodeIndex = UINT32_MAX;

    ~VulkanSwapChain()
    {
        cleanup();
        destroySurface();
    }
    void initSurface(VkSurfaceKHR _surface);

    /**
     * Set instance, physical and logical device to use for the swapchain and get all required function pointers
     *
     * @param instance Vulkan instance to use
     * @param physicalDevice Physical device used to query properties and formats relevant to the swapchain
     * @param device Logical representation of the device to create the swapchain for
     *
     */
    void connect(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device);

    /**
     * Create the swapchain and get it's images with given width and height
     *
     * @param width Pointer to the width of the swapchain (may be adjusted to fit the requirements of the swapchain)
     * @param height Pointer to the height of the swapchain (may be adjusted to fit the requirements of the swapchain)
     * @param vsync (Optional) Can be used to force vsync'd rendering (by using VK_PRESENT_MODE_FIFO_KHR as presentation
     * mode)
     */
    void create(int* width, int* height, bool vsync = false);

    /**
     * Acquires the next image in the swap chain
     *
     * @param presentCompleteSemaphore (Optional) Semaphore that is signaled when the image is ready for use
     * @param imageIndex Pointer to the image index that will be increased if the next image could be acquired
     *
     * @note The function will always wait until the next image has been acquired by setting timeout to UINT64_MAX
     *
     * @return VkResult of the image acquisition
     */
    VkResult acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t* imageIndex);

    /**
     * Queue an image for presentation
     *
     * @param queue Presentation queue for presenting the image
     * @param imageIndex Index of the swapchain image to queue for presentation
     * @param waitSemaphore (Optional) Semaphore that is waited on before the image is presented (only used if !=
     * VK_NULL_HANDLE)
     *
     * @return VkResult of the queue presentation
     */
    VkResult queuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore = VK_NULL_HANDLE);


    /**
     * Destroy and free Vulkan resources used for the swapchain
     */
    void cleanup();
    void destroySurface();
};
