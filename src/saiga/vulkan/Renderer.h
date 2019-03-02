/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

//#include "saiga/vulkan/memory/ChunkAllocator.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/Parameters.h"
#include "saiga/vulkan/SwapChain.h"
#include "saiga/vulkan/svulkan.h"

namespace Saiga
{
namespace Vulkan
{
class VulkanWindow;

/**
 * Base class for all Vulkan renderers.
 * This already includes basic functionality that every renderer needs:
 * - Window management
 * - Swap Chain
 * - Instance and Device
 */
class SAIGA_VULKAN_API VulkanRenderer : public RendererBase
{
   public:
    Saiga::Vulkan::VulkanBase base;

    VulkanRenderer(VulkanWindow& window, VulkanParameters vulkanParameters);
    virtual ~VulkanRenderer() override;

    virtual void render(Camera*) override {}
    virtual void bindCamera(Camera*) override {}

    virtual float getTotalRenderTime() override;

    void renderImGui(bool* p_open) override;


    void createSwapChain();
    void resizeSwapChain();

   protected:
    /**
     * Shared Member variables common for all vulkan render engines.
     */

    Saiga::Vulkan::VulkanWindow& window;
    VkSurfaceKHR surface;

    uint32_t width  = 1280;
    uint32_t height = 720;

    Saiga::Vulkan::Instance instance;


    std::vector<const char*> enabledDeviceExtensions;
    std::vector<const char*> enabledInstanceExtensions;
    VulkanSwapChain swapChain;
    VulkanParameters vulkanParameters;

   private:
    void initInstanceDevice();
};



}  // namespace Vulkan
}  // namespace Saiga
