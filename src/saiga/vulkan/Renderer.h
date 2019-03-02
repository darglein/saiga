/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

//#include "saiga/vulkan/memory/ChunkAllocator.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/FrameSync.h"
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
    VulkanRenderer(VulkanWindow& window, VulkanParameters vulkanParameters);
    virtual ~VulkanRenderer() override;

    virtual void createFrameBuffers(int numImages, int w, int h) = 0;
    virtual void createDepthBuffer(int w, int h)                 = 0;
    virtual void setupRenderPass()                               = 0;

    virtual void render2(FrameSync& sync, int currentImage) = 0;
    virtual void render(Camera*) override;
    virtual void bindCamera(Camera*) override {}

    virtual float getTotalRenderTime() override;

    void renderImGui(bool* p_open) override;


    void createSwapChain();
    void resizeSwapChain();

    void waitIdle();

    int swapChainSize() { return swapChain.imageCount; }

    inline VulkanBase& base() { return vulkanBase; }

   protected:
    VulkanBase vulkanBase;
    /**
     * Shared Member variables common for all vulkan render engines.
     */

    Saiga::Vulkan::VulkanWindow& window;
    VkSurfaceKHR surface;

    /**
     * Size of the render surface.
     * This might be different to the window size, because of
     * the border.
     *
     * Use these dimensions for your framebuffers!
     */
    int surfaceWidth  = 1280;
    int SurfaceHeight = 720;

    Saiga::Vulkan::Instance instance;


    std::vector<const char*> enabledDeviceExtensions;
    std::vector<const char*> enabledInstanceExtensions;
    VulkanSwapChain swapChain;
    VulkanParameters vulkanParameters;

   private:
    uint32_t currentBuffer      = 0;
    unsigned int nextSyncObject = 0;
    std::vector<FrameSync> syncObjects;

    void initInstanceDevice();


    bool valid       = true;
    int validCounter = 0;
};



}  // namespace Vulkan
}  // namespace Saiga
