/**
 * Copyright (c) 2021 Darius Rückert
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

#include "FrameTimings.h"
namespace Saiga
{
namespace Vulkan
{
class VulkanWindow;
class ImGuiVulkanRenderer;
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
    /**
     * - Create Instance
     * - Create Device
     * - Create Surface
     */
    VulkanRenderer(VulkanWindow& window, VulkanParameters vulkanParameters);
    virtual ~VulkanRenderer() override;

    /**
     * - Create Swapchain
     * - Create SyncObjects
     * - Create Derived Render Objects
     *      - DepthBuffer
     *      - FrameBuffer
     *
     * All these objects depend on the swapchain. Either by the size or the number of
     * images.
     *
     * -> This function is called every time the window resizes or the swapchain has to
     * be recreated.
     */
    void init();


    /**
     * Must be overriden by the actual renderers.
     *
     * Should create (if required):
     *  - Depthbuffer/Stencilbuffer
     *  - Framebuffers
     *  - Main Render Command buffers
     *
     * Important:
     * This function can be called multiple times (for example when window size changes).
     * -> Make sure no memory leaks when creating the buffers.
     *
     */
    virtual void createBuffers(int numImages, int w, int h) = 0;

    /**
     * The main render functions for the derived renderers.
     * This should then call all required subpasses.
     */
    virtual void render(FrameSync& sync, int currentImage) = 0;



    virtual void render(const RenderInfo& renderInfo) override;


    void renderImgui() override;
    void waitIdle();
    int swapChainSize() { return swapChain.imageCount; }
    inline VulkanBase& base() { return vulkanBase; }


   protected:
    /**
     * Size of the render surface.
     * This might be different to the window size, because of
     * the border.
     *
     * Use these dimensions for your framebuffers!
     */
    int surfaceWidth  = 1280;
    int SurfaceHeight = 720;

    enum class State
    {
        // Initial State
        UNINITIALIZED,
        // After Constructor
        INITIALIZED,
        // After init()
        RENDERABLE,
        // Happens at resize window or change swapchain settings
        RESET
    };
    State state = State::UNINITIALIZED;


    Instance instance;
    VulkanBase vulkanBase;
    FrameTimings<> timings;
    VulkanWindow& window;
    VkSurfaceKHR surface;

    std::vector<const char*> enabledDeviceExtensions;
    std::vector<const char*> enabledInstanceExtensions;
    VulkanSwapChain swapChain;
    VulkanParameters vulkanParameters;

    std::unique_ptr<ImGuiVulkanRenderer> imGui;

   private:
    uint32_t currentBuffer      = 0;
    unsigned int nextSyncObject = 0;
    std::vector<FrameSync> syncObjects;
    void reset();
};



}  // namespace Vulkan
}  // namespace Saiga
