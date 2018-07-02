/*
* Vulkan Example base class
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "saiga/util/assert.h"

#include <iostream>
#include <chrono>
#include <sys/stat.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <string>
#include <array>
#include <numeric>

#include "vulkan/vulkan.h"

#include "keycodes.hpp"
#include "VulkanTools.h"
#include "VulkanDebug.h"

#include "VulkanInitializers.hpp"
#include "VulkanDevice.hpp"
#include "VulkanSwapChain.hpp"
#include "camera.hpp"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/base/VulkanImgui.h"

#include "saiga/vulkan/SDLWindow.h"

class SAIGA_GLOBAL RenderThing
{
public:
    virtual void update() = 0;
    virtual void render(VkCommandBuffer cmd) = 0;
    virtual void renderGUI() = 0;
};

class SAIGA_GLOBAL VulkanForwardRenderer
{
private:
//    virtual std::vector<const char*> getRequiredInstanceExtensions() = 0;
//    virtual void setupWindow() = 0;
//    virtual void createSurface(VkInstance instance, VkSurfaceKHR* surface) = 0;
public:
Saiga::Vulkan::SDLWindow& window;
    std::shared_ptr<Saiga::Vulkan::ImGuiVulkanRenderer> imGui;
    RenderThing* thing = nullptr;

    Saiga::Vulkan::Instance instance;
    // Physical device (GPU) that Vulkan will ise
    VkPhysicalDevice physicalDevice;
    // Stores physical device properties (for e.g. checking device limits)
    VkPhysicalDeviceProperties deviceProperties;
    // Stores the features available on the selected physical device (for e.g. checking if a feature is available)
    VkPhysicalDeviceFeatures deviceFeatures;
    // Stores all available memory (type) properties for the physical device
    VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
    /**
    * Set of physical device features to be enabled for this example (must be set in the derived constructor)
    *
    * @note By default no phyiscal device features are enabled
    */
    VkPhysicalDeviceFeatures enabledFeatures{};
    /** @brief Set of device extensions to be enabled for this example (must be set in the derived constructor) */
    std::vector<const char*> enabledDeviceExtensions;
    std::vector<const char*> enabledInstanceExtensions;
    /** @brief Logical device, application's view of the physical device (GPU) */
    // todo: getter? should always point to VulkanDevice->device
    VkDevice device;
    // Handle to the device graphics queue that command buffers are submitted to
    VkQueue queue;
    // Depth buffer format (selected during Vulkan initialization)
    VkFormat depthFormat;
    // Command buffer pool
    VkCommandPool cmdPool;
    /** @brief Pipeline stages used to wait at for graphics queue submissions */
    VkPipelineStageFlags submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // Contains command buffers and semaphores to be presented to the queue
    VkSubmitInfo submitInfo;
    // Command buffers used for rendering
    std::vector<VkCommandBuffer> drawCmdBuffers;
    // Global render pass for frame buffer writes
    VkRenderPass renderPass;
    // List of available frame buffers (same as number of swap chain images)
    std::vector<VkFramebuffer>frameBuffers;
    // Active frame buffer index
    uint32_t currentBuffer = 0;
    // Descriptor set pool
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    // List of shader modules created (stored for cleanup)
    //	std::vector<VkShaderModule> shaderModules;
    // Pipeline cache object
    VkPipelineCache pipelineCache;
    // Wraps the swap chain to present images (framebuffers) to the windowing system
    VulkanSwapChain swapChain;
    // Synchronization semaphores
    struct {
        // Swap chain image presentation
        VkSemaphore presentComplete;
        // Command buffer submission and execution
        VkSemaphore renderComplete;
        // UI overlay submission and execution
        VkSemaphore overlayComplete;
    } semaphores;
    std::vector<VkFence> waitFences;
public:
    uint32_t width = 1280;
    uint32_t height = 720;


    /** @brief Encapsulated physical and logical vulkan device */
    vks::VulkanDevice *vulkanDevice;

    /** @brief Example settings that can be changed e.g. by command line arguments */
    struct Settings {
        /** @brief Activates validation layers (and message output) when set to true */
        bool validation = false;
        /** @brief Set to true if fullscreen mode has been requested via command line */
        bool fullscreen = false;
        /** @brief Set to true if v-sync will be forced for the swapchain */
        bool vsync = false;
        /** @brief Enable UI overlay */
        bool overlay = false;
    } settings;

    VkClearColorValue defaultClearColor = { { 0.025f, 0.025f, 0.025f, 1.0f } };


    static std::vector<const char*> args;

    struct
    {
        VkImage image;
        VkDeviceMemory mem;
        VkImageView view;
    } depthStencil;

    struct {
        glm::vec2 axisLeft = glm::vec2(0.0f);
        glm::vec2 axisRight = glm::vec2(0.0f);
    } gamePadState;

    struct {
        bool left = false;
        bool right = false;
        bool middle = false;
    } mouseButtons;

    bool quit = false;


    // Default ctor
    VulkanForwardRenderer(Saiga::Vulkan::SDLWindow& window, bool enableValidation = true);

    // dtor
    virtual ~VulkanForwardRenderer();

    // Setup the vulkan instance, enable required extensions and connect to the physical device (GPU)
    bool initVulkan();

    virtual void update() {};
    virtual void render(VkCommandBuffer cmd) {};
    virtual void renderGUI() {};

    void updateIntern();
    void renderIntern();

    // Pure virtual function to be overriden by the dervice class
    // Called in case of an event where e.g. the framebuffer has to be rebuild and thus
    // all command buffers that may reference this
    virtual void buildCommandBuffers();

    void createSynchronizationPrimitives();

    // Creates a new (graphics) command pool object storing command buffers
    void createCommandPool();
    // Setup default depth and stencil views
    virtual void setupDepthStencil();
    // Create framebuffers for all requested swap chain images
    // Can be overriden in derived class to setup a custom framebuffer (e.g. for MSAA)
    virtual void setupFrameBuffer();
    // Setup a default render pass
    // Can be overriden in derived class to setup a custom render pass (e.g. for MSAA)
    virtual void setupRenderPass();



    // Connect and prepare the swap chain
    void initSwapchain();
    // Create swap chain images
    void setupSwapChain();

    // Check if command buffers are valid (!= VK_NULL_HANDLE)
    bool checkCommandBuffers();
    // Create command buffers for drawing commands
    void createCommandBuffers();
    // Destroy all command buffers and set their handles to VK_NULL_HANDLE
    // May be necessary during runtime if options are toggled
    void destroyCommandBuffers();

    // Command buffer creation
    // Creates and returns a new command buffer
    VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, bool begin);
    // End the command buffer, submit it to the queue and free (if requested)
    // Note : Waits for the queue to become idle
    void flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free);

    // Create a cache pool for rendering pipelines
    void createPipelineCache();

    // Prepare commonly used Vulkan functions
    virtual void prepare();

    // Start the main render loop
    void renderLoop();

    // Render one frame of a render loop on platforms that sync rendering
    void renderFrame();

    //	void updateOverlay();

    // Prepare the frame for workload submission
    // - Acquires the next image from the swap chain
    // - Sets the default wait and signal semaphores
    void prepareFrame();

    // Submit the frames' workload
    void submitFrame();

};
