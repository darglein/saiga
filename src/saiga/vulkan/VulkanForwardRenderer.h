/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"
#include "saiga/vulkan/CommandPool.h"
#include "saiga/vulkan/FrameSync.h"
#include "saiga/vulkan/Queue.h"
#include "saiga/vulkan/Renderer.h"
#include "saiga/vulkan/buffer/DepthBuffer.h"
#include "saiga/vulkan/pipeline/PipelineInfo.h"
#include "saiga/vulkan/window/Window.h"

namespace Saiga
{
namespace Vulkan
{
class StagingBuffer;

class SAIGA_VULKAN_API VulkanForwardRenderingInterface : public RenderingInterface
{
   public:
    // Record uploads (e.g. uniform buffer updates). Called outside of the rendering pass.
    using RenderingInterface::render;
    virtual void transfer(vk::CommandBuffer cmd) {}
    // Record the draw calls. Called inside the rendering pass (dynamic rendering) with the color and depth attachment.
    virtual void render(vk::CommandBuffer cmd) {}
    virtual void renderGUI() {}
};

struct SAIGA_VULKAN_API ForwardRenderingParameters{void fromConfigFile(const std::string& file){}};

/**
 * Renders into the swap chain image with a depth buffer, using dynamic rendering (Vulkan 1.3).
 */
class SAIGA_VULKAN_API VulkanForwardRenderer : public VulkanRenderer
{
   public:
    using InterfaceType = VulkanForwardRenderingInterface;
    using ParameterType = ForwardRenderingParameters;

    CommandPool renderCommandPool;

    VulkanForwardRenderer(VulkanWindow& window, VulkanParameters vulkanParameters);
    virtual ~VulkanForwardRenderer() override;

    virtual void render(const FrameInfo& frame) override;

    virtual void createBuffers(int numImages, int w, int h) override;

    // The attachment formats of the main pass. Create the pipelines used in VulkanForwardRenderingInterface::render
    // with these.
    RenderingFormats renderingFormats() const;

    /**
     * Saves the next rendered frame (including the GUI) to the image file (e.g. png). The file is written at the end
     * of that frame's render call (which then waits for the GPU).
     */
    void requestScreenshot(const std::string& file) { screenshotFile = file; }

    /**
     * For automated tests: if the environment variable SAIGA_SCREENSHOT is set to a file name, a screenshot of frame
     * 60 is saved to that file and the window is closed afterwards.
     */

   protected:
    DepthBuffer depthBuffer;

    // One per frame in flight (FrameInfo::frameIndex)
    std::vector<vk::CommandBuffer> drawCmdBuffers;

    std::string screenshotFile;
    std::string autoScreenshotFile;
    int renderedFrames = 0;
    // Records the copy of the color image into a host visible buffer. Returns the buffer.
    std::unique_ptr<StagingBuffer> recordScreenshot(vk::CommandBuffer cmd, vk::Image colorImage);
    void saveScreenshot(StagingBuffer& buffer, const std::string& file);
};

}  // namespace Vulkan
}  // namespace Saiga
