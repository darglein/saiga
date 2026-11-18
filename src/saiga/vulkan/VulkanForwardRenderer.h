/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/templatedImage.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"
#include "saiga/vulkan/CommandPool.h"
#include "saiga/vulkan/FrameSync.h"
#include "saiga/vulkan/Queue.h"
#include "saiga/vulkan/Renderer.h"
#include "saiga/vulkan/buffer/DepthBuffer.h"
#include "saiga/vulkan/pipeline/PipelineInfo.h"
#include "saiga/vulkan/window/Window.h"

#include <functional>

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

struct SAIGA_VULKAN_API ForwardRenderingParameters
{
    // MSAA sample count of the main pass (1: off). Clamped to the largest count the device supports for the color and
    // depth attachments. The multisampled image is resolved into the swap chain image at the end of the pass.
    int msaaSamples = 1;

    // Reversed depth: the depth buffer is cleared with 0 and closer fragments have a larger depth, so the pipelines
    // must use eGreater / eGreaterOrEqual. With a float depth buffer this gives an almost uniform precision over the
    // whole view distance, also with an infinite far plane (see reversedZInfinitePerspective). The render modules in
    // renderModules/ assume the default (not reversed) depth.
    bool reversedZ = false;

    // eUndefined: eD32Sfloat with reversedZ, otherwise eD16Unorm.
    vk::Format depthFormat = vk::Format::eUndefined;

    // This is blender's default viewport background color :)
    vec4 clearColor = vec4(57, 57, 57, 255) / 255.0f;

    void fromConfigFile(const std::string& file) {}
};

/**
 * Projection matrix for Vulkan's clip space (y down, depth in [0, 1]) with reversed depth and an infinite far plane:
 * depth = zNear / (distance along the view direction). View space as in OpenGL (the camera looks along -z, y is up).
 * fovy is the vertical field of view in radians.
 */
SAIGA_VULKAN_API mat4 reversedZInfinitePerspective(float fovy, float aspect, float zNear);

/**
 * Renders into the swap chain image with a depth buffer, using dynamic rendering (Vulkan 1.3).
 * With a HeadlessWindow it renders into offscreen images instead (see VulkanRenderer::isHeadless); the result is
 * read with requestScreenshot.
 */
class SAIGA_VULKAN_API VulkanForwardRenderer : public VulkanRenderer
{
   public:
    using InterfaceType = VulkanForwardRenderingInterface;
    using ParameterType = ForwardRenderingParameters;

    CommandPool renderCommandPool;

    VulkanForwardRenderer(VulkanWindow& window, VulkanParameters vulkanParameters,
                          const ForwardRenderingParameters& parameters = ForwardRenderingParameters());
    virtual ~VulkanForwardRenderer() override;

    virtual void render(const FrameInfo& frame) override;

    virtual void createBuffers(int numImages, int w, int h) override;

    // The attachment formats (and sample count) of the main pass. Create the pipelines used in
    // VulkanForwardRenderingInterface::render with these.
    RenderingFormats renderingFormats() const;

    const ForwardRenderingParameters& parameters() const { return params; }
    // The MSAA sample count actually used (ForwardRenderingParameters::msaaSamples clamped to the device limits).
    vk::SampleCountFlagBits sampleCount() const { return samples; }
    // The depth value the depth buffer is cleared with: 0 with reversed depth, otherwise 1.
    float clearDepth() const { return params.reversedZ ? 0.0f : 1.0f; }

    /**
     * Saves the next rendered frame (including the GUI) to the image file (e.g. png). The file is written at the end
     * of that frame's render call (which then waits for the GPU).
     */
    void requestScreenshot(const std::string& file);

    /**
     * Passes the next rendered frame (including the GUI) to the callback: RGBA with alpha = 255, row 0 is the top
     * row. The callback is called at the end of that frame's render call (which then waits for the GPU). Every
     * requested callback is called exactly once; all requests made before a frame get the same image. If the color
     * images can not be read back (see the error output), the callbacks get an empty image (width 0).
     */
    using ScreenshotCallback = std::function<void(const TemplatedImage<ucvec4>&)>;
    void requestScreenshot(ScreenshotCallback callback) { screenshotCallbacks.push_back(std::move(callback)); }

    /**
     * For automated tests: if the environment variable SAIGA_SCREENSHOT is set to a file name, a screenshot of frame
     * 60 is saved to that file and the window is closed afterwards. disableAutoScreenshot() turns this off (e.g. for
     * benchmark runs) and returns true if it was enabled.
     */
    bool disableAutoScreenshot()
    {
        bool was = !autoScreenshotFile.empty();
        autoScreenshotFile.clear();
        return was;
    }

   protected:
    ForwardRenderingParameters params;
    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
    vk::Format depthFormat          = vk::Format::eD16Unorm;

    // Shared by all frames in flight. With MSAA the depth buffer is multisampled and the color is rendered into
    // msaaColor, which is resolved into the swap chain image.
    Texture2D depthImage;
    Texture2D msaaColor;

    // One per frame in flight (FrameInfo::frameIndex)
    std::vector<vk::CommandBuffer> drawCmdBuffers;

    std::vector<ScreenshotCallback> screenshotCallbacks;
    std::string autoScreenshotFile;
    int renderedFrames = 0;
    // Records the copy of the color image into a host visible buffer. Returns the buffer.
    std::unique_ptr<StagingBuffer> recordScreenshot(vk::CommandBuffer cmd, vk::Image colorImage);
    TemplatedImage<ucvec4> readScreenshot(StagingBuffer& buffer);
};

}  // namespace Vulkan
}  // namespace Saiga
