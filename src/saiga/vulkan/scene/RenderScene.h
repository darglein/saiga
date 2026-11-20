/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/managedImage.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
class VulkanWindow;

/**
 * The frame of the engine app (docs/engine_design.md §8.3, §10.1; S13: the frame shell): the swap chain of the window
 * (or offscreen images for a HeadlessWindow), two frames in flight on the frame timeline, one MSAA color + reversed-Z
 * depth pass (infinite far plane) with resolve, ImGui on top, screenshots, GPU timestamps. Until S17 replaces the frame
 * with the HDR frame, the shell is the forward renderer's frame (VulkanForwardRenderer) driven by renderFrame, so the
 * prototype renderers (WorldRenderer, the demos' MeshRenderer) keep their init(VulkanForwardRenderer&).
 */
class SAIGA_VULKAN_API RenderScene : private VulkanForwardRenderingInterface
{
   public:
    RenderScene(VulkanWindow& window, const VulkanParameters& vulkanParameters,
                const RenderSceneParameters& parameters);
    ~RenderScene() override;
    RenderScene(const RenderScene&)            = delete;
    RenderScene& operator=(const RenderScene&) = delete;

    using Callback = std::function<void(vk::CommandBuffer)>;
    // Records and submits one frame: transfer (uploads), render (inside the pass), gui (ImGui calls). False if no
    // frame was submitted (a minimized window, an out-of-date swap chain that is recreated)
    bool renderFrame(const Callback& transfer, const Callback& render, const std::function<void()>& gui);

    RenderingFormats formats() const { return renderer->renderingFormats(); }
    vk::Format colorFormat() const { return renderer->getColorFormat(); }
    VulkanBase& base() { return renderer->base(); }
    VulkanForwardRenderer& forwardRenderer() { return *renderer; }
    bool headless() const { return renderer->isHeadless(); }
    // The size of the rendered images (0 before the first frame)
    ivec2 size() const { return renderer->surfaceSize(); }
    uint64_t frames() const { return frameCount; }

    // The next rendered frame (with ImGui) as RGBA, alpha 255
    void requestScreenshot(VulkanForwardRenderer::ScreenshotCallback callback);
    void setImguiVisible(bool visible) { renderer->should_render_imgui = visible; }
    bool imguiVisible() const { return renderer->should_render_imgui; }

    // The GPU times of the oldest completed frame not yet taken (2-3 frames after it was rendered); false if none
    bool takeGpuFrameTimes(FrameTimes& out);
    // The sections of FrameTimes::gpuSectionMs
    static const std::vector<const char*>& gpuSectionNames();

    // Waits until the GPU has finished every frame (before resources used by frames are destroyed)
    void waitIdle() { renderer->waitIdle(); }

   private:
    void transfer(vk::CommandBuffer cmd) override;
    void render(vk::CommandBuffer cmd) override;
    void renderGUI() override;

    std::unique_ptr<VulkanForwardRenderer> renderer;
    const Callback* transferFn         = nullptr;
    const Callback* renderFn           = nullptr;
    const std::function<void()>* guiFn = nullptr;
    uint64_t frameCount                = 0;
    // The frame numbers of the submitted frames by timeline value (GPU times arrive by timeline value)
    std::deque<std::pair<uint64_t, uint64_t>> submittedFrames;
};

}  // namespace Vulkan
}  // namespace Saiga
