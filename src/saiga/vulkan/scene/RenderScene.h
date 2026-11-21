/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/managedImage.h"
#include "saiga/vulkan/Renderer.h"
#include "saiga/vulkan/pipeline/PipelineInfo.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
class VulkanWindow;
class SceneFrameRenderer;

/**
 * The frame of the engine app (docs/engine_design.md §8.3, §10.1, §10.5; S13: the frame shell, S17: the HDR frame):
 * the swap chain of the window (or offscreen images for a HeadlessWindow), two frames in flight on the frame timeline,
 * GPU timestamps, screenshots. Per frame:
 *   1. transfer (uploads), outside of any pass;
 *   2. the scene pass: HDR color (R16G16B16A16 float, msaaSamples) + reversed-Z depth (D32, infinite far plane),
 *      cleared to the clear color and depth 0; the render callback draws here (pipelines: formats());
 *   3. the MSAA resolve: Identity (clay mode) with the hardware average of the scene pass, the other tone maps with a
 *      tone-map weighted custom resolve (bright edges do not alias); none with 1 sample;
 *   4. post (exposure, tone map, output transfer) and ImGui on top, single-sampled, into the swap chain image.
 *
 * Pre-exposure contract: while sceneWritesLinear() every scene shader writes linear radiance multiplied by
 * preExposure() (negatives clamped, no clamp to 1); otherwise display values clamped to [0, 1] (clay mode).
 */
class SAIGA_VULKAN_API RenderScene
{
   public:
    RenderScene(VulkanWindow& window, const VulkanParameters& vulkanParameters,
                const RenderSceneParameters& parameters);
    ~RenderScene();
    RenderScene(const RenderScene&)            = delete;
    RenderScene& operator=(const RenderScene&) = delete;

    using Callback = std::function<void(vk::CommandBuffer)>;
    // Records and submits one frame: transfer (uploads), render (inside the scene pass), gui (ImGui calls). False if no
    // frame was submitted (a minimized window, an out-of-date swap chain that is recreated). The tone map, exposure and
    // pre-exposure are latched when the frame's recording starts, before any callback (not without a frame).
    bool renderFrame(const Callback& transfer, const Callback& render, const std::function<void()>& gui);

    // The scene pass (the HDR color format, D32 depth, the sample count): the game's pipelines use it
    RenderingFormats formats() const;
    // The output format (swap chain or offscreen image)
    vk::Format colorFormat() const;
    VulkanBase& base();
    // CPU time of the last frame's command recording and submit (milliseconds)
    double lastRecordingMs() const;
    bool headless() const;
    // The size of the rendered images (0 before the first frame)
    ivec2 size() const;
    uint64_t frames() const { return frameCount; }

    // The next rendered frame (tone mapped, with ImGui) as RGBA, alpha 255
    void requestScreenshot(VulkanRenderer::ScreenshotCallback callback);
    void setImguiVisible(bool visible);
    bool imguiVisible() const;
    // Draws ImGui (true) or not (false) in the rendered frames until cleared, without changing imguiVisible (the
    // user's choice, e.g. a key that toggles it); for screenshots with or without the GUI
    void setImguiOverride(std::optional<bool> visible) { imguiOverride = visible; }
    std::optional<bool> imguiOverridden() const { return imguiOverride; }

    // From the next frame on (R-MECH-1). The exposure is clamped to [1/1024, 1024]
    void setToneMap(ToneMap toneMap);
    void setExposure(float exposure);
    // The requested values (the next frame's)
    const RenderSceneParameters& parameters() const { return params; }
    // The latched values of the frame being recorded (the last frame between frames)
    ToneMap toneMap() const { return frameToneMap; }
    // The effective exposure (1 with Identity)
    float exposure() const { return frameExposure; }
    // toneMap() != Identity: the scene shaders write linear values multiplied by preExposure()
    bool sceneWritesLinear() const { return frameToneMap != ToneMap::Identity; }
    // The exposure the scene pass of this frame writes with: the previous frame's effective exposure (1 after
    // Identity), or this frame's if it changed by more than a factor of kMaxPreExposureChange (fp16 range)
    float preExposure() const { return framePreExposure; }
    static constexpr float kMaxPreExposureChange = 16;

    // The GPU times of the oldest completed frame not yet taken (2-3 frames after it was rendered); false if none
    bool takeGpuFrameTimes(FrameTimes& out);
    // The sections of FrameTimes::gpuSectionMs in frame order: transfer, scene, resolve, post, imgui
    static const std::vector<const char*>& gpuSectionNames();

    // Waits until the GPU has finished every frame (before resources used by frames are destroyed)
    void waitIdle();

   private:
    friend class SceneFrameRenderer;
    void latchFrame();
    RenderSceneParameters params;
    ToneMap frameToneMap   = ToneMap::Identity;
    float frameExposure    = 1;
    float framePreExposure = 1;
    // The effective exposure of the last submitted frame (the next frame's pre-exposure)
    float lastExposure = 1;

    std::unique_ptr<SceneFrameRenderer> renderer;
    const Callback* transferFn         = nullptr;
    const Callback* renderFn           = nullptr;
    const std::function<void()>* guiFn = nullptr;
    uint64_t frameCount                = 0;
    std::optional<bool> imguiOverride;
    // The frame numbers of the submitted frames by timeline value (GPU times arrive by timeline value)
    std::deque<std::pair<uint64_t, uint64_t>> submittedFrames;
};

}  // namespace Vulkan
}  // namespace Saiga
