/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/RenderScene.h"

#include "saiga/vulkan/window/Window.h"

namespace Saiga
{
namespace Vulkan
{
RenderScene::RenderScene(VulkanWindow& window, const VulkanParameters& vulkanParameters,
                         const RenderSceneParameters& parameters)
{
    VulkanParameters vp = vulkanParameters;
    vp.enableImgui      = parameters.imgui;
    ForwardRenderingParameters fp;
    fp.msaaSamples = parameters.msaaSamples;
    fp.reversedZ   = true;
    fp.clearColor  = parameters.clearColor;
    renderer       = std::make_unique<VulkanForwardRenderer>(window, vp, fp);
    // The app decides about screenshots (--screenshot, SAIGA_SCREENSHOT), not the renderer
    renderer->disableAutoScreenshot();
    renderer->setRenderObject(*this);
    renderer->should_render_imgui = parameters.imgui;
}

RenderScene::~RenderScene()
{
    if (renderer) renderer->waitIdle();
}

bool RenderScene::renderFrame(const Callback& transferCb, const Callback& renderCb, const std::function<void()>& gui)
{
    transferFn              = &transferCb;
    renderFn                = &renderCb;
    guiFn                   = &gui;
    const uint64_t before   = renderer->submittedFrames();
    const uint64_t timeline = renderer->base().currentFrameSignalValue();
    RenderInfo info;
    static_cast<VulkanRenderer&>(*renderer).render(info);
    transferFn = nullptr;
    renderFn   = nullptr;
    guiFn      = nullptr;
    if (renderer->submittedFrames() == before) return false;
    frameCount++;
    submittedFrames.push_back({timeline, frameCount});
    if (submittedFrames.size() > 32) submittedFrames.pop_front();
    return true;
}

double RenderScene::lastRecordingMs() const
{
    return renderer->lastRecordingMs();
}

void RenderScene::transfer(vk::CommandBuffer cmd)
{
    if (transferFn && *transferFn) (*transferFn)(cmd);
}

void RenderScene::render(vk::CommandBuffer cmd)
{
    if (renderFn && *renderFn) (*renderFn)(cmd);
}

void RenderScene::renderGUI()
{
    if (guiFn && *guiFn) (*guiFn)();
}

void RenderScene::requestScreenshot(VulkanForwardRenderer::ScreenshotCallback callback)
{
    renderer->requestScreenshot(std::move(callback));
}

const std::vector<const char*>& RenderScene::gpuSectionNames()
{
    // The sections of VulkanRenderer's frame timings
    static const std::vector<const char*> names = {"transfer", "main", "imgui"};
    return names;
}

bool RenderScene::takeGpuFrameTimes(FrameTimes& out)
{
    FrameTimings<>::CompletedFrame c;
    while (renderer->takeCompletedFrame(c))
    {
        // The frame number by its timeline value (frames that were not submitted by renderFrame are skipped)
        while (!submittedFrames.empty() && submittedFrames.front().first < c.timelineValue) submittedFrames.pop_front();
        if (submittedFrames.empty() || submittedFrames.front().first != c.timelineValue) continue;
        out.frame        = submittedFrames.front().second;
        out.gpuMs        = c.frameMs;
        out.gpuSectionMs = c.sectionMs;
        submittedFrames.pop_front();
        return true;
    }
    return false;
}

}  // namespace Vulkan
}  // namespace Saiga
