/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/templatedImage.h"
#include "saiga/vulkan/Renderer.h"
#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/scene/LightingTypes.h"
#include "saiga/vulkan/texture/Sampler.h"
#include "saiga/vulkan/texture/Texture.h"

#include <array>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
/**
 * The preview images and pixel readbacks of S36's lighting debug core (A8 of docs/stages/S36_lighting.plan.md;
 * RenderScene owns it). Depth and data images are not displayable, so on demand a compute pass
 * (shader/vulkan/scene/lighting/debug_preview.comp) converts a LightingImage into an RGBA8 preview texture (the
 * PreviewMapping: range, log2, colour map) and reduces its values (min, max, mean, 16-bin histogram). A preview is
 * computed only in a frame in which it was requested (capture_lighting, probe_lighting: request(), at the source's
 * resolution) or shown (the Lighting tab draws it: show(), at most 512 texels per side, at most once per
 * previewInterval() per image unless its mapping changes, and at most 2^20 preview texels per frame (setPreviewLimits):
 * the others wait for a later frame, the oldest first): without a request nothing is dispatched. The preview textures
 * are read by ImGui in the same frame (the conversion runs before the post + ImGui pass) and stay in
 * eShaderReadOnlyOptimal; their statistics arrive one frame slot later (statistics()) or, for a capture, after the
 * frame (blocking, like a screenshot).
 *
 * Pixel readbacks (probe_lighting): the resolved fp16 colour (the MSAA average) and the sampled depth of pixels, copied
 * after the resolve, read after the frame (blocking).
 */
class SAIGA_VULKAN_API LightingDebug
{
   public:
    struct Capture
    {
        TemplatedImage<ucvec4> image;
        PreviewStatistics statistics;
        ivec2 size = ivec2(0, 0);
    };
    using CaptureCallback = std::function<void(const Capture&)>;

    struct PixelValue
    {
        ivec2 pixel = ivec2(0, 0);
        vec4 color  = vec4(0, 0, 0, 0);
        float depth = 0;  // reversed-Z (1 near, 0 far)
        bool valid  = false;
    };
    using PixelCallback = std::function<void(const std::vector<PixelValue>&)>;

    void init(VulkanBase& base);
    void destroy();
    ~LightingDebug() { destroy(); }

    // The preview of image in the next recorded frame with the mapping; withImage: the capture downloads the preview
    // (else only the statistics). A request without a callback only computes the preview and its statistics. Requests
    // of one image with another mapping than the frame's first request wait for the next frames (one conversion per
    // image and frame), so each callback gets its own mapping's preview and statistics
    void request(LightingImage image, const PreviewMapping& mapping, CaptureCallback callback = nullptr,
                 bool withImage = true);
    // The tab shows image this frame (rate-limited and at preview resolution, see the class comment)
    void show(LightingImage image, const PreviewMapping& mapping);
    bool requested() const;
    // The minimum time between two conversions of a shown image (seconds; 0: every frame)
    void setPreviewInterval(double seconds) { previewIntervalSeconds = seconds; }
    double previewInterval() const { return previewIntervalSeconds; }
    // A shown preview's longest side (texels) and the preview texels a frame converts for shown images (a show() beyond
    // it waits; the first conversion of a frame is always made)
    void setPreviewLimits(uint32_t maxSize, uint64_t texelBudget)
    {
        previewMaxSize     = std::max(maxSize, 1u);
        previewTexelBudget = texelBudget;
    }
    static constexpr uint32_t kPreviewMaxSize     = 512;
    static constexpr uint64_t kPreviewTexelBudget = 1u << 20;
    bool requested(LightingImage image) const { return images[size_t(image)].requested; }
    // The pixels of the next recorded frame (in the image; out of range pixels are answered invalid)
    void requestPixels(const std::vector<ivec2>& pixels, PixelCallback callback);
    bool pixelsRequested() const { return !pixelRequests.empty(); }

    // ---- RenderScene's frame
    // The sources of this frame's previews (in eShaderReadOnlyOptimal; a missing one is not converted): per image the
    // texture, the value (kind 0: the luminance of rgb times scale, 1: the first channel) and the scale; returns the
    // number of preview conversions
    struct Sources
    {
        std::array<Texture2D*, size_t(LightingImage::Count)> image{};
        std::array<uint32_t, size_t(LightingImage::Count)> kind{};
        std::array<float, size_t(LightingImage::Count)> scale{};
        void set(LightingImage i, Texture2D* texture, uint32_t valueKind, float valueScale = 1)
        {
            image[size_t(i)] = texture;
            kind[size_t(i)]  = valueKind;
            scale[size_t(i)] = valueScale;
        }
    };
    uint32_t recordPreviews(vk::CommandBuffer cmd, uint32_t frameIndex, const Sources& sources);
    // Records the pixel copies (the resolved colour and the sampled depth are in eTransferSrcOptimal)
    void recordPixels(vk::CommandBuffer cmd, uint32_t frameIndex, vk::Image hdr, vk::Image depth, ivec2 size);
    // After the frame's submit: waits for the frame (only with captures or pixels) and answers them
    void finishFrame(uint32_t frameIndex, const std::function<void()>& waitFrame);
    // The frame slot's previous frame has completed: its statistics become statistics()
    void harvest(uint32_t frameIndex);

    // The preview texture of an image (nullptr before its first request) and a counter that changes when the texture is
    // recreated (the ImGui registration of the tab follows it)
    Texture2D* preview(LightingImage image);
    uint64_t previewVersion(LightingImage image) const { return images[size_t(image)].version; }
    // The newest statistics of an image and their frame (0: none yet)
    const PreviewStatistics& statistics(LightingImage image) const { return images[size_t(image)].stats; }
    uint64_t statisticsFrame(LightingImage image) const { return images[size_t(image)].statsFrame; }
    // The conversions of the last recorded frame; the GPU memory of the previews
    uint32_t lastDispatches() const { return dispatches; }
    uint64_t memoryBytes() const;
    // RenderScene's frame number of the frame being recorded (the statistics' frame)
    void setFrame(uint64_t frame) { currentFrame = frame; }

    // IEEE half -> float
    static float halfToFloat(uint16_t h);

   private:
    // The frame slots (frames in flight)
    static constexpr size_t kSlots = VulkanRenderer::framesInFlight;
    struct Image
    {
        std::unique_ptr<Texture2D> texture;
        uint64_t version = 0;
        // Per frame slot: the statistics buffer and the frame it was requested in (0: none)
        std::array<Buffer, kSlots> statsBuffers;
        std::array<uint64_t, kSlots> pendingFrame{};
        PreviewStatistics stats;
        uint64_t statsFrame = 0;
        // This frame's request (fullResolution: a request(), else a show()) and the time of the last conversion of a
        // show() (steady clock seconds)
        bool requested      = false;
        bool fullResolution = false;
        double lastShown    = -1e30;
        PreviewMapping mapping;
        PreviewMapping shownMapping;
        // Per frame slot: the device-local statistics the shader writes (copied to statsBuffers after the conversion)
        std::array<Buffer, kSlots> gpuStats;
        std::vector<std::pair<CaptureCallback, bool>> callbacks;
        // The requests with another mapping than this frame's (the next frames, in order)
        struct Queued
        {
            PreviewMapping mapping;
            std::vector<std::pair<CaptureCallback, bool>> callbacks;
        };
        std::deque<Queued> queued;
        // The callbacks of the recorded frame (answered in finishFrame)
        std::array<std::vector<std::pair<CaptureCallback, bool>>, kSlots> recorded;
    };
    VulkanBase* base = nullptr;
    ComputePipeline pipeline;
    Sampler nearestSampler;
    Buffer partials;
    std::array<vk::DescriptorSet, size_t(LightingImage::Count) * kSlots> sets{};
    std::array<Image, size_t(LightingImage::Count)> images;
    uint32_t dispatches           = 0;
    uint64_t currentFrame         = 0;
    double previewIntervalSeconds = 0.25;
    uint32_t previewMaxSize       = kPreviewMaxSize;
    uint64_t previewTexelBudget   = kPreviewTexelBudget;

    std::vector<std::pair<std::vector<ivec2>, PixelCallback>> pixelRequests;
    std::array<std::vector<std::pair<std::vector<ivec2>, PixelCallback>>, kSlots> recordedPixels;
    std::array<Buffer, kSlots> pixelBuffers;
    std::array<ivec2, kSlots> pixelSize{};

    static PreviewStatistics readStatistics(Buffer& b);
};

}  // namespace Vulkan
}  // namespace Saiga
