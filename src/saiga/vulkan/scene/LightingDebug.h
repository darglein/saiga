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
    // cancelled: the caller's flag (set when it gives up; the callback is then not called, and a request with only
    // cancelled callbacks is not converted)
    void request(LightingImage image, const PreviewMapping& mapping, CaptureCallback callback = nullptr,
                 bool withImage = true, std::shared_ptr<const bool> cancelled = nullptr);
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
    // Removes the cancelled callbacks; drops a request whose callbacks were all cancelled (the next waiting one moves
    // up). RenderScene calls it before it prepares the frame's sources (a source of a layer, slice or tile follows the
    // mapping recordPreviews converts; CodeRabbit #86); recordPreviews calls it too
    void pruneCancelled();
    // The source mip of this frame's conversion of image (its request's or show's PreviewMapping::mip)
    int sourceMip(LightingImage image) const { return images[size_t(image)].mapping.mip; }
    // The source layer or channel (PreviewMapping::layer) the same way
    int sourceLayer(LightingImage image) const { return images[size_t(image)].mapping.layer; }
    // The whole mapping of this frame's conversion (the cluster slice, the atlas tile's light and face)
    const PreviewMapping& sourceMapping(LightingImage image) const { return images[size_t(image)].mapping; }
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
        // S36a2: a layer (or with layers > 1 that many layers side by side) of a 2D array, step source texels per
        // preview texel (kind 2: the minimum of the first channel over them; 3: the first, 4: the second channel)
        struct ArraySource
        {
            Texture2DArray* texture = nullptr;
            uint32_t layer = 0, layers = 1, step = 1;
            vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal;
            // S36b: a sub-rectangle of the layer (size 0: the whole layer; the atlas's tiles)
            uint32_t originX = 0, originY = 0, width = 0, height = 0;
        };
        std::array<ArraySource, size_t(LightingImage::Count)> array{};
        void set(LightingImage i, Texture2D* texture, uint32_t valueKind, float valueScale = 1)
        {
            image[size_t(i)] = texture;
            kind[size_t(i)]  = valueKind;
            scale[size_t(i)] = valueScale;
        }
        void setArray(LightingImage i, Texture2DArray* texture, uint32_t valueKind, uint32_t layer, uint32_t layers,
                      uint32_t step, vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal)
        {
            array[size_t(i)] =
                ArraySource{texture, layer, std::max(layers, 1u), std::max(step, 1u), layout, 0, 0, 0, 0};
            kind[size_t(i)]  = valueKind;
            scale[size_t(i)] = 1;
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
    // A request's callback: withImage, and the caller's cancellation (round 2, finding 13: true when the caller gave
    // up, e.g. capture_lighting's deadline; a request whose callbacks are all cancelled is not converted)
    struct PendingCallback
    {
        CaptureCallback callback;
        bool withImage = true;
        std::shared_ptr<const bool> cancelled;
        bool isCancelled() const { return cancelled && *cancelled; }
    };
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
        std::vector<PendingCallback> callbacks;
        // A request() without a callback (a conversion wanted without an answer): cancelling the callbacks keeps it
        // (CodeRabbit #86)
        bool demand = false;
        // The requests with another mapping than this frame's (the next frames, in order)
        struct Queued
        {
            PreviewMapping mapping;
            std::vector<PendingCallback> callbacks;
            bool demand = false;
        };
        std::deque<Queued> queued;
        // The callbacks of the recorded frame (answered in finishFrame)
        std::array<std::vector<PendingCallback>, kSlots> recorded;
    };
    VulkanBase* base = nullptr;
    ComputePipeline pipeline;
    // The pipeline, its sets, the sampler and the dummies: created by the first recordPreviews with work
    bool resourcesCreated = false;
    void createResources(vk::CommandBuffer cmd);
    Sampler nearestSampler;
    // The unused source binding's dummies (a 2D source or a 2D array source)
    Texture2D dummy2D;
    Texture2DArray dummyArray;
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
