/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/RenderScene.h"

#include "saiga/vulkan/buffer/StagingBuffer.h"
#include "saiga/vulkan/imgui/ImGuiVulkanRenderer.h"
#include "saiga/vulkan/pipeline/DescriptorSet.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/scene/Atmosphere.h"
#include "saiga/vulkan/scene/BrdfLut.h"
#include "saiga/vulkan/scene/CascadedShadows.h"
#include "saiga/vulkan/scene/CloudLayer.h"
#include "saiga/vulkan/scene/DebugLines.h"
#include "saiga/vulkan/scene/EnvironmentMap.h"
#include "saiga/vulkan/scene/GrassPass.h"
#include "saiga/vulkan/scene/HorizonShadows.h"
#include "saiga/vulkan/scene/InstanceBuffer.h"
#include "saiga/vulkan/scene/LightClusters.h"
#include "saiga/vulkan/scene/LightingDebug.h"
#include "saiga/vulkan/scene/LocalLights.h"
#include "saiga/vulkan/scene/MeshPool.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/SceneCulling.h"
#include "saiga/vulkan/scene/SceneLightingSet.h"
#include "saiga/vulkan/scene/ShadowAtlas.h"
#include "saiga/vulkan/scene/ShadowPipelineTable.h"
#include "saiga/vulkan/scene/SkyAmbient.h"
#include "saiga/vulkan/texture/Sampler.h"
#include "saiga/vulkan/texture/Texture.h"
#include "saiga/vulkan/window/Window.h"
#include "saiga/vulkan/world/FoliageGenerator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

namespace
{
constexpr vk::Format hdrFormat   = vk::Format::eR16G16B16A16Sfloat;
constexpr vk::Format depthFormat = vk::Format::eD32Sfloat;

// The push constants of shader/vulkan/scene/resolve.frag and post.frag
struct ResolvePush
{
    int32_t samples;
    float exposureScale;
};
struct PostPush
{
    float exposureScale;
    int32_t toneMap;
    int32_t srgbTarget;
};

// The piecewise sRGB decode (the clear color is a display value)
float srgbToLinear(float c)
{
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}
// The piecewise sRGB encode (an enclosed space's linear background in clay mode, which writes display values)
float linearToSrgb(float c)
{
    return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

vk::ImageSubresourceRange colorRange()
{
    return vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
}
vk::ImageSubresourceRange depthRange()
{
    return vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);
}

// The GPU sections in frame order (the dotted names: sub-sections inside the pass before the dot)
const std::vector<const char*>& sectionNames()
{
    static const std::vector<const char*> names = {
        "transfer",         "environment",  "cull",      "lights",        "shadow",      "shadow.cascades",
        "shadow.horizon",   "shadow.atlas", "depth",     "depth.terrain", "scene",       "scene.terrain",
        "scene.instances",  "scene.grass",  "scene.sky", "scene.clouds",  "scene.debug", "resolve",
        "lighting.preview", "post",         "imgui"};
    return names;
}
std::vector<std::string> sectionNameStrings()
{
    std::vector<std::string> r;
    for (const char* n : sectionNames()) r.push_back(n);
    return r;
}
}  // namespace

// S36b: an atlas face of the frame with its culling view and terrain sphere
struct RenderScene::AtlasFace
{
    ShadowAtlas::Face face;
    uint32_t view = 0;
};

// probe_lighting's sequence (S36a1b): the views of its frames, the pixels and the values per frame
struct RenderScene::ProbeRun
{
    std::vector<RenderScene::ProbeRequest> requests;
    int step = 0;
    // Per frame: the values of the pixels (in request order)
    std::array<std::vector<LightingDebug::PixelValue>, RenderScene::kProbeFrames> values;
    // Frame 1's camera, its matrices, size and exposure
    Vec3 camera       = Vec3::Zero();
    mat4 viewProj     = mat4::Identity();
    ivec2 size        = ivec2(0, 0);
    float preExposure = 1;
    bool physical     = false;
    // Frame 1's camera is taken (the camera given from then on is held; frame 1's transfer callback still sets it)
    bool cameraTaken = false;
    // Frame 1's post pass (the later frames show frame 1's image: the window does not flash the debug frames)
    float postScale = 1;
    bool postLinear = false;
    // S36b: frame 1's lights and cluster grid (the probe's cluster keys)
    ClusterGrid grid;
    bool clusters = false;
    mat4 view     = mat4::Identity();
    mat4 proj     = mat4::Identity();
    Vec3 forward  = Vec3(0, 0, -1);
    std::vector<LocalLights::GpuLight> lights;
    std::vector<uint32_t> handles;
    // The surface changed size during the sequence (its pixels are other surface points; frame 1's held image is gone):
    // the run ends with the next latch and answers without values (CodeRabbit #82)
    bool resized = false;
};

namespace
{
// The views of the probe's frames (frame 1: normal); frames 5-7 write raw radiance, frame 8 the cloud shadow, frame 9
// the cascades (S36a2: factor, cascade + 1), frame 10 the horizon map (factor, R and G above the receiver; raw data),
// frame 11 the local lights' atlas shadow (S36b: factor, the shadowed lights that reach the pixel)
const std::array<LightingDebugView, RenderScene::kProbeFrames> kProbeViews = {
    LightingDebugView::None,          LightingDebugView::Albedo,      LightingDebugView::Normals,
    LightingDebugView::Material,      LightingDebugView::DirectOnly,  LightingDebugView::AmbientOnly,
    LightingDebugView::SpecularOnly,  LightingDebugView::CloudShadow, LightingDebugView::ShadowCascades,
    LightingDebugView::HorizonShadow, LightingDebugView::AtlasShadow};
}  // namespace

/**
 * The frame of RenderScene (S17): a VulkanRenderer with the HDR targets, the resolve and the post pass. The targets are
 * shared by the frames in flight (as the depth buffer of the forward renderer): each frame waits for the previous
 * frame's use in its barriers.
 */
class SceneFrameRenderer : public VulkanRenderer
{
   public:
    SceneFrameRenderer(RenderScene& scene, VulkanWindow& window, const VulkanParameters& vulkanParameters)
        : VulkanRenderer(window, vulkanParameters, sectionNameStrings()), scene(scene)
    {
        const int requested = scene.params.msaaSamples;
        samples             = selectSampleCount(base().physicalDevice, requested, hdrFormat);
        if (int(samples) != requested)
            std::cout << "RenderScene: " << requested << "x MSAA is not supported, using " << int(samples) << "x"
                      << std::endl;

        SamplerParameters sp;
        sp.setFilter(vk::Filter::eNearest);
        sp.setAddressMode(vk::SamplerAddressMode::eClampToEdge);
        sampler.create(base(), sp);

        PipelineInfo info;
        info.rasterizationState.cullMode        = vk::CullModeFlagBits::eNone;
        info.depthStencilState.depthTestEnable  = VK_FALSE;
        info.depthStencilState.depthWriteEnable = VK_FALSE;
        auto create = [&](Pipeline& pipeline, const char* fragment, uint32_t pushSize, vk::Format format)
        {
            DescriptorSetLayout layout(
                {{0, {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}}});
            pipeline.init(base(), 1);
            pipeline.addDescriptorSetLayout(layout);
            pipeline.addPushConstantRange({vk::ShaderStageFlagBits::eFragment, 0, pushSize});
            pipeline.shaderPipeline.load(base().device, {"vulkan/scene/fullscreen.vert", fragment});
            RenderingFormats formats;
            formats.color = {format};
            pipeline.create(formats, info);
        };
        if (samples != vk::SampleCountFlagBits::e1)
        {
            create(resolvePipeline, "vulkan/scene/resolve.frag", sizeof(ResolvePush), hdrFormat);
            resolveSet = resolvePipeline.createDescriptorSet();
        }
        create(postPipeline, "vulkan/scene/post.frag", sizeof(PostPush), getColorFormat());
        postSet      = postPipeline.createDescriptorSet();
        probeHeldSet = postPipeline.createDescriptorSet();

        // ImGui draws on top of the post pass (single-sampled, no depth)
        if (imGui) imGui->initResources(base(), outputFormats());
        should_render_imgui = scene.params.imgui;
    }

    ~SceneFrameRenderer() override { waitIdle(); }

    // The section marks of the frame (FrameTimings)
    void enter(const char* name, vk::CommandBuffer cmd) { timings.enterSection(name, cmd); }
    void leave(const char* name, vk::CommandBuffer cmd) { timings.leaveSection(name, cmd); }
    void subBegin(const char* name, vk::CommandBuffer cmd) { timings.markSectionBegin(name, cmd); }
    void subEnd(const char* name, vk::CommandBuffer cmd) { timings.markSectionEnd(name, cmd); }

    using VulkanRenderer::render;

    RenderingFormats sceneFormats() const
    {
        RenderingFormats f;
        f.color   = {hdrFormat};
        f.depth   = depthFormat;
        f.samples = samples;
        return f;
    }
    RenderingFormats outputFormats() const
    {
        RenderingFormats f;
        f.color = {getColorFormat()};
        return f;
    }

    void createBuffers(int numImages, int w, int h) override
    {
        (void)numImages;
        // Only called after waitIdle (VulkanRenderer::reset), so the old images and the descriptor sets are not in use
        depthImage.destroy();
        depthResolved.destroy();
        hdrMsaa.destroy();
        hdrResolved.destroy();
        const bool msaa = samples != vk::SampleCountFlagBits::e1;
        // The sampled depth (S19a): the SAMPLE_ZERO resolve of the multisampled depth, or with 1 sample the depth
        // image itself
        const auto depthUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
        // The sampled depth is also a copy source (RenderScene::readDepthForTests, S20)
        const auto sampledUsage = depthUsage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc;
        depthImage.create(base(), w, h, depthFormat, msaa ? depthUsage : sampledUsage, 1, samples);
        if (msaa) depthResolved.create(base(), w, h, depthFormat, sampledUsage);
        if (msaa)
            hdrMsaa.create(base(), w, h, hdrFormat,
                           vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, 1, samples);
        hdrResolved.create(base(), w, h, hdrFormat,
                           vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
        // The sets are read in eShaderReadOnlyOptimal (the frame's barriers), not in the tracked layout of now
        auto write = [&](StaticDescriptorSet& set, Texture2D& texture)
        {
            vk::DescriptorImageInfo info(vk::Sampler(sampler), texture.view(), vk::ImageLayout::eShaderReadOnlyOptimal);
            base().device.updateDescriptorSets(vk::WriteDescriptorSet(vk::DescriptorSet(set), 0, 0, 1,
                                                                      vk::DescriptorType::eCombinedImageSampler, &info),
                                               nullptr);
        };
        if (msaa) write(resolveSet, hdrMsaa);
        write(postSet, hdrResolved);
        probeHeld.destroy();
    }

    bool probeHeldValid() const
    {
        return probeHeld.memoryLocation && probeHeld.width == uint32_t(surfaceWidth) &&
               probeHeld.height == uint32_t(SurfaceHeight);
    }

    // The probe's frame 1 (hdrResolved in eTransferSrcOptimal): copies the resolved image into probeHeld (created at
    // the first probe, with its post set), which the post pass of the probe's debug frames shows
    void recordProbeHold(vk::CommandBuffer cmd)
    {
        if (!probeHeldValid())
        {
            probeHeld.destroy();
            probeHeld.create(base(), uint32_t(surfaceWidth), uint32_t(SurfaceHeight), hdrFormat,
                             vk::ImageUsageFlagBits::eSampled);
            vk::DescriptorImageInfo info(vk::Sampler(sampler), probeHeld.view(),
                                         vk::ImageLayout::eShaderReadOnlyOptimal);
            base().device.updateDescriptorSets(vk::WriteDescriptorSet(vk::DescriptorSet(probeHeldSet), 0, 0, 1,
                                                                      vk::DescriptorType::eCombinedImageSampler, &info),
                                               nullptr);
        }
        vk::ImageMemoryBarrier2 toDst(Stage::eFragmentShader, {}, Stage::eAllTransfer, Access::eTransferWrite,
                                      vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                                      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, probeHeld.image(),
                                      colorRange());
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toDst));
        const vk::ImageSubresourceLayers layers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
        cmd.copyImage(
            hdrResolved.image(), vk::ImageLayout::eTransferSrcOptimal, probeHeld.image(),
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageCopy(layers, {0, 0, 0}, layers, {0, 0, 0}, {uint32_t(surfaceWidth), uint32_t(SurfaceHeight), 1}));
        vk::ImageMemoryBarrier2 toRead(Stage::eAllTransfer, Access::eTransferWrite, Stage::eFragmentShader,
                                       Access::eShaderSampledRead, vk::ImageLayout::eTransferDstOptimal,
                                       vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED,
                                       VK_QUEUE_FAMILY_IGNORED, probeHeld.image(), colorRange());
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toRead));
        probeHeld.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    void render(const FrameInfo& frame) override;

    // Tests: the sampled depth of the last frame (in eShaderReadOnlyOptimal after a frame), row 0 at the top
    std::vector<float> readDepth(int& w, int& h)
    {
        waitIdle();
        Texture2D& depth   = sampledDepth();
        w                  = surfaceWidth;
        h                  = SurfaceHeight;
        const size_t bytes = size_t(w) * size_t(h) * 4;
        Buffer host;
        host.createBuffer(base(), bytes, vk::BufferUsageFlagBits::eTransferDst,
                          vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        auto& q               = base().mainQueue;
        vk::CommandBuffer cmd = q.commandPool.createAndBeginOneTimeBuffer();
        auto barrier          = [&](vk::ImageLayout from, vk::ImageLayout to)
        {
            vk::ImageMemoryBarrier2 b(Stage::eAllCommands, Access::eMemoryWrite | Access::eMemoryRead,
                                      Stage::eAllCommands, Access::eMemoryWrite | Access::eMemoryRead, from, to,
                                      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, depth.image(), depthRange());
            cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, b));
        };
        barrier(vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal);
        {
            Memory::SafeAccessor acc(*host.memoryLocation());
            vk::BufferImageCopy region(0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eDepth, 0, 0, 1),
                                       {0, 0, 0}, {uint32_t(w), uint32_t(h), 1});
            cmd.copyImageToBuffer(depth.image(), vk::ImageLayout::eTransferSrcOptimal, host.memoryLocation()->data,
                                  region);
        }
        barrier(vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        cmd.end();
        q.submitAndWait(cmd);
        q.commandPool.freeCommandBuffer(cmd);
        std::vector<float> out(size_t(w) * size_t(h));
        std::memcpy(out.data(), host.getMappedPointer(), bytes);
        host.destroyImmediately();
        return out;
    }

    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;

   private:
    RenderScene& scene;
    Sampler sampler;
    Pipeline resolvePipeline, postPipeline;
    StaticDescriptorSet resolveSet, postSet, probeHeldSet;
    // Shared by the frames in flight. hdrMsaa only with MSAA (stored only for the custom resolve); hdrResolved is the
    // scene target with 1 sample. depthResolved: the SAMPLE_ZERO depth resolve, only with MSAA.
    Texture2D depthImage, depthResolved, hdrMsaa, hdrResolved;
    // probe_lighting: frame 1's resolved image, shown by the sequence's debug frames (created at the first probe)
    Texture2D probeHeld;
    // The depth image that later passes sample (S37's consumers)
    Texture2D& sampledDepth() { return samples != vk::SampleCountFlagBits::e1 ? depthResolved : depthImage; }

    void viewport(vk::CommandBuffer cmd)
    {
        cmd.setViewport(0, vk::Viewport(0, 0, (float)surfaceWidth, (float)SurfaceHeight, 0.0f, 1.0f));
        cmd.setScissor(0, vk::Rect2D({0, 0}, {(uint32_t)surfaceWidth, (uint32_t)SurfaceHeight}));
    }
    vk::Rect2D area() const { return vk::Rect2D({0, 0}, {(uint32_t)surfaceWidth, (uint32_t)SurfaceHeight}); }
};

void SceneFrameRenderer::render(const FrameInfo& frame)
{
    RenderScene& s = scene;
    // An acquired frame: latched before any callback (a renderFrame call without a frame keeps the last frame's values)
    s.latchFrame();
    // The previews' statistics of this slot's previous frame (completed)
    s.debugImages->harvest(frame.frameIndex);
    s.debugImages->setFrame(s.frameCount + 1);
    // S36b: the clusters' statistics of this slot's previous frame (completed)
    s.clusterCull->harvest(frame.frameIndex);
    // probe_lighting's frames 2-7 (debug views): no preview conversion (they would reduce the debug values; the
    // requests wait for the frame after the sequence), the post pass shows frame 1's image
    RenderScene::ProbeRun* probe = s.probeRun.get();
    const bool probeDebugFrame   = probe && probe->step > 0;
    const bool msaa              = samples != vk::SampleCountFlagBits::e1;
    const bool linear            = s.sceneWritesLinear();
    const bool custom            = msaa && linear;  // the tone-map weighted resolve (Identity: the hardware average)
    const float scale            = s.frameEffectiveExposure / s.framePreExposure;
    const bool drawImgui         = imguiFrame(
        [&s]()
        {
            if (s.guiFn && *s.guiFn) (*s.guiFn)();
        });
    s.frameDrewImgui = drawImgui;

    vk::CommandBuffer cmd = beginCommands(frame);
    timings.resetFrame(cmd);

    // The statistics slot and the scene's camera of this frame (the aspect of the surface)
    s.beginSceneFrame(frame.frameIndex);
    SceneCamera camera;
    s.frameCamera(surfaceWidth, SurfaceHeight, camera);
    // The prototype settings of this frame: one snapshot after the transfer callback (which may set them) for the
    // lighting, the pre-pass and every prototype draw; a later change (e.g. in the render callback) is the next frame's
    PrototypeSettings proto;
    bool prepass = false;
    // The camera space's render settings (S21): an enclosed space draws no terrain, grass, water or sky
    bool enclosed = false;
    // The lighting block of this frame (written after the environment section)
    SceneLighting lighting;

    timings.enterSection("transfer", cmd);
    {
        // The buffers written in transfer (uniform buffers, vertex data, the scene's instances, meshes and indirect
        // commands) are shared by all frames in flight: wait until earlier frames have read them (write-after-read)
        // and the culling passes have written them (write-after-write) ...
        vk::MemoryBarrier2 war(Stage::eVertexInput | Stage::eVertexShader | Stage::eFragmentShader |
                                   Stage::eComputeShader | Stage::eDrawIndirect,
                               Access::eShaderStorageWrite, Stage::eAllTransfer, Access::eTransferWrite);
        cmd.pipelineBarrier2(vk::DependencyInfo({}, war, nullptr, nullptr));
        if (s.transferFn && *s.transferFn) (*s.transferFn)(cmd);
        // The terrain uploads (S20): at most the cap per frame, in queue order
        s.atlas->recordUploads(cmd, frame.frameIndex);
        // The transfer callback may set the camera of this frame (S19a)
        s.frameCamera(surfaceWidth, SurfaceHeight, camera);
        proto    = *s.protoSettings;
        prepass  = proto.depthPrePass;
        enclosed = camera.valid && !s.spaceSettings(camera.space).openWorld;
        // The S20 terrain: the selection over the GPU set (nothing without a resident pyramid of the camera's space;
        // nothing in an enclosed camera space, S21)
        if (enclosed)
            s.terrain->skipFrame();
        else
            s.terrain->transfer(cmd, frame.frameIndex, *s.atlas, camera, proto,
                                s.envSet ? vec2(s.env.phases[3], s.env.phases[4]) : vec2(0, 0), surfaceWidth,
                                SurfaceHeight);
        s.grass->update(*s.atlas, *s.terrain, camera, proto);
        s.slotStatistics[frame.frameIndex].terrainNodes = uint32_t(s.terrain->stats().nodes);
        s.frameLighting(camera, proto, enclosed, int(samples), lighting);
        // S36a2: the cascades of the shadow light (fit, views, the block's matrices) and their terrain casters
        s.frameShadows(camera, enclosed, lighting, frame.frameIndex);
        if (s.frameCascades && s.params.lighting.terrainCastsInCascades)
            s.terrain->selectForShadow(frame.frameIndex, *s.atlas, camera, s.shadowViews);
        s.cascadeOverlays();
        // S36b: the dynamics snapped in an earlier frame and not since come to rest; the local lights of the frame
        s.settleSnapped();
        s.frameLocalLights(camera, lighting, frame.frameIndex);
        // S36b: the shadow atlas (its faces are views of the culling after the cascades')
        s.frameAtlas(cmd, camera, enclosed, lighting, frame.frameIndex);
        s.lightOverlays();
        s.clusterOverlays();
        if (probe && probe->step == 0)
        {
            // The probe's frame 1: its camera and exposure (the later frames hold them); a world point's pixel with
            // this camera
            probe->camera      = camera.position;
            probe->viewProj    = camera.viewProj;
            probe->size        = ivec2(surfaceWidth, SurfaceHeight);
            probe->preExposure = s.framePreExposure;
            probe->physical    = s.framePhysical && linear;
            probe->postScale   = scale;
            probe->postLinear  = linear;
            probe->grid        = *s.clusterGrid;
            probe->clusters    = s.frameClusters;
            probe->view        = s.frameView;
            probe->proj        = s.frameProjection;
            probe->forward     = s.cameraRotation * Vec3(0, 0, -1);
            probe->lights      = s.lightTable->frameRecords();
            probe->handles.clear();
            for (const FrameLightInfo& f : s.lightTable->frameLights()) probe->handles.push_back(f.handle);
            probe->cameraTaken = true;
            for (RenderScene::ProbeRequest& r : probe->requests)
            {
                if (!r.world) continue;
                const Vec3 rel = *r.world - camera.position;
                const vec4 h   = camera.viewProj * vec4(float(rel(0)), float(rel(1)), float(rel(2)), 1.0f);
                // The range is checked in float before the conversion (a point far off screen projects beyond the
                // int range: undefined behaviour)
                const float px    = std::floor((h(0) / h(3) * 0.5f + 0.5f) * float(surfaceWidth));
                const float py    = std::floor((h(1) / h(3) * 0.5f + 0.5f) * float(SurfaceHeight));
                const bool inside = h(3) > 0 && std::isfinite(px) && std::isfinite(py) && px >= 0 && py >= 0 &&
                                    px < float(surfaceWidth) && py < float(SurfaceHeight);
                r.pixel = inside ? ivec2(int(px), int(py)) : ivec2(-1, -1);
            }
        }
        if (probe && probe->step > 0 && probe->size != ivec2(surfaceWidth, SurfaceHeight)) probe->resized = true;
        // probe_lighting: the pixels of this frame of the sequence
        if (probe)
        {
            std::vector<ivec2> pixels;
            for (auto& r : probe->requests) pixels.push_back(r.pixel);
            RenderScene* scene = &s;
            s.debugImages->requestPixels(pixels,
                                         [scene, probe](const std::vector<LightingDebug::PixelValue>& v)
                                         {
                                             if (scene->probeRun.get() != probe) return;
                                             probe->values[size_t(probe->step)] = v;
                                             probe->step++;
                                             if (probe->step == RenderScene::kProbeFrames) scene->finishProbe();
                                         });
        }
        // The overlays (DebugLines): the sun and the moon from the camera's focus
        if ((s.frameDebug.overlays & OverlaySunDirection) && camera.valid)
        {
            const Vec3 forward = s.cameraRotation * Vec3(0, 0, -1);
            const Vec3 focus   = camera.position + forward * 20.0;
            const vec3 sun     = s.envSet ? s.env.sunDir : s.studioEnvironment(enclosed).sunDir;
            s.lines->addLine(focus, focus + sun.cast<double>().normalized() * 8.0, vec3(1.0f, 0.85f, 0.2f), true);
            if (s.envSet)
                s.lines->addLine(focus, focus + s.env.moonDir.cast<double>().normalized() * 6.0, vec3(0.6f, 0.7f, 1.0f),
                                 true);
            s.lines->addLine(focus - Vec3(0.5, 0, 0), focus + Vec3(0.5, 0, 0), vec3(1, 1, 1), true);
            s.lines->addLine(focus - Vec3(0, 0, 0.5), focus + Vec3(0, 0, 0.5), vec3(1, 1, 1), true);
        }
        // S23c (F4): the PrototypeFoliage class is checked in the frame, not at its material's registration (a walk
        // over every instance set: only with SAIGA_ASSERTS; without the texture the class is never drawn)
#ifdef SAIGA_ASSERTS
        if (!s.foliage) s.checkFoliageInstances();
#endif
        s.culling->recordTransfer(cmd, frame.frameIndex, *s.meshPool, *s.instanceBuffer, camera,
                                  s.shadowViewsOverride ? *s.shadowViewsOverride : s.shadowViews);
        // ... and make the writes visible to this frame's culling and draws.
        vk::MemoryBarrier2 raw(Stage::eAllTransfer, Access::eTransferWrite,
                               Stage::eVertexInput | Stage::eVertexShader | Stage::eFragmentShader |
                                   Stage::eComputeShader | Stage::eDrawIndirect,
                               Access::eUniformRead | Access::eShaderRead | Access::eVertexAttributeRead |
                                   Access::eIndexRead | Access::eIndirectCommandRead);
        cmd.pipelineBarrier2(vk::DependencyInfo({}, raw, nullptr, nullptr));
    }
    timings.leaveSection("transfer", cmd);

    // The environment (S36a1c, §10.5 pass 1): the atmosphere's LUTs and the environment map, only in the physical path
    // of an open camera space (written every frame, empty otherwise)
    timings.enterSection("environment", cmd);
    const bool environmentPass =
        s.framePhysical && s.envSet && !enclosed && !s.env.enclosed && camera.valid && s.atmosphere;
    // A full update of the incremental effects (the environment map, S36a2: the horizon map) in this frame
    const bool fullUpdate = s.params.lighting.forceFullUpdates || s.fullUpdateRequested;
    s.fullUpdateRequested = false;
    if (environmentPass)
    {
        // A full update also on the first frame after frames without this pass (clay, an enclosed space, a debug
        // frame): the bound environment map is of an older environment (review: it was swapped in 7 frames later)
        LightingParameters lp = s.params.lighting;
        lp.forceFullUpdates   = fullUpdate || s.environmentSkipped;
        // probe_lighting: one capture for the whole sequence (CodeRabbit #83: a capture in progress was swapped in
        // between its frames): a capture in progress completes in frame 1, the later frames keep the front cube
        const bool probeHold = probe && probe->step > 0;
        // A held frame records no capture: its full update goes to the first frame after the probe (CodeRabbit #84: a
        // requestFullUpdate during the probe was lost)
        s.environmentSkipped = probeHold && lp.forceFullUpdates;
        if (probe && probe->step == 0 && s.envMap->capturing()) lp.forceFullUpdates = true;
        s.atmosphere->record(cmd, s.env, lighting.atmosphere(0));
        s.clouds->record(cmd, s.env.clouds.seed);
        EnvironmentMap::CaptureInputs in;
        in.skyView        = s.atmosphere->skyViewInfo();
        in.transmittance  = s.atmosphere->transmittanceInfo();
        in.skyViewHorizon = s.atmosphere->skyViewHorizonElevation();
        in.coverage       = s.clouds->coverageInfo();
        in.cloudOrigin    = vec2(lighting.cloudMotion(0), lighting.cloudMotion(1));
        in.clouds         = lighting.cloudMotion(3) > 0.5f;
        in.cloudiness     = lighting.clouds(2);
        if (lighting.shadowLight(3) > 0)
        {
            in.light            = vec3(lighting.shadowLight(0), lighting.shadowLight(1), lighting.shadowLight(2));
            in.lightIlluminance = lighting.shadowLight(3) == 1 ? s.env.sunIlluminance : s.env.moonIlluminance;
        }
        if (!probeHold) s.envMap->record(cmd, s.env, lighting.atmosphere(0), lp, in, s.frameCount + 1);
    }
    else
        s.environmentSkipped = true;
    timings.leaveSection("environment", cmd);
    // The lighting set of this frame (after the environment: the environment map swapped in this frame is bound)
    s.writeLightingSet(cmd, frame.frameIndex, lighting, proto.terrain);

    // The GPU culling of the scene's instances (written every frame, empty without instances or a camera)
    timings.enterSection("cull", cmd);
    s.culling->recordCull(cmd, frame.frameIndex, *s.meshPool, camera);
    timings.leaveSection("cull", cmd);

    // S36b: the clustered light culling (B3; written every frame, empty without lights)
    timings.enterSection("lights", cmd);
    if (s.frameClusters)
        s.clusterCull->record(cmd, frame.frameIndex, *s.clusterGrid, s.frameView, s.frameProjection, *s.lightTable);
    timings.leaveSection("lights", cmd);

    // S36a2: the cascades' depth passes (§10.5 pass 1; written every frame, empty without cascades): the casters of the
    // updated cascades (instances by class with the shadow pipeline table, the terrain nodes of selectForShadow)
    timings.enterSection("shadow", cmd);
    subBegin("shadow.cascades", cmd);
    if (s.frameCascades && !s.shadowViewsOverride)
    {
        CascadedShadows& cs   = *s.cascades;
        const int res         = cs.resolution();
        const float slopeBias = s.params.lighting.slopeBias;
        std::vector<vk::ImageMemoryBarrier2> toWrite, toRead;
        for (int c = 0; c < cs.count(); ++c)
        {
            if (!cs.cascades()[size_t(c)].updated) continue;
            const vk::ImageSubresourceRange layer(vk::ImageAspectFlagBits::eDepth, 0, 1, uint32_t(c), 1);
            // The previous frames' sampled reads (scene: also the grass's vertex shader, previews) before the writes;
            // the content is replaced
            toWrite.emplace_back(SceneLightingSet::kReaderStages, Access::eShaderSampledRead,
                                 Stage::eEarlyFragmentTests | Stage::eLateFragmentTests,
                                 Access::eDepthStencilAttachmentRead | Access::eDepthStencilAttachmentWrite,
                                 vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                 VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, cs.depth()->image(), layer);
            toRead.emplace_back(Stage::eLateFragmentTests, Access::eDepthStencilAttachmentWrite,
                                SceneLightingSet::kReaderStages, Access::eShaderSampledRead,
                                vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED,
                                VK_QUEUE_FAMILY_IGNORED, cs.depth()->image(), layer);
        }
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toWrite));
        for (int c = 0; c < cs.count(); ++c)
        {
            if (!cs.cascades()[size_t(c)].updated) continue;
            vk::RenderingAttachmentInfo depth;
            depth.imageView                     = cs.layerView(c);
            depth.imageLayout                   = vk::ImageLayout::eDepthStencilAttachmentOptimal;
            depth.loadOp                        = vk::AttachmentLoadOp::eClear;
            depth.storeOp                       = vk::AttachmentStoreOp::eStore;
            depth.clearValue.depthStencil.depth = 1.0f;  // not reversed
            vk::RenderingInfo info;
            info.renderArea       = vk::Rect2D({0, 0}, {uint32_t(res), uint32_t(res)});
            info.layerCount       = 1;
            info.pDepthAttachment = &depth;
            cmd.beginRendering(info);
            cmd.setViewport(0, vk::Viewport(0, 0, float(res), float(res), 0.0f, 1.0f));
            cmd.setScissor(0, info.renderArea);
            const ShadowView& v = s.shadowViews[size_t(c)];
            s.culling->recordShadow(cmd, frame.frameIndex, uint32_t(c) + 1, v.viewProj, camera.alpha, *s.shadowTable,
                                    slopeBias);
            if (s.params.lighting.terrainCastsInCascades)
                s.terrain->renderShadow(cmd, frame.frameIndex, size_t(c), v.viewProj, slopeBias, proto);
            cmd.endRendering();
        }
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toRead));
    }
    subEnd("shadow.cascades", cmd);
    // The horizon map's incremental update (S36a2, A6): the chain when the pyramid changed, the due rows of the back
    // map; only for the camera space's resident pyramid and a shadow light (empty otherwise)
    subBegin("shadow.horizon", cmd);
    {
        HorizonShadows::Input in;
        const bool terrainSet = s.atlas->hasSet() && s.atlas->pyramidResident() && camera.valid &&
                                s.atlas->space() == camera.space && !enclosed;
        in.atlas          = terrainSet ? s.atlas.get() : nullptr;
        in.terrainCleared = !s.atlas->hasSet();
        in.camera         = camera.position;
        in.hasLight   = lighting.shadowLight(3) > 0;
        in.light      = Vec3(lighting.shadowLight(0), lighting.shadowLight(1), lighting.shadowLight(2));
        in.parameters = s.params.lighting;
        in.force      = fullUpdate;
        in.frame      = s.frameCount + 1;
        in.frameIndex = frame.frameIndex;
        // probe_lighting: no update during the sequence, so every frame binds the maps of the frame before it (the
        // factors and the radiance of one map state; CodeRabbit #85 round 2); a full update goes to the first frame
        // after
        if (probe)
        {
            if (fullUpdate) s.horizon->requestRecompute(true);
        }
        else
            s.horizon->update(cmd, in);
    }
    subEnd("shadow.horizon", cmd);
    // S36b: the shadow atlas's faces (B4; empty without a face to render): per layer one pass that keeps the other
    // tiles, per face its tile cleared to 1 and the casters of its view (the static layer: static sets and the terrain
    // at the fixed level, the dynamic layer: dynamics)
    subBegin("shadow.atlas", cmd);
    if (!s.atlasFaces.empty() && !s.shadowViewsOverride && s.atlasShadows->hasImage())
    {
        ShadowAtlas& at = *s.atlasShadows;
        const vk::ImageSubresourceRange layers(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 2);
        // (the lighting set's readers: SceneLightingSet::kReaderStages; round 2, finding 18)
        vk::ImageMemoryBarrier2 toWrite(SceneLightingSet::kReaderStages, Access::eShaderSampledRead,
                                        Stage::eEarlyFragmentTests | Stage::eLateFragmentTests,
                                        Access::eDepthStencilAttachmentRead | Access::eDepthStencilAttachmentWrite,
                                        vk::ImageLayout::eShaderReadOnlyOptimal,
                                        vk::ImageLayout::eDepthStencilAttachmentOptimal, VK_QUEUE_FAMILY_IGNORED,
                                        VK_QUEUE_FAMILY_IGNORED, at.texture()->image(), layers);
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toWrite));
        const int tile    = at.tileSize();
        const int perSide = at.tilesPerSide();
        for (int layer = 0; layer < 2; ++layer)
        {
            bool any = false;
            for (const RenderScene::AtlasFace& f : s.atlasFaces) any = any || f.face.layer == layer;
            if (!any) continue;
            vk::RenderingAttachmentInfo depth;
            depth.imageView   = at.layerView(layer);
            depth.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
            depth.loadOp      = vk::AttachmentLoadOp::eLoad;
            depth.storeOp     = vk::AttachmentStoreOp::eStore;
            vk::RenderingInfo info;
            info.renderArea       = vk::Rect2D({0, 0}, {uint32_t(at.resolution()), uint32_t(at.resolution())});
            info.layerCount       = 1;
            info.pDepthAttachment = &depth;
            cmd.beginRendering(info);
            for (const RenderScene::AtlasFace& f : s.atlasFaces)
            {
                if (f.face.layer != layer) continue;
                const int tx = f.face.tile % perSide, ty = f.face.tile / perSide;
                const vk::Rect2D rect({tx * tile, ty * tile}, {uint32_t(tile), uint32_t(tile)});
                cmd.setViewport(0, vk::Viewport(float(tx * tile), float(ty * tile), float(tile), float(tile), 0, 1));
                cmd.setScissor(0, rect);
                vk::ClearAttachment clear(vk::ImageAspectFlagBits::eDepth, 0, vk::ClearDepthStencilValue(1.0f, 0));
                cmd.clearAttachments(clear, vk::ClearRect(rect, 0, 1));
                if (f.face.clearOnly) continue;
                s.culling->recordAtlas(cmd, frame.frameIndex, f.view, f.face.viewProj, camera.alpha, *s.shadowTable,
                                       f.face.lightPush);
                if (layer == 0)
                    s.terrain->renderAtlas(cmd, frame.frameIndex, f.face.light, f.face.viewProj, f.face.lightPush,
                                           proto);
            }
            cmd.endRendering();
        }
        vk::ImageMemoryBarrier2 toRead(Stage::eEarlyFragmentTests | Stage::eLateFragmentTests,
                                       Access::eDepthStencilAttachmentWrite, SceneLightingSet::kReaderStages,
                                       Access::eShaderSampledRead, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                       vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED,
                                       VK_QUEUE_FAMILY_IGNORED, at.texture()->image(), layers);
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toRead));
    }
    subEnd("shadow.atlas", cmd);
    timings.leaveSection("shadow", cmd);

    // ImGui's vertex data is in host visible memory, one buffer per frame in flight.
    if (drawImgui) imGui->updateBuffers(cmd, frame.frameIndex);

    {
        // The targets are shared by the frames in flight: the previous frame's attachment writes and sampled reads
        // (resolve, post, the sampled depth's consumers) come before this frame's writes; the previous content is
        // discarded.
        // (the compute stage: the preview conversion samples hdrResolved)
        const auto prevColor  = Stage::eColorAttachmentOutput | Stage::eFragmentShader | Stage::eComputeShader;
        const auto colorWrite = Access::eColorAttachmentWrite | Access::eColorAttachmentRead;
        const auto depthTests = Stage::eEarlyFragmentTests | Stage::eLateFragmentTests;
        const auto depthRW    = Access::eDepthStencilAttachmentWrite | Access::eDepthStencilAttachmentRead;
        std::vector<vk::ImageMemoryBarrier2> barriers;
        auto toAttachment = [&](Texture2D& t)
        {
            barriers.emplace_back(prevColor, Access::eColorAttachmentWrite, Stage::eColorAttachmentOutput, colorWrite,
                                  vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, t.image(), colorRange());
        };
        if (msaa) toAttachment(hdrMsaa);
        toAttachment(hdrResolved);
        // The depth: the previous frame's depth tests and (1 sample) its sampled reads
        barriers.emplace_back(depthTests | Stage::eFragmentShader | Stage::eComputeShader,
                              Access::eDepthStencilAttachmentWrite, depthTests, depthRW, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eDepthStencilAttachmentOptimal, VK_QUEUE_FAMILY_IGNORED,
                              VK_QUEUE_FAMILY_IGNORED, depthImage.image(), depthRange());
        // The depth resolve target: written by the resolve (the attachment output stage), read after the frame
        if (msaa)
            barriers.emplace_back(Stage::eColorAttachmentOutput | Stage::eLateFragmentTests | Stage::eFragmentShader |
                                      Stage::eComputeShader,
                                  Access::eColorAttachmentWrite | Access::eDepthStencilAttachmentWrite,
                                  Stage::eColorAttachmentOutput | Stage::eLateFragmentTests,
                                  Access::eColorAttachmentWrite | Access::eDepthStencilAttachmentWrite,
                                  vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, depthResolved.image(),
                                  depthRange());
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, barriers));
    }

    // 2. The depth pre-pass of the opaque content (written every frame, empty without the pre-pass)
    timings.enterSection("depth", cmd);
    if (prepass)
    {
        vk::RenderingAttachmentInfo depth;
        depth.imageView                     = depthImage.view();
        depth.imageLayout                   = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        depth.loadOp                        = vk::AttachmentLoadOp::eClear;
        depth.storeOp                       = vk::AttachmentStoreOp::eStore;
        depth.clearValue.depthStencil.depth = 0.0f;  // reversed-Z
        vk::RenderingInfo info;
        info.renderArea       = area();
        info.layerCount       = 1;
        info.pDepthAttachment = &depth;
        cmd.beginRendering(info);
        viewport(cmd);
        subBegin("depth.terrain", cmd);
        s.terrain->render(cmd, frame.frameIndex, TerrainPass::Mode::Depth, proto);
        subEnd("depth.terrain", cmd);
        viewport(cmd);
        s.culling->recordPrepass(cmd, frame.frameIndex, camera);
        cmd.endRendering();
        // The scene pass tests against (and writes) the pre-pass depth
        vk::MemoryBarrier2 b(Stage::eEarlyFragmentTests | Stage::eLateFragmentTests,
                             Access::eDepthStencilAttachmentWrite,
                             Stage::eEarlyFragmentTests | Stage::eLateFragmentTests,
                             Access::eDepthStencilAttachmentRead | Access::eDepthStencilAttachmentWrite);
        cmd.pipelineBarrier2(vk::DependencyInfo({}, b, nullptr, nullptr));
    }
    else
    {
        subBegin("depth.terrain", cmd);
        subEnd("depth.terrain", cmd);
    }
    timings.leaveSection("depth", cmd);

    // 3. The scene pass
    timings.enterSection("scene", cmd);
    {
        vec4 c = s.params.clearColor;
        if (enclosed)
        {
            // An enclosed space's background (S21, linear RGB) replaces the sky: pre-exposed like the scene shaders'
            // output, or as a display value in clay mode
            const vec3 b = s.spaceSettings(camera.space).background;
            for (int i = 0; i < 3; ++i)
                c[i] = linear ? std::max(b[i], 0.0f) * s.framePreExposure : linearToSrgb(std::clamp(b[i], 0.0f, 1.0f));
            c[3] = 1;
        }
        else if (linear)
        {
            // A display value: linear and pre-exposed as the scene shaders' output
            for (int i = 0; i < 3; ++i) c[i] = srgbToLinear(std::clamp(c[i], 0.0f, 1.0f)) * s.framePreExposure;
        }
        vk::RenderingAttachmentInfo color;
        color.imageLayout      = vk::ImageLayout::eColorAttachmentOptimal;
        color.loadOp           = vk::AttachmentLoadOp::eClear;
        color.clearValue.color = vk::ClearColorValue(std::array<float, 4>{c[0], c[1], c[2], c[3]});
        if (!msaa)
        {
            color.imageView = hdrResolved.view();
            color.storeOp   = vk::AttachmentStoreOp::eStore;
        }
        else if (custom)
        {
            color.imageView = hdrMsaa.view();
            color.storeOp   = vk::AttachmentStoreOp::eStore;
        }
        else
        {
            // Clay mode: the hardware average, only the resolved image is kept
            color.imageView          = hdrMsaa.view();
            color.storeOp            = vk::AttachmentStoreOp::eDontCare;
            color.resolveMode        = vk::ResolveModeFlagBits::eAverage;
            color.resolveImageView   = hdrResolved.view();
            color.resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        }

        vk::RenderingAttachmentInfo depth;
        depth.imageView                     = depthImage.view();
        depth.imageLayout                   = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        depth.loadOp                        = prepass ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear;
        depth.clearValue.depthStencil.depth = 0.0f;  // reversed-Z
        if (msaa)
        {
            // The single-sampled depth (S19a): SAMPLE_ZERO (lavapipe supports only it; reversed-Z "nearest" would be
            // MAX, not needed before S37's consumers); the multisampled depth is not kept
            depth.storeOp            = vk::AttachmentStoreOp::eDontCare;
            depth.resolveMode        = vk::ResolveModeFlagBits::eSampleZero;
            depth.resolveImageView   = depthResolved.view();
            depth.resolveImageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        }
        else
        {
            // 1 sample: the depth image itself is the sampled depth
            depth.storeOp = vk::AttachmentStoreOp::eStore;
        }

        vk::RenderingInfo info;
        info.renderArea           = area();
        info.layerCount           = 1;
        info.colorAttachmentCount = 1;
        info.pColorAttachments    = &color;
        info.pDepthAttachment     = &depth;
        cmd.beginRendering(info);
        viewport(cmd);
        // terrain -> the game's render callback -> instances -> grass -> sky (the sky's EQUAL at depth 0 last)
        subBegin("scene.terrain", cmd);
        // S36a2: the variant without the cascade and horizon code while both are off
        s.terrain->render(cmd, frame.frameIndex, prepass ? TerrainPass::Mode::Equal : TerrainPass::Mode::Scene, proto,
                          s.params.lighting.sunShadows || s.params.lighting.horizonShadows);
        subEnd("scene.terrain", cmd);
        viewport(cmd);
        if (s.renderFn && *s.renderFn) (*s.renderFn)(cmd);
        // The scene's instances after the game's draws
        viewport(cmd);
        subBegin("scene.instances", cmd);
        s.slotStatistics[frame.frameIndex].drawCommands = s.culling->recordDraw(cmd, frame.frameIndex, camera, prepass);
        subEnd("scene.instances", cmd);
        viewport(cmd);
        subBegin("scene.grass", cmd);
        if (s.foliage && !enclosed) s.grass->render(cmd, frame.frameIndex, *s.terrain, lighting.lightCounts(0) > 0);
        subEnd("scene.grass", cmd);
        subBegin("scene.sky", cmd);
        // The environment's sky (S20): with the foliage texture (the prototype), with or without terrain, and (S36a1c)
        // the atmosphere in the physical path; not in an enclosed camera space (S21: the background is the clear color)
        if ((s.foliage || environmentPass) && !enclosed) s.terrain->renderSky(cmd, frame.frameIndex, proto);
        subEnd("scene.sky", cmd);
        // The 2D cloud layer (S36a1c): after the sky, in the physical path with LightingParameters::clouds
        subBegin("scene.clouds", cmd);
        if (environmentPass && lighting.cloudMotion(3) > 0.5f)
        {
            viewport(cmd);
            s.clouds->draw(cmd, s.lightingSet->set(frame.frameIndex), camera.viewProj, s.framePreExposure);
        }
        subEnd("scene.clouds", cmd);
        // The debug lines (S36: the overlays; written every frame, empty without lines)
        subBegin("scene.debug", cmd);
        viewport(cmd);
        s.lines->record(cmd, frame.frameIndex, camera.position, camera.viewProj,
                        linear ? s.framePreExposure / std::max(s.frameEffectiveExposure, 1e-30f) : 0.0f);
        subEnd("scene.debug", cmd);
        cmd.endRendering();

        // The sampled depth for later passes (S37's consumers): after the resolve (MSAA) or the depth writes
        vk::ImageMemoryBarrier2 toRead(Stage::eColorAttachmentOutput | Stage::eLateFragmentTests,
                                       Access::eColorAttachmentWrite | Access::eDepthStencilAttachmentWrite,
                                       Stage::eFragmentShader | Stage::eComputeShader, Access::eShaderSampledRead,
                                       vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                       vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED,
                                       VK_QUEUE_FAMILY_IGNORED, sampledDepth().image(), depthRange());
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toRead));
    }
    timings.leaveSection("scene", cmd);

    // 3. The tone-map weighted resolve (written every frame, empty without it)
    timings.enterSection("resolve", cmd);
    if (custom)
    {
        vk::ImageMemoryBarrier2 toRead(Stage::eColorAttachmentOutput, Access::eColorAttachmentWrite,
                                       Stage::eFragmentShader, Access::eShaderSampledRead,
                                       vk::ImageLayout::eColorAttachmentOptimal,
                                       vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED,
                                       VK_QUEUE_FAMILY_IGNORED, hdrMsaa.image(), colorRange());
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toRead));

        vk::RenderingAttachmentInfo color;
        color.imageView   = hdrResolved.view();
        color.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        color.loadOp      = vk::AttachmentLoadOp::eDontCare;
        color.storeOp     = vk::AttachmentStoreOp::eStore;
        vk::RenderingInfo info;
        info.renderArea           = area();
        info.layerCount           = 1;
        info.colorAttachmentCount = 1;
        info.pColorAttachments    = &color;
        cmd.beginRendering(info);
        viewport(cmd);
        if (resolvePipeline.bind(cmd))
        {
            ResolvePush push{int32_t(samples), scale};
            resolvePipeline.bindDescriptorSet(cmd, resolveSet);
            resolvePipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eFragment, sizeof(push), &push);
            cmd.draw(3, 1, 0, 0);
        }
        cmd.endRendering();
    }
    timings.leaveSection("resolve", cmd);

    // S36a1b: the pixel readbacks of probe_lighting (the resolved colour and the sampled depth, after the resolve)
    const bool readPixels = s.debugImages->pixelsRequested();
    if (readPixels)
    {
        std::array<vk::ImageMemoryBarrier2, 2> toSrc = {
            vk::ImageMemoryBarrier2(Stage::eColorAttachmentOutput, Access::eColorAttachmentWrite, Stage::eAllTransfer,
                                    Access::eTransferRead, vk::ImageLayout::eColorAttachmentOptimal,
                                    vk::ImageLayout::eTransferSrcOptimal, VK_QUEUE_FAMILY_IGNORED,
                                    VK_QUEUE_FAMILY_IGNORED, hdrResolved.image(), colorRange()),
            vk::ImageMemoryBarrier2(Stage::eFragmentShader | Stage::eComputeShader, Access::eShaderSampledRead,
                                    Stage::eAllTransfer, Access::eTransferRead, vk::ImageLayout::eShaderReadOnlyOptimal,
                                    vk::ImageLayout::eTransferSrcOptimal, VK_QUEUE_FAMILY_IGNORED,
                                    VK_QUEUE_FAMILY_IGNORED, sampledDepth().image(), depthRange())};
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toSrc));
        s.debugImages->recordPixels(cmd, frame.frameIndex, hdrResolved.image(), sampledDepth().image(),
                                    ivec2(surfaceWidth, SurfaceHeight));
        // The probe's frame 1: a copy of the resolved image, which the post pass of frames 2-7 shows
        if (probe && probe->step == 0) recordProbeHold(cmd);
        vk::ImageMemoryBarrier2 depthBack(
            Stage::eAllTransfer, Access::eTransferRead, Stage::eFragmentShader | Stage::eComputeShader,
            Access::eShaderSampledRead, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, sampledDepth().image(), depthRange());
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, depthBack));
    }
    // (not in any frame of a probe, also not its normal frame 1: the captures wait for the frame after the sequence)
    const bool previews = s.debugImages->requested() && !probe;
    {
        // The resolved image -> post's sampled read (and S36's preview conversion); the output image after the acquire
        // (the submit waits for imageAvailable at COLOR_ATTACHMENT_OUTPUT; headless: the frame that last used it has
        // completed)
        vk::Image output                                = colorImage(frame.imageIndex);
        std::array<vk::ImageMemoryBarrier2, 2> barriers = {
            vk::ImageMemoryBarrier2(
                readPixels ? vk::PipelineStageFlags2(Stage::eAllTransfer) : Stage::eColorAttachmentOutput,
                readPixels ? vk::AccessFlags2() : vk::AccessFlags2(Access::eColorAttachmentWrite),
                Stage::eFragmentShader | Stage::eComputeShader, Access::eShaderSampledRead,
                readPixels ? vk::ImageLayout::eTransferSrcOptimal : vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                hdrResolved.image(), colorRange()),
            vk::ImageMemoryBarrier2(Stage::eColorAttachmentOutput, {}, Stage::eColorAttachmentOutput,
                                    Access::eColorAttachmentWrite | Access::eColorAttachmentRead,
                                    vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, output, colorRange())};
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, barriers));
    }
    // The preview conversions (S36a1b; written every frame, empty without a request)
    timings.enterSection("lighting.preview", cmd);
    {
        LightingDebug::Sources src;
        src.set(LightingImage::Luminance, &hdrResolved, 0, linear ? 1.0f / s.framePreExposure : 1.0f);
        // (the BRDF LUT is created for its preview; the environment's resources exist after the first environment)
        if (s.debugImages->requested(LightingImage::BrdfLut)) s.ensureBrdf();
        if (s.brdf) src.set(LightingImage::BrdfLut, &s.brdf->lut(), 1);
        // S36a1c: the atmosphere's LUTs and the environment map's debug images (recorded on demand), only in a frame
        // with the environment pass (round 2, finding 6: in clay or an enclosed space they are stale or never written;
        // the capture is an error then)
        if (s.atmosphere && environmentPass)
        {
            src.set(LightingImage::TransmittanceLut, &s.atmosphere->transmittance(), 0);
            src.set(LightingImage::SkyViewLut, &s.atmosphere->skyView(), 0, 1.0f / Atmosphere::kSkyLutScale);
            src.set(LightingImage::MultiScatteringLut, &s.atmosphere->multiScattering(), 0);
        }
        if (previews && environmentPass && s.envMap && s.debugImages->requested(LightingImage::Environment))
            src.set(
                LightingImage::Environment,
                s.envMap->recordDebug(cmd, frame.frameIndex, 0, s.debugImages->sourceMip(LightingImage::Environment)),
                0);
        if (previews && environmentPass && s.envMap && s.debugImages->requested(LightingImage::Irradiance))
            src.set(LightingImage::Irradiance, s.envMap->recordDebug(cmd, frame.frameIndex, 1, 0), 0);
        // S36a2: the cascades' depth (one layer, or every layer side by side; the minimum over the covered texels)
        if (s.cascades->hasImages())
        {
            const uint32_t res    = uint32_t(s.cascades->resolution());
            const uint32_t layers = uint32_t(s.cascades->layers());
            const uint32_t layer =
                uint32_t(std::clamp(s.debugImages->sourceLayer(LightingImage::Cascade), 0, int(layers) - 1));
            src.setArray(LightingImage::Cascade, s.cascades->depth(), 2, layer, 1, std::max(1u, res / 1024));
            src.setArray(LightingImage::CascadeStrip, s.cascades->depth(), 2, 0, layers, std::max(1u, res / 512));
        }
        // S36a2: the front horizon map's R or G (in eGeneral)
        if (s.horizon->hasImages() && s.horizon->transform(s.horizon->frontLayer()).valid)
        {
            const uint32_t res = s.horizon->images()->width;
            src.setArray(LightingImage::Horizon, s.horizon->images(),
                         s.debugImages->sourceLayer(LightingImage::Horizon) == 1 ? 4 : 3,
                         uint32_t(s.horizon->frontLayer()), 1, std::max(1u, res / 1024), vk::ImageLayout::eGeneral);
            s.lightStats.horizonPreview = s.horizon->statistics();
        }
        // The cloud layer's previews (the coverage over the tile, the shadow around the camera)
        // (the cloud layer's: with the environment pass and the layer on, as it is drawn)
        if (s.clouds && environmentPass && lighting.cloudMotion(3) > 0.5f &&
            (s.debugImages->requested(LightingImage::CloudCoverage) ||
             s.debugImages->requested(LightingImage::CloudShadow)))
        {
            CloudLayer::DebugInput in;
            in.clouds = lighting.clouds;
            in.origin = vec2(lighting.cloudMotion(0), lighting.cloudMotion(1));
            in.mean   = lighting.cloudMotion(2);
            in.height = lighting.atmosphere(0);
            in.light  = lighting.shadowLight(3) > 0
                            ? vec3(lighting.shadowLight(0), lighting.shadowLight(1), lighting.shadowLight(2))
                            : vec3(0, 1, 0);
            // The shadow preview is 1 without a shadow light (as the scene's cloud shadow)
            in.shadowLight = lighting.shadowLight(3) > 0;
            if (s.debugImages->requested(LightingImage::CloudCoverage))
                src.set(LightingImage::CloudCoverage, s.clouds->recordDebug(cmd, frame.frameIndex, 1, in), 1);
            if (s.debugImages->requested(LightingImage::CloudShadow))
                src.set(LightingImage::CloudShadow, s.clouds->recordDebug(cmd, frame.frameIndex, 2, in), 1);
        }
        // S36b: the shadow atlas (a layer; one light's tile) after this frame's faces
        if (s.atlasShadows->hasImage())
        {
            ShadowAtlas& at    = *s.atlasShadows;
            const uint32_t res = uint32_t(at.resolution());
            // (the layer, light and face of the request: round 2, finding 5)
            const PreviewMapping& am = s.debugImages->sourceMapping(LightingImage::Atlas);
            src.setArray(LightingImage::Atlas, at.texture(), 2, uint32_t(std::clamp(am.layer, 0, 1)), 1,
                         std::max(1u, res / 1024));
            const PreviewMapping& tm     = s.debugImages->sourceMapping(LightingImage::AtlasTile);
            const std::vector<int> tiles = at.tilesOf(tm.light);
            if (!tiles.empty())
            {
                const int t = tiles[size_t(std::clamp(tm.face, 0, int(tiles.size()) - 1))];
                src.setArray(LightingImage::AtlasTile, at.texture(), 2, uint32_t(std::clamp(tm.layer, 0, 1)), 1, 1);
                LightingDebug::Sources::ArraySource& a = src.array[size_t(LightingImage::AtlasTile)];
                a.originX                              = uint32_t((t % at.tilesPerSide()) * at.tileSize());
                a.originY                              = uint32_t((t / at.tilesPerSide()) * at.tileSize());
                a.width = a.height = uint32_t(at.tileSize());
            }
        }
        // S36b: the lights per cluster (from this frame's table)
        if (s.debugImages->requested(LightingImage::Clusters) && s.frameClusters)
            src.set(LightingImage::Clusters,
                    s.clusterCull->recordDebug(cmd, frame.frameIndex,
                                               s.debugImages->sourceMapping(LightingImage::Clusters).slice),
                    1);
        s.lightStats.previewDispatches = previews ? s.debugImages->recordPreviews(cmd, frame.frameIndex, src) : 0;
    }
    timings.leaveSection("lighting.preview", cmd);

    // 4. Post and ImGui on the output image
    {
        vk::RenderingAttachmentInfo color;
        color.imageView   = colorImageView(frame.imageIndex);
        color.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        color.loadOp      = vk::AttachmentLoadOp::eDontCare;  // post writes every pixel
        color.storeOp     = vk::AttachmentStoreOp::eStore;
        vk::RenderingInfo info;
        info.renderArea           = area();
        info.layerCount           = 1;
        info.colorAttachmentCount = 1;
        info.pColorAttachments    = &color;
        cmd.beginRendering(info);
        viewport(cmd);

        timings.enterSection("post", cmd);
        if (postPipeline.bind(cmd))
        {
            // The probe's frames 2-7 show frame 1's image (held by recordProbeHold)
            const bool held = probeDebugFrame && probeHeldValid();
            PostPush push{held ? probe->postScale : scale, (held ? probe->postLinear : linear) ? 1 : 0,
                          isSrgbFormat(getColorFormat()) ? 1 : 0};
            postPipeline.bindDescriptorSet(cmd, held ? probeHeldSet : postSet);
            postPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eFragment, sizeof(push), &push);
            cmd.draw(3, 1, 0, 0);
        }
        timings.leaveSection("post", cmd);

        timings.enterSection("imgui", cmd);
        if (drawImgui) imGui->render(cmd, frame.frameIndex);
        cmd.endRendering();
        // After the end of the pass: the frame's last section also covers the store of the output image
        timings.leaveSection("imgui", cmd);
    }

    // The layouts after this frame (Texture tracks them)
    depthImage.setImageLayout(msaa ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                   : vk::ImageLayout::eShaderReadOnlyOptimal);
    if (msaa) depthResolved.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    if (msaa)
        hdrMsaa.setImageLayout(custom ? vk::ImageLayout::eShaderReadOnlyOptimal
                                      : vk::ImageLayout::eColorAttachmentOptimal);
    hdrResolved.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    // Screenshots of the final image (tone mapped, with ImGui)
    vk::Image output = colorImage(frame.imageIndex);
    std::vector<ScreenshotCallback> callbacks;
    // (not in a probe frame: the screenshots wait for the first frame after the sequence)
    std::unique_ptr<StagingBuffer> screenshot = probe ? nullptr : beginScreenshots(cmd, output, callbacks);
    recordPresentTransition(cmd, output, screenshot != nullptr);
    submitCommands(cmd, frame);
    finishScreenshots(frame, screenshot.get(), callbacks);
    // S36a1b: the captures and pixel readbacks of this frame (blocking, like the screenshots)
    s.debugImages->finishFrame(frame.frameIndex, [this, &frame]() { base().waitFrameValue(frame.timelineValue); });
}

//==================================================================================================

RenderScene::RenderScene(VulkanWindow& window, const VulkanParameters& vulkanParameters,
                         const RenderSceneParameters& parameters)
    : params(parameters)
{
    // As setExposure: a NaN is 1
    params.exposure     = std::isnan(params.exposure) ? 1.0f : std::clamp(params.exposure, 1.0f / 1024, 1024.0f);
    VulkanParameters vp = vulkanParameters;
    vp.enableImgui      = parameters.imgui;
    renderer            = std::make_unique<SceneFrameRenderer>(*this, window, vp);
    meshPool            = std::make_unique<MeshPool>();
    meshPool->init(base(), params.vertexCapacity, params.indexCapacity);
    // The lighting (S36): set 1 of the scene draws, the ambient, the BRDF LUT
    lightingSet = std::make_unique<SceneLightingSet>();
    lightingSet->init(base());
    ambient = std::make_unique<SkyAmbient>();
    ambient->init(base());
    debugImages = std::make_unique<LightingDebug>();
    debugImages->init(base());
    lines = std::make_unique<DebugLines>();
    lines->init(base(), formats());
    cascades = std::make_unique<CascadedShadows>();
    cascades->init(base());
    shadowTable = std::make_unique<ShadowPipelineTable>();
    shadowTable->init(base());
    horizon = std::make_unique<HorizonShadows>();
    horizon->init(base());
    lightTable = std::make_unique<LocalLights>();
    lightTable->init(base());
    clusterCull = std::make_unique<LightClusters>();
    clusterCull->init(base());
    clusterGrid  = std::make_unique<ClusterGrid>();
    atlasShadows = std::make_unique<ShadowAtlas>();
    atlasShadows->init(base());
    culling = std::make_unique<SceneCulling>();
    culling->init(base(), formats(), params, *lightingSet);
    protoSettings = std::make_unique<PrototypeSettings>();
    // The static instance buffer is bound whole: it grows up to the storage buffer limit
    instanceBuffer = std::make_unique<InstanceBuffer>();
    instanceBuffer->init(base(), params.staticInstanceCapacity, culling->storageLimit());
    atlas = std::make_unique<TerrainAtlas>();
    atlas->init(base(), params, culling->storageLimit());
    terrain = std::make_unique<TerrainPass>();
    terrain->init(base(), formats(), *lightingSet);
    grass = std::make_unique<GrassPass>();
    grass->init(base(), formats(), *lightingSet);
    // The values the first frame latches (the game's init may read them)
    latchFrame();
}

RenderScene::~RenderScene()
{
    if (renderer) renderer->waitIdle();
    if (culling) culling->destroy();
    culling.reset();
    grass.reset();
    terrain.reset();
    atlas.reset();
    foliage.reset();
    lines.reset();
    debugImages.reset();
    shadowTable.reset();
    cascades.reset();
    horizon.reset();
    clusterCull.reset();
    atlasShadows.reset();
    lightTable.reset();
    brdf.reset();
    ambient.reset();
    envMap.reset();
    clouds.reset();
    atmosphere.reset();
    lightingSet.reset();
    instanceBuffer.reset();
    meshPool.reset();
}

float RenderScene::frameExposureViewCentre() const
{
    return exposureViewCentre ? *exposureViewCentre : physicalLighting() ? env.ev100 : float(std::log2(0.18 * 8));
}

bool RenderScene::renderFrame(const Callback& transferCb, const Callback& renderCb, const std::function<void()>& gui)
{
    transferFn              = &transferCb;
    renderFn                = &renderCb;
    guiFn                   = &gui;
    const uint64_t before   = renderer->submittedFrames();
    const uint64_t timeline = renderer->base().currentFrameSignalValue();
    RenderInfo info;
    // The override applies to this frame only, and only if it is read back (a screenshot that waits for a later frame
    // does not hide the GUI of the frames before it); the user's choice stays in should_render_imgui
    const bool userImgui = renderer->should_render_imgui;
    // (not in a probe frame: the screenshot waits for the frame after the sequence)
    const bool overridden = imguiOverride.has_value() && renderer->screenshotRequested() && !probePending();
    if (overridden) renderer->should_render_imgui = *imguiOverride;
    renderer->render(info);
    if (overridden) renderer->should_render_imgui = userImgui;
    transferFn = nullptr;
    renderFn   = nullptr;
    guiFn      = nullptr;
    if (renderer->submittedFrames() == before) return false;
    // The scene of the next frame is written with this frame's exposure; a debug frame (Identity latched) keeps the
    // history, so the next normal frame continues from the last tone-mapped frame (D12)
    if (frameDebug.view == LightingDebugView::None) lastExposure = frameEffectiveExposure;
    frameCount++;
    submittedFrames.push_back({timeline, frameCount});
    if (submittedFrames.size() > 32) submittedFrames.pop_front();
    return true;
}

void RenderScene::latchFrame()
{
    // Once per recorded frame, before any callback (setters apply from the next frame)
    latchDebug();
    // A debug frame latches Identity (no pipeline change: the output mode is a push constant, the resolve mode is
    // chosen per frame); the light model stays the requested tone map's (the views show the physical path's values)
    frameToneMap    = frameDebug.view != LightingDebugView::None ? ToneMap::Identity : params.toneMap;
    framePhysical   = envSet && params.toneMap != ToneMap::Identity;
    frameMultiplier = frameToneMap == ToneMap::Identity ? 1.0f : params.exposure;
    // S36 (D18): a scene with an environment exposes its EV100 (the multiplier on top); without one the multiplier is
    // the unit-less exposure, as before
    frameEffectiveExposure = frameToneMap == ToneMap::Identity ? 1.0f
                             : envSet                          ? exposureFromEv100(env.ev100, params.exposure)
                                                               : params.exposure;
    // The previous frame's exposure; this frame's after a change by more than kMaxPreExposureChange (fp16: the scaled
    // values would clip at 65504 or lose precision, a one-frame flash). Clay mode ignores exposure: 1.
    const float e = frameEffectiveExposure;
    float pre     = lastExposure;
    if (e > pre * kMaxPreExposureChange || e * kMaxPreExposureChange < pre) pre = e;
    framePreExposure = frameToneMap == ToneMap::Identity ? 1.0f : pre;
}

RenderingFormats RenderScene::formats() const
{
    return renderer->sceneFormats();
}

vk::Format RenderScene::colorFormat() const
{
    return renderer->getColorFormat();
}

VulkanBase& RenderScene::base()
{
    return renderer->base();
}

double RenderScene::lastRecordingMs() const
{
    return renderer->lastRecordingMs();
}

double RenderScene::lastWaitMs() const
{
    return renderer->lastWaitMs();
}

std::array<double, 3> RenderScene::lastWaitParts() const
{
    return {renderer->lastSlotWaitMs(), renderer->lastAcquireWaitMs(), renderer->lastPresentWaitMs()};
}

bool RenderScene::lastPresented() const
{
    return renderer->lastPresented();
}

bool RenderScene::vsync() const
{
    return renderer->vsync();
}

bool RenderScene::headless() const
{
    return renderer->isHeadless();
}

ivec2 RenderScene::size() const
{
    return renderer->surfaceSize();
}

void RenderScene::requestScreenshot(VulkanRenderer::ScreenshotCallback callback)
{
    renderer->requestScreenshot(std::move(callback));
}

std::vector<VulkanRenderer::ScreenshotCallback> RenderScene::takeScreenshotRequests()
{
    return renderer->takeScreenshotRequests();
}

ImGuiVulkanRenderer* RenderScene::imguiRenderer()
{
    return renderer->imguiRenderer();
}

void RenderScene::setImguiVisible(bool visible)
{
    renderer->should_render_imgui = visible;
}

bool RenderScene::imguiVisible() const
{
    return renderer->should_render_imgui;
}

void RenderScene::setToneMap(ToneMap toneMap)
{
    // A probe holds the tone map of its frame 1: it chooses the light model (physical or studio) of every frame
    if (probeRun)
    {
        heldToneMap = toneMap;
        return;
    }
    params.toneMap = toneMap;
}

void RenderScene::ensureBrdf()
{
    if (brdf) return;
    brdf = std::make_unique<BrdfLut>();
    brdf->init(base());
}

void RenderScene::ensureEnvironmentResources()
{
    if (atmosphere) return;
    ensureBrdf();
    atmosphere = std::make_unique<Atmosphere>();
    atmosphere->init(base());
    envMap = std::make_unique<EnvironmentMap>();
    envMap->init(base(), params.lighting.environmentResolution);
    clouds = std::make_unique<CloudLayer>();
    clouds->init(base(), formats(), CloudSettings().seed);
    ambient->setEnvironmentMap(envMap.get());
}

BrdfLut& RenderScene::brdfLut()
{
    ensureBrdf();
    return *brdf;
}

Atmosphere& RenderScene::atmosphereLuts()
{
    ensureEnvironmentResources();
    return *atmosphere;
}

EnvironmentMap& RenderScene::environmentMap()
{
    ensureEnvironmentResources();
    return *envMap;
}

CloudLayer& RenderScene::cloudLayer()
{
    ensureEnvironmentResources();
    return *clouds;
}

void RenderScene::setEnvironment(const EnvironmentState& environment)
{
    // The environment's resources (S36 review round 2, finding 4a: created with the first environment)
    ensureEnvironmentResources();
    // A probe holds the environment of its frame 1
    if (probeRun)
    {
        heldEnvironment = environment;
        return;
    }
    env    = environment;
    envSet = true;
}

void RenderScene::setLightingParameters(const LightingParameters& parameters)
{
    // Only stored: the resources that depend on the parameters (the environment map's resolution, the clouds, the
    // cascades and the horizon map) are created or resized lazily when the next frame records them
    // (EnvironmentMap::record, CascadedShadows::ensureImages, HorizonShadows::ensureImages)
    params.lighting = parameters;
}

void RenderScene::setExposure(float exposure)
{
    // NaN -> 1
    params.exposure = exposure == exposure ? std::clamp(exposure, 1.0f / 1024, 1024.0f) : 1.0f;
}

void RenderScene::waitIdle()
{
    renderer->waitIdle();
}

const std::vector<const char*>& RenderScene::gpuSectionNames()
{
    // The sections of the frame timings (SceneFrameRenderer), in frame order
    return sectionNames();
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
        lastTaken = out;
        return true;
    }
    return false;
}

//==================================================================================================
// Scene content (S18)

void RenderScene::addMaterial(MaterialId id, const std::string& name, const MaterialData& data)
{
    meshPool->addMaterial(id, name, data);
}

void RenderScene::setMaterial(MaterialId id, const MaterialData& data)
{
    // S36b: a caster class or its alpha may change: every static atlas tile is re-rendered (a rare call)
    atlasShadows->invalidateAll();
    meshPool->setMaterial(id, data);
}

void RenderScene::checkFoliageInstances() const
{
    // The class index of PipelineClass::PrototypeFoliage; no registered mesh of the class: nothing to check
    const int klass = int(PipelineClass::PrototypeFoliage) - 1;
    if (meshPool->classRanges()[size_t(klass)].count == 0) return;
    auto foliageMesh = [&](MeshId mesh) { return mesh != 0 && meshPool->meshClass(mesh) == klass; };
    bool draws       = false;
    for (const auto& entry : instanceBuffer->sets())
        for (MeshId mesh : entry.second.meshes) draws = draws || foliageMesh(mesh);
    for (const auto& d : instanceBuffer->dynamics()) draws = draws || (d.alive && d.set && foliageMesh(d.mesh));
    SAIGA_ASSERT(!draws, "PrototypeFoliage instances need the foliage texture of setFoliageTexture");
}

void RenderScene::setMeshLodDistances(MeshId id, const std::vector<float>& endDistances, float boundsPadding)
{
    meshPool->setLodDistances(id, endDistances, boundsPadding);
    instanceBuffer->refreshBounds(id, *meshPool);
}

uint16_t RenderScene::reserveIds(SceneIdKind kind, uint32_t count)
{
    uint32_t& next = nextReservedId[kind == SceneIdKind::Mesh ? 0 : 1];
    // count <= kGameReservedIds - next: next + count could wrap around (S19a review)
    SAIGA_ASSERT(count > 0 && count <= kGameReservedIds - next,
                 "reserveIds: the game range 1 .. kGameReservedIds - 1 is exhausted");
    const uint32_t first = next;
    next += count;
    return uint16_t(first);
}


//==================================================================================================
// Terrain (S20)

void RenderScene::setTerrainPyramid(SpaceId space, const TerrainPyramidData& pyramid)
{
    atlasShadows->invalidateAll();
    atlas->setPyramid(space, pyramid);
}

void RenderScene::setTerrainTile(const RegionId& region, const TerrainTileData& tile)
{
    // S36b: the atlas's static tiles under the region (their terrain key follows the residency as well)
    markRegionDirty(region);
    atlas->setTile(region, tile);
}

void RenderScene::removeTerrainTile(const RegionId& region)
{
    markRegionDirty(region);
    atlas->removeTile(region);
}

void RenderScene::setGrassMask(const ChunkId& chunk, const GrassMaskData& mask)
{
    atlas->setGrassMask(chunk, mask);
}

void RenderScene::removeGrassMask(const ChunkId& chunk)
{
    atlas->removeGrassMask(chunk);
}

void RenderScene::clearTerrain()
{
    atlasShadows->invalidateAll();
    atlas->clear();
}

void RenderScene::patchTerrainTile(const RegionId& region, int x0, int z0, int w, int h, const int32_t* heights)
{
    markRegionDirty(region);
    atlas->patchTile(region, x0, z0, w, h, heights);
}

void RenderScene::patchTerrainPyramid(SpaceId space, int x0, int z0, int w, int h, const int32_t* heights)
{
    // The pyramid's coarse levels: every static tile (rare: terraforming patches the tiles)
    atlasShadows->invalidateAll();
    atlas->patchPyramid(space, x0, z0, w, h, heights);
}

size_t RenderScene::terrainUploadsPending() const
{
    return atlas->uploadsPending();
}

const TerrainAtlas::Stats& RenderScene::terrainAtlasStats() const
{
    return atlas->stats();
}

const TerrainPass::Stats& RenderScene::terrainPassStats() const
{
    return terrain->stats();
}

std::vector<float> RenderScene::readDepthForTests(int& w, int& h)
{
    return renderer->readDepth(w, h);
}

void RenderScene::setFoliageTexture(uint32_t foliageSeed)
{
    SAIGA_ASSERT(!transferFn, "setFoliageTexture: not inside renderFrame");
    // The frames in flight may use the previous texture
    renderer->waitIdle();
    culling->setFoliageTexture(nullptr);
    foliage = std::make_unique<Texture2DArray>();
    {
        auto layers = FoliageGenerator::foliageTextures(512, foliageSeed);
        std::vector<const Image*> images;
        for (auto& l : layers) images.push_back(&l);
        foliage->fromImages(base(), images, vk::ImageUsageFlagBits::eSampled, false, true);
    }
    culling->setFoliageTexture(foliage.get());
}

void RenderScene::setPrototypeSettings(const PrototypeSettings& settings)
{
    *protoSettings = settings;
}

const PrototypeSettings& RenderScene::prototypeSettings() const
{
    return *protoSettings;
}

void RenderScene::setSpaceSettings(SpaceId space, const SpaceRenderSettings& settings)
{
    // A copy first: settings may be an element of spaceRenderSettings (spaceSettings(other)), which the resize can
    // move (PR review, S21)
    const SpaceRenderSettings copy = settings;
    if (spaceRenderSettings.size() <= space) spaceRenderSettings.resize(size_t(space) + 1);
    spaceRenderSettings[space] = copy;
}

const SpaceRenderSettings& RenderScene::spaceSettings(SpaceId space) const
{
    static const SpaceRenderSettings openWorld;
    return space < spaceRenderSettings.size() ? spaceRenderSettings[space] : openWorld;
}

void RenderScene::frameLighting(const SceneCamera& camera, const PrototypeSettings& proto, bool enclosed, int samples,
                                SceneLighting& out)
{
    out = SceneLighting();
    // The fog's base: the terrain set's sea level (the new path's environment, S20)
    const std::optional<int64_t> sea = atlas->seaLevel();
    const double fogBase             = sea ? double(*sea) * kHeightUnit : 0.0;
    // The foliage's alpha cutoff, the camera, the trees' wind (the environment's phases: no absolute time in a shader,
    // §10.1; 0 without an environment) and the debug tint
    const vec3 p      = camera.position.cast<float>();
    out.foliageParams = vec4(proto.alphaCutoff, 0, 0, 0);
    out.camera        = vec4(p(0), p(1), p(2), float(samples));
    out.wind  = envSet ? vec4(env.phases[0], env.phases[1], env.phases[2], proto.wind) : vec4(0, 0, 0, proto.wind);
    out.debug = vec4(float(proto.debugMode), 0, 0, 0);

    const bool physical = physicalLighting();
    out.mode            = uvec4(physical ? 1u : 0u, enclosed ? 1u : 0u, 0u, 0u);
    // x: the physical exposure (also in a debug frame, which latched Identity: the views' display colours)
    const float displayExposure = physical ? exposureFromEv100(env.ev100, params.exposure) : 1.0f;
    out.exposure = vec4(displayExposure, framePreExposure, envSet ? env.ev100 : 0.0f, sceneWritesLinear() ? 1 : 0);
    out.mode(2)  = uint32_t(frameDebug.view);
    out.mode(3)  = frameDebug.overlays;
    const float centre = frameExposureViewCentre();
    // w: the probe's raw radiance frames write radiance times frame 1's pre-exposure (fp16: a glossy highlight in
    // sunlight exceeds 65504 cd/m^2), finishProbe divides it out
    const float rawScale           = frameDebugRaw && probeRun ? probeRun->preExposure : 1.0f;
    out.debugParams                = vec4(centre, exposureViewRange, frameDebugRaw ? 1.0f : 0.0f, rawScale);
    const StudioEnvironment studio = studioEnvironment(enclosed);
    const vec3 sd                  = studio.sunDir.normalized();
    out.studioSun                  = vec4(sd(0), sd(1), sd(2), studio.sunIntensity);
    // Clay (and a scene without an environment): the studio sun is the shadow light of the cascades (S36a2); the
    // physical path chooses it below
    if (!physical) out.shadowLight = vec4(sd(0), sd(1), sd(2), 1);
    out.studioShading           = vec4(studio.ambient, studio.hemisphere ? 1 : 0, studio.albedo, 0);
    const float cameraAboveBase = float(camera.position.y() - fogBase);
    if (!physical)
    {
        out.fogParams   = vec4(studio.fogDensity, studio.fogFalloff, studio.fog ? 1 : 0, cameraAboveBase);
        out.fogColorLin = vec4(studio.fogColor(0), studio.fogColor(1), studio.fogColor(2), 1);
    }
    if (envSet)
    {
        const EnvironmentState& e = env;
        out.sun                   = vec4(e.sunDir(0), e.sunDir(1), e.sunDir(2), e.sunIlluminance);
        out.moon                  = vec4(e.moonDir(0), e.moonDir(1), e.moonDir(2), e.moonIlluminance);
        // The shadow light (D13): the body with the higher illuminance on the ground (an air-mass model of the clear
        // atmosphere), with a hysteresis of a factor 2 (S36a2)
        auto elevationOf = [](const vec3& d)
        { return std::asin(std::clamp(double(d(1)) / std::max(double(d.norm()), 1e-9), -1.0, 1.0)); };
        const double sunGround  = CascadedShadows::groundIlluminance(e.sunIlluminance, elevationOf(e.sunDir));
        const double moonGround = CascadedShadows::groundIlluminance(e.moonIlluminance, elevationOf(e.moonDir));
        if (physical)
        {
            shadowLightChoice = CascadedShadows::selectShadowLight(shadowLightChoice, sunGround, moonGround);
            if (shadowLightChoice == 1)
                out.shadowLight = vec4(e.sunDir(0), e.sunDir(1), e.sunDir(2), 1);
            else if (shadowLightChoice == 2)
                out.shadowLight = vec4(e.moonDir(0), e.moonDir(1), e.moonDir(2), 2);
        }
        // The moon in daylight adds about 1e-6 of the sun's light: no transmittance lookup and GGX for it (the sky's
        // LUTs and the moon disc take the environment's values)
        if (moonGround < 1e-4 * sunGround) out.moon(3) = 0;
        out.minAmbient      = vec4(e.minAmbient(0), e.minAmbient(1), e.minAmbient(2), e.ambientScale);
        out.enclosedAmbient = vec4(e.enclosedAmbient(0), e.enclosedAmbient(1), e.enclosedAmbient(2), 0);
        // The cloud layer: the cloudiness through the noise's histogram (the layer's mean coverage follows the
        // weather's cloudiness)
        out.clouds = vec4(e.clouds.altitude, e.clouds.radius,
                          clouds ? clouds->effectiveCloudiness(e.cloudiness) : e.cloudiness, e.clouds.shadowFade);
        // The coverage texture's position of the camera (double: the camera's xz and the wrapped offset)
        auto wrap = [](double v)
        {
            const double t = std::fmod(v, CloudLayer::kTile);
            return float(t < 0 ? t + CloudLayer::kTile : t);
        };
        const bool cloudsOn = physical && params.lighting.clouds && !e.enclosed && !enclosed && e.cloudiness > 0;
        out.cloudMotion     = vec4(wrap(camera.position.x() + double(e.cloudOffset(0))),
                                   wrap(camera.position.z() + double(e.cloudOffset(1))),
                               clouds ? clouds->meanCoverage(out.clouds(2)) : 0.0f, cloudsOn ? 1.0f : 0.0f);
        for (int r = 0; r < 3; ++r)
            out.starRotation[r] = vec4(e.starRotation(r, 0), e.starRotation(r, 1), e.starRotation(r, 2), 0);
        out.phases[0] = vec4(e.phases[0], e.phases[1], e.phases[2], e.phases[3]);
        out.phases[1] = vec4(e.phases[4], e.phases[5], e.phases[6], e.phases[7]);
        if (physical)
        {
            // The height fog with the weather's density (rain lowers the visibility: a density multiplier) and
            // CloudSettings::fogFalloff; its colour is the sky-view LUT at the horizon (lighting.glsl)
            const float density = e.fogDensity * (1.0f + kRainFogFactor * e.rain);
            const bool on       = !e.enclosed && density > 0;
            out.fogParams       = vec4(density, e.clouds.fogFalloff, on ? 1 : 0, cameraAboveBase);
            out.fogColorLin     = vec4(0, 0, 0, 1);
        }
        // The stars fade out with the sun's elevation from -10 to -4 degrees (the twilight sky outshines them; no star
        // loop per sky pixel in daylight; review minor 3); the full moon's illuminance for the disc (minor 4)
        const float sunElevationDeg = float(std::asin(std::clamp(double(e.sunDir(1)), -1.0, 1.0)) * 57.29577951308232);
        const float t               = std::clamp((-4.0f - sunElevationDeg) / 6.0f, 0.0f, 1.0f);
        out.starRotation[0](3)      = e.starIntensity * t * t * (3 - 2 * t);
        out.starRotation[1](3)      = e.moonFullIlluminance;
    }
    // The prototype albedo colours (D21)
    const PrototypeAlbedo& a = proto.colors;
    auto rgb                 = [](const vec3& v, float w) { return vec4(v(0), v(1), v(2), w); };
    out.albedoGround         = rgb(a.ground, a.rockSlope);
    out.albedoRock           = rgb(a.rock, 0);
    out.albedoShore          = rgb(a.shore, 0);
    out.albedoWater          = rgb(a.water, 0);
    out.albedoGrass          = rgb(a.grass, 0);
    out.albedoLeaves         = rgb(a.leaves, 0);
    out.albedoBark           = rgb(a.bark, 0);
    out.sea                  = vec4(sea ? float(camera.position.y() - fogBase) : 0.0f, sea ? 1.0f : 0.0f, 0, 0);
    // The camera height above the sea level for the atmosphere (S36a1c; without a sea level: y = 0); the environment
    // cube's validity is set with the bindings (writeLightingSet)
    out.atmosphere = vec4(float(camera.position.y() - fogBase), 0, float(EnvironmentMap::kMips - 1), 0);
}

void RenderScene::frameShadows(const SceneCamera& camera, bool enclosed, SceneLighting& block, uint32_t frameIndex)
{
    shadowViews.clear();
    frameCascades               = false;
    const LightingParameters& p = params.lighting;
    // The camera's speed over the last second (the staggered cascades' margin) and the frame time, on the frames' clock
    // (setFrameTime, else the wall clock's step)
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    frameClock += nextFrameDt ? *nextFrameDt : lastWallClock < 0 ? 0.0 : wall - lastWallClock;
    nextFrameDt.reset();
    lastWallClock    = wall;
    const double now = frameClock;
    // A jump of more than 500 m in a frame is a teleport, not a speed (the cascades refit when the camera leaves their
    // margin, CascadedShadows::fit)
    if (camera.valid && !cameraHistory.empty() && (camera.position - cameraHistory.back().second).norm() > 500)
        cameraHistory.clear();
    // One sample per time (frames with a zero step, e.g. a paused clock, keep the newest position: the speed's dt is
    // the span over the samples)
    if (camera.valid && !cameraHistory.empty() && cameraHistory.back().first == now)
        cameraHistory.back().second = camera.position;
    else if (camera.valid)
        cameraHistory.emplace_back(now, camera.position);
    while (!cameraHistory.empty() && now - cameraHistory.front().first > 1.0) cameraHistory.pop_front();
    // The mean speed over the last second (a debug camera's jump of a few metres in one frame is not a speed; a move
    // beyond a staggered cascade's margin refits it anyway)
    double maxSpeed = 50, dt = 1.0 / 60.0;
    if (cameraHistory.size() >= 2)
    {
        const double span = cameraHistory.back().first - cameraHistory.front().first;
        dt                = span / double(cameraHistory.size() - 1);
        double path       = 0;
        for (size_t i = 1; i < cameraHistory.size(); ++i)
            path += (cameraHistory[i].second - cameraHistory[i - 1].second).norm();
        if (span > 0.05) maxSpeed = std::max(maxSpeed, path / span);
    }
    const bool on = p.sunShadows && camera.valid && !enclosed && !(envSet && env.enclosed) && block.shadowLight(3) > 0;
    if (!on)
    {
        // No shadow light (or off): no cascade, the factor is 1; the images are released while shadows are off
        cascades->invalidate();
        if (!p.sunShadows && cascades->hasImages()) cascades->releaseImages();
        block.shadowParams =
            vec4(0, p.depthBiasTexels, p.normalOffsetTexels, lightingSet->linearShadowFilter() ? 1 : 0);
        return;
    }
    const int count = std::clamp(p.cascadeCount, 1, kMaxCascades);
    const int res   = CascadedShadows::clampedResolution(p);
    cascades->ensureImages(count, res);
    CascadeInput in;
    in.camera         = camera.position;
    in.cameraRotation = cameraRotation.matrix();
    in.fovy           = cameraFovy;
    const ivec2 sz    = size();
    in.aspect         = sz(1) > 0 ? double(sz(0)) / double(sz(1)) : 1.0;
    in.nearPlane      = params.nearPlane;
    in.towardsLight   = Vec3(block.shadowLight(0), block.shadowLight(1), block.shadowLight(2));
    in.frame          = frameCount + 1;
    in.maxSpeed       = maxSpeed;
    in.dt             = dt;
    cascades->fit(in, p, freezeCascades);
    // The light the cascades were fitted to (in steps)
    const double elevation = std::asin(std::clamp(cascades->fittedLight()(1), -1.0, 1.0));
    frameCascades          = true;
    for (int c = 0; c < count; ++c)
    {
        const CascadedShadows::Cascade& k = cascades->cascades()[size_t(c)];
        ShadowView v                      = k.valid ? cascades->view(c, camera.position, p, elevation) : ShadowView();
        block.cascadeMatrix[c]            = v.viewProj;
        // The texel, the band and the edge margin (fractions of the half extent), valid
        block.cascadeParams[c] = vec4(float(k.texelSize), float(CascadedShadows::blendBand(p, res)),
                                      k.valid ? 1.0f : 0.0f, float(CascadedShadows::edgeMargin(p, res)));
        if (!k.updated)
        {
            // Not rendered this frame: an empty volume (the culling rejects everything)
            v.rectMin = Vec2(1e30, 1e30);
            v.rectMax = Vec2(-1e30, -1e30);
        }
        else
            slotStatistics[frameIndex].cascadeMask |= 1u << c;
        shadowViews.push_back(v);
    }
    block.shadowParams =
        vec4(float(count), p.depthBiasTexels, p.normalOffsetTexels, lightingSet->linearShadowFilter() ? 1 : 0);
}

const std::vector<FrameLightInfo>& RenderScene::frameLights() const
{
    return lightTable->frameLights();
}

void RenderScene::frameLocalLights(const SceneCamera& camera, SceneLighting& block, uint32_t frameIndex)
{
    const LightingParameters& p = params.lighting;
    LocalLights::FrameInput in;
    in.valid  = camera.valid;
    in.space  = camera.space;
    in.camera = camera.position;
    for (size_t k = 0; k < in.planes.size(); ++k) in.planes[k] = camera.planes[k];
    in.on           = p.localLights;
    in.maxLights    = p.maxLights;
    in.clusterFar   = std::max(double(p.clusterFar), 1.0);
    in.flickerPhase = envSet ? env.phases[6] : 0.0f;
    in.highlight    = frameDebug.highlightLight;
    lightTable->prepareFrame(frameIndex, in);
    const uint32_t count = uint32_t(lightTable->frameRecords().size());
    // The clusters touched by each light (the newest completed frame's)
    for (FrameLightInfo& f : lightTable->frameLightsMutable()) f.clusters = clusterCull->clustersOf(f.handle);
    // The cluster grid of the frame (B3): computed whenever the frame has lights (also with the brute-force loop: the
    // ClusterLights view and the statistics)
    ClusterGrid& g        = *clusterGrid;
    g.x                   = std::clamp(p.clusterX, 1, 64);
    g.y                   = std::clamp(p.clusterY, 1, 64);
    g.z                   = std::clamp(p.clusterZ, 2, 128);
    g.maxLightsPerCluster = std::clamp(p.maxLightsPerCluster, 1, 1024);
    g.nearPlane           = double(params.nearPlane);
    g.clusterFar          = std::max(double(p.clusterFar), 2.0 * double(params.nearPlane));
    g.clusterNear         = std::clamp(double(p.clusterNear), double(params.nearPlane) * 1.001, g.clusterFar * 0.999);
    const ivec2 sz        = size();
    g.width               = std::max(sz(0), 1);
    g.height              = std::max(sz(1), 1);
    g.fovy                = cameraFovy;
    g.aspect              = double(g.width) / double(g.height);
    frameClusters         = camera.valid && count > 0;
    if (!frameClusters) clusterCull->clearStatistics();
    const bool clustered  = frameClusters && p.clusteredLights;
    block.lightCounts     = uvec4(count, clustered ? 1u : 0u, uint32_t(lightTable->highlightIndex() + 1), 0);
    block.clusterGrid     = uvec4(uint32_t(g.x), uint32_t(g.y), uint32_t(g.z), uint32_t(g.maxLightsPerCluster));
    block.clusterDepth    = vec4(float(g.nearPlane), float(g.clusterNear), float(g.clusterFar),
                                 float(double(g.z - 1) / std::log(g.clusterFar / g.clusterNear)));
    block.atlasParams(3)  = clusterHeatScale;
    // The lights' cluster ranges (light_cluster.comp's pre-test)
    if (frameClusters)
    {
        // The slot's cluster buffers for this grid before writeLightingSet binds them
        clusterCull->prepare(frameIndex, g, count);
        for (LocalLights::GpuLight& r : lightTable->frameRecordsMutable())
            r.clusterRange = LightClusters::clusterRange(g, frameView, frameProjection, r);
        lightTable->writeRecords(frameIndex);
    }
    const Vec3 forward = cameraRotation * Vec3(0, 0, -1);
    block.viewForward  = vec4(float(forward(0)), float(forward(1)), float(forward(2)), 0);
    block.viewport     = vec4(float(sz(0)), float(sz(1)), 0, 0);
}

void RenderScene::frameAtlas(vk::CommandBuffer cmd, const SceneCamera& camera, bool enclosed, SceneLighting& block,
                             uint32_t frameIndex)
{
    atlasFaces.clear();
    const LightingParameters& p = params.lighting;
    // The atlas exists while the local lights are on (its image is released otherwise)
    if (!p.localLights)
    {
        if (atlasShadows->hasImage()) atlasShadows->release();
        std::vector<ShadowAtlas::Face> none;
        atlasShadows->update({}, camera.position, 0, frameCount + 1, none);
        return;
    }
    int res  = std::clamp(p.atlasResolution, 512, 8192);
    int tile = std::clamp(p.atlasTile, 64, res);
    if (res % tile != 0) tile = res / std::max(res / tile, 1);
    // The frame's shadowed lights (importance order)
    std::vector<ShadowAtlas::LightInput> in;
    for (const FrameLightInfo& f : lightTable->frameLights())
    {
        const LocalLights::Light* l = lightTable->find(LightHandle{f.handle});
        if (!l || !l->data.castShadows) continue;
        ShadowAtlas::LightInput li;
        li.handle     = f.handle;
        li.index      = int(f.index);
        li.world      = l->worldPosition();
        li.data       = l->data;
        li.importance = f.importance;
        in.push_back(li);
    }
    // The image on first use (a frame with a shadowed light; kept afterwards while the local lights are on)
    if (!in.empty()) atlasShadows->ensure(res, tile, cmd);
    if (!atlasShadows->hasImage())
    {
        std::vector<ShadowAtlas::Face> none;
        atlasShadows->update({}, camera.position, 0, frameCount + 1, none);
        block.lightCounts(3) = 0;
        return;
    }
    // The tiles of removed lights and of lights without castShadows are freed (other cached tiles stay until needed)
    atlasShadows->retain(
        [this](uint32_t h)
        {
            const LocalLights::Light* l = lightTable->find(LightHandle{h});
            return l && l->data.castShadows;
        });
    // Moving dynamic instances in a light's sphere: the faces whose frustum contains one render their dynamic layer
    // every frame (resting ones are in the static layer)
    if (!in.empty())
        for (const InstanceBuffer::Dynamic& d : instanceBuffer->dynamics())
        {
            if (!d.alive || !d.set || d.chunk.space != camera.space || !InstanceBuffer::moving(d)) continue;
            const MeshPool::Mesh* m = meshPool->mesh(d.mesh);
            if (!m) continue;
            const double scale = d.curr.scale;
            const Vec3 centre =
                chunkOrigin(d.chunk) + d.curr.position + d.curr.rotation * (m->sphereCenter.cast<double>() * scale);
            const double radius = double(m->sphereRadius + m->boundsPadding) * scale;
            for (ShadowAtlas::LightInput& li : in)
                if ((centre - li.world).norm() <= radius + double(li.data.range))
                    li.movers.emplace_back(centre, radius);
        }
    // The terrain under each light at the finest resident level (its key: a finer level that becomes resident changes
    // it); none in an enclosed space
    std::vector<std::pair<Vec3, double>> spheres;
    for (const ShadowAtlas::LightInput& li : in)
        spheres.emplace_back(li.world - camera.position, double(li.data.range));
    std::vector<uint64_t> keys;
    if (!enclosed && camera.valid)
        terrain->selectForAtlas(frameIndex, *atlas, camera, spheres, keys);
    else
        keys.assign(in.size(), 0);
    for (size_t i = 0; i < in.size(); ++i) in[i].terrainKey = i < keys.size() ? keys[i] : 0;
    // The faces of this frame: views of the culling after the cascades' (the 16-bit view masks: at most 11 atlas views)
    const int freeViews = kMaxCullViews - 1 - int(shadowViews.size());
    // Golden mode (forceFullUpdates, §17.6; nit 25): every due face up to the 11 views of a frame
    const int faces = std::clamp(std::min(p.forceFullUpdates ? 11 : p.facesPerFrame, freeViews), 0, 11);
    std::vector<ShadowAtlas::Face> list;
    atlasShadows->update(in, camera.position, shadowViewsOverride ? 0 : faces, frameCount + 1, list);
    // The faces are recorded in this frame's shadow section, before the scene pass reads them: rendered for the table
    atlasShadows->rendered(list, frameCount + 1);
    for (const ShadowAtlas::Face& f : list)
    {
        AtlasFace af;
        af.face = f;
        if (!f.clearOnly)
        {
            // The static layer: static sets, the terrain and the resting dynamic instances (cached); the dynamic layer:
            // the moving dynamic instances
            ShadowView v;
            v.perspective   = true;
            v.facePlanes    = f.planes;
            v.castStatic    = f.layer == 0;
            v.castDynamic   = true;
            v.dynamicFilter = f.layer == 0 ? 1 : 2;
            v.viewProj      = f.viewProj;
            // The instances' LOD and end distance from the light (a cached face is camera-independent)
            v.lodFromOrigin = true;
            v.lodOrigin     = Vec3(f.lightPush(0), f.lightPush(1), f.lightPush(2));
            shadowViews.push_back(v);
            af.view = uint32_t(shadowViews.size());
        }
        atlasFaces.push_back(af);
    }
    // The tile table and the lights' shadow fields
    std::vector<int> first;
    std::vector<bool> ready;
    atlasShadows->writeTable(frameIndex, in, camera.position, first, ready);
    std::vector<FrameLightInfo>& frame = lightTable->frameLightsMutable();
    for (size_t i = 0; i < in.size(); ++i)
    {
        lightTable->setShadow(size_t(in[i].index), first[i], ready[i]);
        FrameLightInfo& fi           = frame[size_t(in[i].index)];
        const std::vector<int> tiles = atlasShadows->tilesOf(in[i].handle);
        fi.tiles                     = first[i] >= 0 ? int(tiles.size()) : 0;
        fi.firstTile                 = first[i] >= 0 && !tiles.empty() ? tiles[0] : -1;
        fi.shadowReady               = ready[i];
        uint64_t last                = 0;
        for (int t : tiles) last = std::max(last, atlasShadows->tiles()[size_t(t)].staticFrame);
        fi.lastRender = last;
    }
    lightTable->writeRecords(frameIndex);
    block.lightCounts(3) = atlasShadows->hasImage() ? (lightingSet->linearAtlasFilter() ? 2u : 1u) : 0u;
    block.atlasParams =
        vec4(float(atlasShadows->resolution()), p.atlasDepthBias, p.atlasNormalOffset, clusterHeatScale);
}

const ClusterGrid& RenderScene::frameClusterGrid() const
{
    return *clusterGrid;
}

void RenderScene::clusterOverlays()
{
    if (!(frameDebug.overlays & OverlayClusterGrid) || !cameraSet) return;
    const ClusterGrid& g = *clusterGrid;
    // The camera of the grid: the frozen camera of the cascades while they are frozen, else this frame's
    Vec3 eye    = cameraPosition;
    Mat3 R      = cameraRotation.matrix();
    double fovy = cameraFovy;
    if (freezeCascades && cascades->fittedCamera().fovy > 0)
    {
        eye  = cascades->fittedCamera().camera;
        R    = cascades->fittedCamera().cameraRotation;
        fovy = cascades->fittedCamera().fovy;
    }
    const double tanY = std::tan(fovy * 0.5), tanX = tanY * g.aspect;
    // Pixel row 0 at the top: +y in view space
    auto point = [&](double sx, double sy, double d) { return eye + R * Vec3(sx * tanX * d, -sy * tanY * d, -d); };
    for (int k = 0; k <= g.z; ++k)
    {
        double d0, d1;
        g.sliceDepth(std::min(k, g.z - 1), d0, d1);
        const double d = k == g.z ? d1 : d0;
        // Near slices cyan, far ones orange
        const float t  = float(k) / float(g.z);
        const vec3 col = k == 0 ? vec3(1, 1, 1) : vec3(0.2f + 0.8f * t, 0.9f - 0.4f * t, 1.0f - 0.8f * t);
        // The slice plane's tile lines
        for (int i = 0; i <= g.x; ++i)
        {
            const double sx = -1 + 2.0 * i / g.x;
            lines->addLine(point(sx, -1, d), point(sx, 1, d), col, false);
        }
        for (int j = 0; j <= g.y; ++j)
        {
            const double sy = -1 + 2.0 * j / g.y;
            lines->addLine(point(-1, sy, d), point(1, sy, d), col, false);
        }
    }
    // The frustum's edges to clusterFar
    for (int c = 0; c < 4; ++c)
    {
        const double sx = (c & 1) ? 1 : -1, sy = (c & 2) ? 1 : -1;
        lines->addLine(point(sx, sy, g.nearPlane), point(sx, sy, g.clusterFar), vec3(1, 1, 1), false);
    }
}

void RenderScene::lightOverlays()
{
    if (!(frameDebug.overlays & OverlayLightVolumes)) return;
    const LocalLights& t                    = *lightTable;
    const std::vector<FrameLightInfo>& list = t.frameLights();
    const double kTwoPi                     = 6.283185307179586;
    // At most 512 lights (the most important ones), 24 segments per circle
    const size_t n = std::min<size_t>(list.size(), 512);
    for (size_t i = 0; i < n; ++i)
    {
        const LocalLights::Light* l = t.find(LightHandle{list[i].handle});
        if (!l) continue;
        const LightData& d     = l->data;
        const bool highlighted = int(i) == t.highlightIndex();
        const float m          = std::max({d.color(0), d.color(1), d.color(2), 1e-6f});
        const vec3 color       = highlighted ? vec3(1, 1, 1) : vec3(d.color / m);
        const Vec3 c           = l->worldPosition();
        const double r         = double(d.range);
        // Shadowed lights thick (a second copy slightly larger), the highlighted light on top
        const int copies = d.castShadows || highlighted ? 2 : 1;
        auto circle      = [&](const Vec3& centre, const Vec3& u, const Vec3& v, double radius)
        {
            for (int copy = 0; copy < copies; ++copy)
            {
                const double rr = radius * (1.0 + 0.01 * copy);
                for (int k = 0; k < 24; ++k)
                {
                    const double a0 = kTwoPi * k / 24, a1 = kTwoPi * (k + 1) / 24;
                    lines->addLine(centre + (u * std::cos(a0) + v * std::sin(a0)) * rr,
                                   centre + (u * std::cos(a1) + v * std::sin(a1)) * rr, color, highlighted);
                }
            }
        };
        if (d.kind == LightKind::Point)
        {
            circle(c, Vec3(1, 0, 0), Vec3(0, 1, 0), r);
            circle(c, Vec3(1, 0, 0), Vec3(0, 0, 1), r);
            circle(c, Vec3(0, 1, 0), Vec3(0, 0, 1), r);
        }
        else
        {
            // The cone to the range: its end circle and four edges from the apex
            const Vec3 axis   = d.direction.cast<double>().normalized();
            const Vec3 helper = std::abs(axis(1)) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
            const Vec3 u      = axis.cross(helper).normalized();
            const Vec3 v      = axis.cross(u);
            const double co   = std::clamp(double(d.cosOuter), 0.0, 1.0);
            const double so   = std::sqrt(1 - co * co);
            const Vec3 end    = c + axis * (r * co);
            const double rad  = r * so;
            circle(end, u, v, rad);
            for (int k = 0; k < 4; ++k)
            {
                const double a = kTwoPi * k / 4;
                lines->addLine(c, end + (u * std::cos(a) + v * std::sin(a)) * rad, color, highlighted);
            }
        }
        if (highlighted)
        {
            const double s = std::max(0.25, 0.05 * r);
            lines->addLine(c - Vec3(s, 0, 0), c + Vec3(s, 0, 0), color, true);
            lines->addLine(c - Vec3(0, s, 0), c + Vec3(0, s, 0), color, true);
            lines->addLine(c - Vec3(0, 0, s), c + Vec3(0, 0, s), color, true);
        }
    }
}

void RenderScene::requestHorizonRecompute(bool full)
{
    horizon->requestRecompute(full);
}

void RenderScene::cascadeOverlays()
{
    const uint32_t overlays = frameDebug.overlays;
    // The horizon map's extent and its inner half (the anchor moves when the camera leaves it), below the camera
    if ((overlays & OverlayHorizonAnchor) && horizon->hasImages() && cameraSet)
    {
        const HorizonShadows::MapTransform& t = horizon->transform(horizon->frontLayer());
        if (t.valid)
        {
            const double y = cameraPosition.y() - 20;
            auto square    = [&](double x0, double z0, double size, const vec3& color)
            {
                const Vec3 c[4] = {Vec3(x0, y, z0), Vec3(x0 + size, y, z0), Vec3(x0 + size, y, z0 + size),
                                   Vec3(x0, y, z0 + size)};
                for (int e = 0; e < 4; ++e) lines->addLine(c[e], c[(e + 1) % 4], color, true);
            };
            square(t.cornerX, t.cornerZ, t.extent, vec3(0.9f, 0.5f, 1.0f));
            square(t.cornerX + 0.25 * t.extent, t.cornerZ + 0.25 * t.extent, 0.5 * t.extent, vec3(0.6f, 0.35f, 0.7f));
        }
    }
    if (!frameCascades || !(overlays & (OverlayCascadeBounds | OverlayCameraFrustumSplits))) return;
    const CascadedShadows& cs = *cascades;
    if (overlays & OverlayCascadeBounds)
    {
        for (int c = 0; c < cs.count(); ++c)
        {
            const CascadedShadows::Cascade& k = cs.cascades()[size_t(c)];
            if (!k.valid) continue;
            const vec3 color = cascadeColor(c);
            auto corner      = [&](double sx, double sy, double z) -> Vec3
            { return k.axisX * (k.cx + sx * k.halfExtent) + k.axisY * (k.cy + sy * k.halfExtent) + k.axisZ * z; };
            // The receivers' box (the shared depth range) and the instances' caster volume towards the light (the
            // cap the culling used, CascadedShadows::Cascade::casterTop; nit 22: no copy of its formula)
            const double zs[3] = {k.zFar, k.zNear, k.casterTop};
            for (int layer = 0; layer < 3; ++layer)
            {
                const vec3 col = layer == 2 ? vec3(color * 0.5f) : color;
                for (int e = 0; e < 4; ++e)
                {
                    const double ax[4] = {-1, 1, 1, -1}, ay[4] = {-1, -1, 1, 1};
                    lines->addLine(corner(ax[e], ay[e], zs[layer]), corner(ax[(e + 1) % 4], ay[(e + 1) % 4], zs[layer]),
                                   col, true);
                    if (layer > 0)
                        lines->addLine(corner(ax[e], ay[e], zs[layer - 1]), corner(ax[e], ay[e], zs[layer]), col, true);
                }
            }
        }
    }
    if (overlays & OverlayCameraFrustumSplits)
    {
        // The split planes of the camera the cascades were fitted to (frozen: the frozen camera)
        const CascadeInput& in           = cs.fittedCamera();
        const std::vector<double> splits = CascadedShadows::cascadeSplits(in.nearPlane, params.lighting.shadowDistance,
                                                                          cs.count(), params.lighting.splitLambda);
        const double tanY = std::tan(in.fovy * 0.5), tanX = tanY * in.aspect;
        auto point = [&](double sx, double sy, double d) -> Vec3
        { return in.camera + in.cameraRotation * Vec3(sx * tanX * d, sy * tanY * d, -d); };
        const double ax[4] = {-1, 1, 1, -1}, ay[4] = {-1, -1, 1, 1};
        for (size_t i = 0; i < splits.size(); ++i)
        {
            const vec3 color = cascadeColor(int(std::min<size_t>(i, splits.size() - 2)));
            for (int e = 0; e < 4; ++e)
            {
                lines->addLine(point(ax[e], ay[e], splits[i]), point(ax[(e + 1) % 4], ay[(e + 1) % 4], splits[i]),
                               color, true);
                if (i > 0)
                    lines->addLine(point(ax[e], ay[e], splits[i - 1]), point(ax[e], ay[e], splits[i]),
                                   cascadeColor(int(i) - 1), true);
            }
        }
    }
}

StudioEnvironment RenderScene::studioEnvironment(bool enclosed) const
{
    if (studioOverride) return *studioOverride;
    // An enclosed space: the prototype preset without fog (S21's enclosed light)
    StudioEnvironment s = prototypeStudio();
    if (enclosed)
    {
        s.fog = false;
        return s;
    }
    if (!foliage) return sceneStudio();
    s.albedo = protoSettings->albedo;
    return s;
}

void RenderScene::writeLightingSet(vk::CommandBuffer cmd, uint32_t frameIndex, const SceneLighting& frameBlock,
                                   bool terrainDrawn)
{
    ambient->update(cmd, frameIndex, env);
    SceneLightingBindings b;
    b.ambient = ambient->bindings(frameIndex);
    if (foliage) b.foliage = foliage->getDescriptorInfo();
    // (the dummies while the environment's resources do not exist: a clay scene)
    if (brdf) b.brdfLut = brdf->descriptorInfo();
    if (atmosphere)
    {
        b.transmittanceLut = atmosphere->transmittanceInfo();
        b.skyViewLut       = atmosphere->skyViewInfo();
        b.cloudCoverage    = clouds->coverageInfo();
    }
    if (cascades->hasImages()) b.cascades = cascades->depth()->getDescriptorInfo(lightingSet->shadowSampler());
    // S36a2: the horizon maps (the state of the previous frame's update: this frame's update writes only the back map)
    SceneLighting block = frameBlock;
    if (params.lighting.horizonShadows && horizon->hasImages())
    {
        b.horizon = horizon->descriptor();
        // G (the cascades shadow the near terrain) only when the cascade pass draws the terrain (not with the
        // prototype's terrain off; CodeRabbit #85 round 2)
        horizon->fillBlock(block, cameraPosition, cameraSpace, params.lighting, block.shadowLight(3) > 0, terrainDrawn);
    }
    else
        horizon->fillBlock(block, cameraPosition, cameraSpace, LightingParameters(), false, false);
    block.atmosphere(1) = ambient->usesEnvironmentMap() ? 1.0f : 0.0f;
    block.atmosphere(3) = atmosphere ? atmosphere->skyViewHorizonElevation() : 0.0f;
    // S36b: the frame's lights
    b.lights         = lightTable->bufferInfo(frameIndex);
    b.clusterCounts  = clusterCull->tableInfo(frameIndex);
    b.clusterIndices = clusterCull->indexInfo(frameIndex);
    b.atlasTiles     = atlasShadows->tableInfo(frameIndex);
    if (atlasShadows->hasImage()) b.atlas = atlasShadows->texture()->getDescriptorInfo(lightingSet->atlasSampler());
    lightingSet->write(frameIndex, block, b);
    lastBlock = block;

    // The statistics of this frame (S36a1b)
    LightingStatistics& st = lightStats;
    st.frame               = frameCount + 1;
    st.shadowLight         = int(block.shadowLight(3));
    auto elevation         = [](const vec3& d)
    { return float(std::asin(std::clamp(double(d(1)), -1.0, 1.0)) * 57.29577951308232); };
    st.sunElevation      = envSet ? elevation(env.sunDir) : 0.0f;
    st.moonElevation     = envSet ? elevation(env.moonDir) : 0.0f;
    st.ev100             = envSet ? env.ev100 : 0.0f;
    st.exposure          = frameEffectiveExposure;
    st.preExposure       = framePreExposure;
    st.physical          = framePhysical;
    st.debug             = frameDebug;
    st.sh                = ambient->statistics().sh;
    st.environmentMap    = envMap ? envMap->statistics() : EnvironmentMapStatistics();
    st.lutComputations   = atmosphere ? atmosphere->lutComputations() : 0;
    st.clouds            = block.cloudMotion(3) > 0.5f;
    st.cloudMeanCoverage = block.cloudMotion(2);
    st.luminance         = debugImages->statistics(LightingImage::Luminance);
    st.luminanceFrame    = debugImages->statisticsFrame(LightingImage::Luminance);
    // The block and SH buffers per frame in flight, the BRDF LUT (RG16F), the previews, the debug lines, the LUTs, the
    // environment map and the clouds (not the dummy textures)
    st.memoryBytes = VulkanRenderer::framesInFlight * (sizeof(SceneLighting) + SkyAmbient::kShBytes) +
                     (brdf ? uint64_t(BrdfLut::kSize) * BrdfLut::kSize * 4 : 0) + debugImages->memoryBytes() +
                     lines->memoryBytes() + (atmosphere ? atmosphere->memoryBytes() : 0) +
                     (envMap ? envMap->memoryBytes() : 0) + (clouds ? clouds->memoryBytes() : 0) +
                     cascades->memoryBytes() + horizon->memoryBytes() + lightTable->statistics().memoryBytes +
                     clusterCull->statistics().memoryBytes + atlasShadows->memoryBytes();
    st.horizon             = horizon->statistics();
    st.horizonPreview = st.horizon;
    st.lights              = lightTable->statistics();
    st.atlas                       = atlasShadows->statistics();
    st.atlas.terrainNodes          = uint32_t(terrain->atlasNodeCount());
    st.atlas.terrainNodesTruncated = uint32_t(terrain->atlasNodesTruncated());
    st.lights.withoutTiles = st.atlas.lightsWithoutTiles;
    {
        const LightClusters::Statistics& cs = clusterCull->statistics();
        st.lights.clusterCount              = cs.clusters;
        st.lights.clusterMax                = cs.maxLights;
        st.lights.clusterMean               = cs.meanLights;
        st.lights.overflowedClusters        = cs.overflowed;
        st.lights.droppedLights             = cs.droppedLights;
        st.lights.clusterFrame              = cs.frame;
        st.lights.clusterMemory             = cs.memoryBytes;
        st.lights.clustered                 = params.lighting.clusteredLights && frameClusters;
    }
    // S36a2: the cascades
    st.shadows            = frameCascades;
    st.frozen             = freezeCascades;
    st.cascadeCount       = frameCascades ? cascades->count() : 0;
    st.shadowLightDir     = vec3(block.shadowLight(0), block.shadowLight(1), block.shadowLight(2));
    st.shadowMemoryBytes  = cascades->memoryBytes();
    st.linearShadowFilter = lightingSet->linearShadowFilter();
    for (int c = 0; c < kMaxCascades; ++c)
    {
        const CascadedShadows::Cascade& k = cascades->cascades()[size_t(c)];
        CascadeStatistics& o              = st.cascades[size_t(c)];
        o                                 = CascadeStatistics();
        if (!frameCascades || c >= cascades->count() || !k.valid) continue;
        o.valid      = true;
        o.splitNear  = float(k.splitNear);
        o.splitFar   = float(k.splitFar);
        o.center     = k.axisX * k.cx + k.axisY * k.cy + k.axisZ * (0.5 * (k.zNear + k.zFar));
        o.halfExtent = float(k.halfExtent);
        o.depthRange = float(k.zNear - k.zFar);
        o.texelSize  = float(k.texelSize);
        o.interval   = k.interval;
        o.lastUpdate       = k.lastUpdate;
        o.updatedThisFrame = k.updated;
        o.instances  = cascadeInstances[size_t(c)];
        o.triangles  = cascadeTriangles[size_t(c)];
        if (k.updated && size_t(c) < terrain->shadowStats().size())
            cascadeTerrain[size_t(c)] = terrain->shadowStats()[size_t(c)];
        o.terrainNodes     = uint32_t(cascadeTerrain[size_t(c)].nodes);
        o.terrainTriangles    = uint32_t(cascadeTerrain[size_t(c)].triangles);
        o.terrainDeduplicated = uint32_t(cascadeTerrain[size_t(c)].deduplicated);
    }
}

void RenderScene::latchDebug()
{
    frameDebugRaw = false;
    // Cancelled requests are dropped; a sequence whose requests are all cancelled ends (round 2, finding 13)
    probeRequests.erase(std::remove_if(probeRequests.begin(), probeRequests.end(),
                                       [](const ProbeRequest& r) { return r.isCancelled(); }),
                        probeRequests.end());
    if (probeRun && std::all_of(probeRun->requests.begin(), probeRun->requests.end(),
                                [](const ProbeRequest& r) { return r.isCancelled(); }))
    {
        probeRun.reset();
        applyHeldProbeState();
    }
    // A resized surface ends the sequence (finishProbe answers without values)
    if (probeRun && probeRun->resized) finishProbe();
    // A probe starts with the next frame: the requests made until then share its frames
    if (!probeRun && !probeRequests.empty())
    {
        probeRun           = std::make_unique<ProbeRun>();
        probeRun->requests = std::move(probeRequests);
        probeRequests.clear();
    }
    if (probeRun)
    {
        frameDebug.view     = kProbeViews[size_t(probeRun->step)];
        frameDebug.overlays = 0;
        frameDebugRaw       = probeRun->step >= 4;
        return;
    }
    // The override applies to the frames that are read back (as the ImGui override)
    if (debugOverride && renderer->screenshotRequested())
        frameDebug = *debugOverride;
    else
        frameDebug = userDebug;
}

void RenderScene::requestProbe(ivec2 pixel, std::function<void(const LightingProbe&)> callback,
                               std::shared_ptr<const bool> cancelled)
{
    probeRequests.push_back(ProbeRequest{pixel, std::nullopt, std::move(callback), std::move(cancelled)});
}

void RenderScene::requestProbe(const Vec3& world, std::function<void(const LightingProbe&)> callback,
                               std::shared_ptr<const bool> cancelled)
{
    probeRequests.push_back(ProbeRequest{ivec2(-1, -1), world, std::move(callback), std::move(cancelled)});
}

void RenderScene::finishProbe()
{
    std::unique_ptr<ProbeRun> run = std::move(probeRun);
    probeRun.reset();
    const mat4 invViewProj = inverse(run->viewProj);
    std::vector<LightingProbe> answers(run->requests.size());
    for (size_t k = 0; k < run->requests.size(); ++k)
    {
        LightingProbe& p = answers[k];
        p.pixel          = run->requests[k].pixel;
        p.frames         = kProbeFrames;
        p.physical       = run->physical;
        // A resized surface: the frames sampled other points, no values (not consistent, not in the image)
        if (run->resized)
        {
            p.consistent = false;
            continue;
        }
        auto value = [&](int frame)
        {
            const auto& v = run->values[size_t(frame)];
            return v.size() > k ? v[k] : LightingDebug::PixelValue();
        };
        const LightingDebug::PixelValue first = value(0);
        p.inImage                             = first.valid;
        if (p.inImage)
        {
            p.depth = first.depth;
            p.hit   = first.depth > 0;
            for (int f = 1; f < kProbeFrames; ++f)
            {
                const float d = value(f).depth;
                if (std::abs(d - p.depth) > 0.01f * std::max(std::abs(p.depth), 1e-6f)) p.consistent = false;
            }
            if (p.hit)
            {
                // The position from the depth (reversed-Z) and frame 1's camera-relative matrices
                const float x  = (float(p.pixel(0)) + 0.5f) / float(run->size(0)) * 2 - 1;
                const float y  = (float(p.pixel(1)) + 0.5f) / float(run->size(1)) * 2 - 1;
                const vec4 h   = invViewProj * vec4(x, y, p.depth, 1);
                const Vec3 rel = Vec3(double(h(0)) / h(3), double(h(1)) / h(3), double(h(2)) / h(3));
                p.position     = run->camera + rel;
                p.distance     = rel.norm();
            }
            auto rgb       = [](const vec4& v) { return vec3(v(0), v(1), v(2)); };
            // Every linear frame is written with the pre-exposure (also ACES without an environment, which is not
            // physical); the clay output is the display value
            const float pe = run->postLinear ? run->preExposure : 1.0f;
            p.color        = rgb(first.color) / pe;
            p.luminance    = 0.2126f * p.color(0) + 0.7152f * p.color(1) + 0.0722f * p.color(2);
            p.ev100        = float(std::log2(std::max(double(p.luminance), 1e-6) * 8.0));
            p.albedo       = rgb(value(1).color);
            const vec4 n   = value(2).color;
            const vec3 nn(n(0) * 2 - 1, n(1) * 2 - 1, n(2) * 2 - 1);
            p.normal     = nn.norm() > 1e-6f ? vec3(nn.normalized()) : nn;
            const vec4 m = value(3).color;
            p.roughness  = m(0);
            p.metalness  = m(1);
            p.ao         = m(2);
            // The raw frames are scaled by frame 1's pre-exposure (frameLighting)
            const float raw = std::max(run->preExposure, 1e-30f);
            p.direct        = rgb(value(4).color) / raw;
            p.ambient       = rgb(value(5).color) / raw;
            p.specular      = rgb(value(6).color) / raw;
            // S36a1c: the cloud shadow; S36a2: the cascades (factor, cascade + 1)
            p.cloudShadow   = value(7).color(0);
            p.cascadeShadow = value(8).color(0);
            p.cascade       = int(std::lround(value(8).color(1))) - 1;
            // The horizon map (factor, R and G above the receiver)
            p.horizonShadow = value(9).color(0);
            p.horizonR      = value(9).color(1);
            p.horizonG      = value(9).color(2);
            p.shadowFactor  = p.cloudShadow * p.cascadeShadow * p.horizonShadow;
            // S36b: the atlas shadow of the local lights (the darkest factor, the lights counted)
            p.atlasShadow       = value(10).color(0);
            p.atlasShadowLights = int(std::lround(value(10).color(1)));
            // S36b: the cluster and its lights (the CPU reference with frame 1's records), each light's radiance
            if (p.hit && run->clusters && run->physical)
            {
                const Vec3 rel = p.position - run->camera;
                p.cluster =
                    run->grid.clusterAt(double(p.pixel(0)) + 0.5, double(p.pixel(1)) + 0.5, rel.dot(run->forward));
                if (p.cluster >= 0)
                {
                    vec3 lo, hi;
                    LightClusters::clusterBounds(run->grid, run->proj, p.cluster, lo, hi);
                    const vec3 v = (-rel).normalized().cast<float>();
                    const vec3 x = rel.cast<float>();
                    for (size_t i = 0; i < run->lights.size(); ++i)
                    {
                        const int cl = p.cluster, gx = run->grid.x, gy = run->grid.y;
                        if (!LightClusters::inRange(run->lights[i].clusterRange, cl % gx, (cl / gx) % gy,
                                                    cl / (gx * gy)) ||
                            !LightClusters::lightTouches(run->lights[i], run->view, lo, hi))
                            continue;
                        p.clusterTotal++;
                        if (p.lights.size() >= size_t(run->grid.maxLightsPerCluster)) continue;
                        LightingProbe::LightContribution c;
                        c.handle                       = i < run->handles.size() ? run->handles[i] : 0;
                        const LocalLights::GpuLight& L = run->lights[i];
                        const vec3 lp(L.posRange(0), L.posRange(1), L.posRange(2));
                        c.distance = double((lp - x).norm());
                        c.radiance =
                            LocalLights::reflectedRadiance(L, x, p.normal, v, p.albedo, p.roughness, p.metalness);
                        p.lights.push_back(c);
                    }
                }
            }
        }
    }
    // The held state first: the callbacks see it, and what they set is not overwritten by it (CodeRabbit #82)
    applyHeldProbeState();
    for (size_t k = 0; k < run->requests.size(); ++k)
        if (!run->requests[k].isCancelled()) run->requests[k].callback(answers[k]);
}

void RenderScene::applyHeldProbeState()
{
    // The environment and the camera given during the sequence
    if (heldEnvironment)
    {
        env    = *heldEnvironment;
        envSet = true;
        heldEnvironment.reset();
    }
    if (heldToneMap)
    {
        params.toneMap = *heldToneMap;
        heldToneMap.reset();
    }
    if (heldCamera)
    {
        const HeldCamera c = *heldCamera;
        heldCamera.reset();
        setCamera(c.space, c.position, c.rotation, c.fovy);
    }
}

uint64_t RenderScene::storageBufferLimit() const
{
    return culling->storageLimit();
}

uint64_t RenderScene::scenePipelineCreations() const
{
    return culling->pipelineCreations() + terrain->pipelineCreations() + grass->pipelineCreations();
}

void RenderScene::addMesh(MeshId id, const std::string& name, const MeshLods& lods, MaterialId material)
{
    meshPool->addMesh(id, name, lods, material);
}

MeshId RenderScene::findMesh(const std::string& name) const
{
    return meshPool->findMesh(name);
}

MaterialId RenderScene::findMaterial(const std::string& name) const
{
    return meshPool->findMaterial(name);
}

void RenderScene::setPalette(const std::vector<PaletteEntry>& palette)
{
    meshPool->setPalette(palette);
}

// S36b: the world bounds of a set as a dirty bound of the shadow atlas (its static tiles in range are re-rendered)
void RenderScene::markSetDirty(InstanceSetHandle set)
{
    auto it = instanceBuffer->sets().find(set.id);
    if (it == instanceBuffer->sets().end() || !it->second.hasBounds) return;
    const Vec3 o = regionOrigin(it->second.region);
    atlasShadows->markDirty(o + it->second.boundsMin, o + it->second.boundsMax);
}

void RenderScene::markRegionDirty(const RegionId& region)
{
    const Vec3 o      = regionOrigin(region);
    const double size = std::ldexp(chunkSize, region.level);
    atlasShadows->markDirty(Vec3(o(0), -1e30, o(2)), Vec3(o(0) + size, 1e30, o(2) + size));
}

InstanceSetHandle RenderScene::addInstanceSet(const RegionId& region, const std::vector<StaticInstance>& instances,
                                              uint32_t capacity)
{
    const InstanceSetHandle h = instanceBuffer->addSet(region, instances, capacity, *meshPool);
    markSetDirty(h);
    return h;
}

void RenderScene::updateInstances(InstanceSetHandle set, uint32_t first, const std::vector<StaticInstance>& instances)
{
    // The bounds before (moved or removed casters) and after (new ones)
    markSetDirty(set);
    instanceBuffer->update(set, first, instances, *meshPool);
    markSetDirty(set);
}

void RenderScene::removeInstance(InstanceSetHandle set, uint32_t index)
{
    markSetDirty(set);
    instanceBuffer->removeInstance(set, index);
}

void RenderScene::removeInstanceSet(InstanceSetHandle set)
{
    markSetDirty(set);
    instanceBuffer->removeSet(set);
}

InstanceHandle RenderScene::addDynamic(MeshId mesh)
{
    return instanceBuffer->addDynamic(mesh);
}

void RenderScene::markDynamicDirty(MeshId mesh, const ChunkId& chunk, const DynamicPose& pose)
{
    // S36b: a resting dynamic instance is in the atlas's static layer: its bounds are dirty when it leaves or takes a
    // resting pose
    const MeshPool::Mesh* m = meshPool->mesh(mesh);
    if (!m) return;
    const double scale  = pose.scale;
    const Vec3 centre   = chunkOrigin(chunk) + pose.position + pose.rotation * (m->sphereCenter.cast<double>() * scale);
    const double radius = double(m->sphereRadius + m->boundsPadding) * scale;
    atlasShadows->markDirty(centre - Vec3(radius, radius, radius), centre + Vec3(radius, radius, radius));
}

void RenderScene::setDynamic(InstanceHandle h, const ChunkId& chunk, const DynamicPose& prev, const DynamicPose& curr,
                             uint32_t boneOffset)
{
    const auto& list = instanceBuffer->dynamics();
    if (h.id >= 1 && h.id <= list.size())
    {
        const InstanceBuffer::Dynamic old = list[h.id - 1];
        const bool wasResting             = old.alive && old.set && !InstanceBuffer::moving(old);
        instanceBuffer->setDynamic(h, chunk, prev, curr, boneOffset);
        const InstanceBuffer::Dynamic& now = list[h.id - 1];
        const bool resting                 = !InstanceBuffer::moving(now);
        const bool samePlace = old.set && old.chunk == now.chunk && old.curr.position == now.curr.position &&
                               old.curr.rotation.coeffs() == now.curr.rotation.coeffs() &&
                               old.curr.scale == now.curr.scale;
        if (wasResting && !(resting && samePlace)) markDynamicDirty(old.mesh, old.chunk, old.curr);
        if (resting && !(wasResting && samePlace)) markDynamicDirty(now.mesh, now.chunk, now.curr);
        if (now.snappedMove)
            snappedDynamics[h.id] = frameCount;
        else
            snappedDynamics.erase(h.id);
        return;
    }
    instanceBuffer->setDynamic(h, chunk, prev, curr, boneOffset);
}

void RenderScene::settleSnapped()
{
    // A dynamic snapped before the previous frame and not since: at rest (its rest pose is a static caster)
    for (auto it = snappedDynamics.begin(); it != snappedDynamics.end();)
    {
        if (it->second >= frameCount)
        {
            ++it;
            continue;
        }
        const InstanceHandle h{it->first};
        if (instanceBuffer->settle(h))
        {
            const InstanceBuffer::Dynamic& d = instanceBuffer->dynamics()[h.id - 1];
            if (!InstanceBuffer::moving(d)) markDynamicDirty(d.mesh, d.chunk, d.curr);
        }
        it = snappedDynamics.erase(it);
    }
}

void RenderScene::removeDynamic(InstanceHandle h)
{
    snappedDynamics.erase(h.id);
    const auto& list = instanceBuffer->dynamics();
    if (h.id >= 1 && h.id <= list.size() && list[h.id - 1].alive && !InstanceBuffer::moving(list[h.id - 1]))
        markDynamicDirty(list[h.id - 1].mesh, list[h.id - 1].chunk, list[h.id - 1].curr);
    instanceBuffer->removeDynamic(h);
}

void RenderScene::setCamera(SpaceId space, const Vec3& position, const Quat& rotation, double fovy)
{
    // A probe holds the camera of its frame 1 (frame 1's transfer callback still sets it: the render path's camera)
    if (probeRun && probeRun->cameraTaken && cameraSet)
    {
        heldCamera = HeldCamera{space, position, rotation, fovy};
        return;
    }
    cameraSet      = true;
    cameraSpace    = space;
    cameraPosition = position;
    cameraRotation = rotation.normalized();
    cameraFovy     = fovy;
    // The matrices at once (a game's draws in the same frame read them)
    const ivec2 s = size();
    if (s(0) > 0 && s(1) > 0) updateMatrices(s(0), s(1));
}

LightHandle RenderScene::addLight(const ChunkId& chunk, const LightData& light)
{
    return lightTable->add(chunk, light);
}

void RenderScene::setLight(LightHandle h, const ChunkId& chunk, const LightData& light)
{
    lightTable->set(h, chunk, light);
}

void RenderScene::removeLight(LightHandle h)
{
    lightTable->remove(h);
}

LightListHandle RenderScene::addLights(const ChunkId& chunk, const std::vector<LightData>& lights)
{
    return lightTable->addList(chunk, lights);
}

void RenderScene::removeLights(LightListHandle list)
{
    lightTable->removeList(list);
}

void RenderScene::setInterpolation(float a)
{
    alpha = a;
}

bool RenderScene::multiDrawIndirect() const
{
    return culling->multiDrawIndirect();
}

void RenderScene::beginSceneFrame(uint32_t frameIndex)
{
    // The slot's previous frame has completed: its statistics are read before the slot is reused
    harvestStatistics();
    SlotStatistics& slot = slotStatistics[frameIndex];
    slot.pending         = true;
    slot.frame           = frameCount + 1;
    slot.timelineValue   = base().currentFrameSignalValue();
    slot.drawCommands    = 0;
    slot.terrainNodes    = 0;
    slot.cascadeMask     = 0;
}

void RenderScene::updateMatrices(int width, int height)
{
    const double asp = double(width) / double(height);
    const Mat3 R     = cameraRotation.matrix();
    // The view matrix: only the rotation (camera-relative positions)
    mat4 view = mat4::Identity();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) view(i, j) = float(R(j, i));
    frameView       = view;
    frameProjection = reversedZInfinitePerspective(float(cameraFovy), float(asp), params.nearPlane);
}

void RenderScene::frameCamera(int width, int height, SceneCamera& c)
{
    c             = SceneCamera();
    c.alpha       = alpha;
    c.preExposure = framePreExposure;
    c.linear      = sceneWritesLinear();
    if (!cameraSet || width <= 0 || height <= 0) return;
    c.valid    = true;
    c.space    = cameraSpace;
    c.position = cameraPosition;
    updateMatrices(width, height);
    c.viewProj       = frameProjection * frameView;
    const double asp = double(width) / double(height);
    const Mat3 R     = cameraRotation.matrix();

    // The frustum planes in camera-relative world coordinates (inward normals; no far plane)
    const double tanY               = std::tan(cameraFovy * 0.5);
    const double tanX               = tanY * asp;
    const std::array<Vec3, 4> sides = {Vec3(1, 0, -tanX), Vec3(-1, 0, -tanX), Vec3(0, 1, -tanY), Vec3(0, -1, -tanY)};
    for (int k = 0; k < 4; ++k)
    {
        const Vec3 n = R * sides[k].normalized();
        c.planes[k]  = Vec4(n(0), n(1), n(2), 0);
    }
    const Vec3 forward = R * Vec3(0, 0, -1);
    c.planes[4]        = Vec4(forward(0), forward(1), forward(2), -double(params.nearPlane));
}

void RenderScene::harvestStatistics()
{
    const uint64_t completed = base().completedFrameValue();
    // In frame order: both slots may have completed since the last call
    std::array<uint32_t, 2> order = {0, 1};
    if (slotStatistics[1].frame < slotStatistics[0].frame) std::swap(order[0], order[1]);
    for (uint32_t i : order)
    {
        SlotStatistics& slot = slotStatistics[i];
        if (!slot.pending || slot.timelineValue > completed) continue;
        slot.pending = false;
        if (slot.frame <= stats.frame) continue;
        RenderStatistics r;
        r.frame = slot.frame;
        culling->readStatistics(i, r);
        r.drawCommands = slot.drawCommands;
        r.terrainNodes = slot.terrainNodes;
        // S36a2: the casters of the cascades rendered in that frame
        for (int c = 0; c < kMaxCascades; ++c)
            if (slot.cascadeMask & (1u << c))
            {
                cascadeInstances[size_t(c)] = r.cascadeInstances[size_t(c)];
                cascadeTriangles[size_t(c)] = r.cascadeTriangles[size_t(c)];
            }
        stats = r;
        completedStatistics.push_back(r);
        if (completedStatistics.size() > 16) completedStatistics.pop_front();
    }
}

bool RenderScene::takeStatistics(RenderStatistics& out)
{
    harvestStatistics();
    if (completedStatistics.empty()) return false;
    out = std::move(completedStatistics.front());
    completedStatistics.pop_front();
    return true;
}

void RenderScene::trackMeshStatistics(const std::vector<MeshId>& meshes)
{
    culling->setTrackedMeshes(meshes);
}

const RenderStatistics& RenderScene::statistics()
{
    harvestStatistics();
    return stats;
}

}  // namespace Vulkan
}  // namespace Saiga
