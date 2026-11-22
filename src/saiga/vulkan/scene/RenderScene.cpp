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
#include "saiga/vulkan/scene/CloudLayer.h"
#include "saiga/vulkan/scene/DebugLines.h"
#include "saiga/vulkan/scene/EnvironmentMap.h"
#include "saiga/vulkan/scene/GrassPass.h"
#include "saiga/vulkan/scene/InstanceBuffer.h"
#include "saiga/vulkan/scene/LightingDebug.h"
#include "saiga/vulkan/scene/MeshPool.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/SceneCulling.h"
#include "saiga/vulkan/scene/SceneLightingSet.h"
#include "saiga/vulkan/scene/SkyAmbient.h"
#include "saiga/vulkan/texture/Sampler.h"
#include "saiga/vulkan/texture/Texture.h"
#include "saiga/vulkan/window/Window.h"
#include "saiga/vulkan/world/FoliageGenerator.h"

#include <algorithm>
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
        "transfer",      "environment",      "cull",        "depth",     "depth.terrain", "scene",
        "scene.terrain", "scene.instances",  "scene.grass", "scene.sky", "scene.clouds",  "scene.debug",
        "resolve",       "lighting.preview", "post",        "imgui"};
    return names;
}
std::vector<std::string> sectionNameStrings()
{
    std::vector<std::string> r;
    for (const char* n : sectionNames()) r.push_back(n);
    return r;
}
}  // namespace

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
};

namespace
{
// The views of the probe's frames (frame 1: normal); frames 5-7 write raw radiance, frame 8 the cloud shadow (raw data)
const std::array<LightingDebugView, RenderScene::kProbeFrames> kProbeViews = {
    LightingDebugView::None,         LightingDebugView::Albedo,     LightingDebugView::Normals,
    LightingDebugView::Material,     LightingDebugView::DirectOnly, LightingDebugView::AmbientOnly,
    LightingDebugView::SpecularOnly, LightingDebugView::CloudShadow};
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
        s.culling->recordTransfer(cmd, frame.frameIndex, *s.meshPool, *s.instanceBuffer, camera);
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
    const bool environmentPass = s.framePhysical && s.envSet && !enclosed && !s.env.enclosed && camera.valid;
    if (environmentPass)
    {
        // A full update also on the first frame after frames without this pass (clay, an enclosed space, a debug
        // frame): the bound environment map is of an older environment (review: it was swapped in 7 frames later)
        LightingParameters lp = s.params.lighting;
        lp.forceFullUpdates   = lp.forceFullUpdates || s.fullUpdateRequested || s.environmentSkipped;
        s.fullUpdateRequested = false;
        s.environmentSkipped  = false;
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
        s.envMap->record(cmd, s.env, lighting.atmosphere(0), lp, in, s.frameCount + 1);
    }
    else
        s.environmentSkipped = true;
    timings.leaveSection("environment", cmd);
    // The lighting set of this frame (after the environment: the environment map swapped in this frame is bound)
    s.writeLightingSet(cmd, frame.frameIndex, lighting);

    // The GPU culling of the scene's instances (written every frame, empty without instances or a camera)
    timings.enterSection("cull", cmd);
    s.culling->recordCull(cmd, frame.frameIndex, *s.meshPool, camera);
    timings.leaveSection("cull", cmd);

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
        s.terrain->render(cmd, frame.frameIndex, prepass ? TerrainPass::Mode::Equal : TerrainPass::Mode::Scene, proto);
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
        if (s.foliage && !enclosed) s.grass->render(cmd, frame.frameIndex, *s.terrain);
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
        src.set(LightingImage::BrdfLut, &s.brdf->lut(), 1);
        // S36a1c: the atmosphere's LUTs and the environment map's debug images (recorded on demand)
        src.set(LightingImage::TransmittanceLut, &s.atmosphere->transmittance(), 0);
        src.set(LightingImage::SkyViewLut, &s.atmosphere->skyView(), 0, 1.0f / Atmosphere::kSkyLutScale);
        src.set(LightingImage::MultiScatteringLut, &s.atmosphere->multiScattering(), 0);
        if (previews && s.debugImages->requested(LightingImage::Environment))
            src.set(
                LightingImage::Environment,
                s.envMap->recordDebug(cmd, frame.frameIndex, 0, s.debugImages->sourceMip(LightingImage::Environment)),
                0);
        if (previews && s.debugImages->requested(LightingImage::Irradiance))
            src.set(LightingImage::Irradiance, s.envMap->recordDebug(cmd, frame.frameIndex, 1, 0), 0);
        // The cloud layer's previews (the coverage over the tile, the shadow around the camera)
        if (s.debugImages->requested(LightingImage::CloudCoverage) ||
            s.debugImages->requested(LightingImage::CloudShadow))
        {
            CloudLayer::DebugInput in;
            in.clouds = lighting.clouds;
            in.origin = vec2(lighting.cloudMotion(0), lighting.cloudMotion(1));
            in.mean   = lighting.cloudMotion(2);
            in.height = lighting.atmosphere(0);
            in.light  = lighting.shadowLight(3) > 0
                            ? vec3(lighting.shadowLight(0), lighting.shadowLight(1), lighting.shadowLight(2))
                            : vec3(0, 1, 0);
            if (s.debugImages->requested(LightingImage::CloudCoverage))
                src.set(LightingImage::CloudCoverage, s.clouds->recordDebug(cmd, frame.frameIndex, 1, in), 1);
            if (s.debugImages->requested(LightingImage::CloudShadow))
                src.set(LightingImage::CloudShadow, s.clouds->recordDebug(cmd, frame.frameIndex, 2, in), 1);
        }
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
    brdf = std::make_unique<BrdfLut>();
    brdf->init(base());
    atmosphere = std::make_unique<Atmosphere>();
    atmosphere->init(base());
    envMap = std::make_unique<EnvironmentMap>();
    envMap->init(base(), params.lighting.environmentResolution);
    clouds = std::make_unique<CloudLayer>();
    clouds->init(base(), formats(), CloudSettings().seed);
    ambient->setEnvironmentMap(envMap.get());
    debugImages = std::make_unique<LightingDebug>();
    debugImages->init(base());
    lines = std::make_unique<DebugLines>();
    lines->init(base(), formats());
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

void RenderScene::setEnvironment(const EnvironmentState& environment)
{
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
    // No resource depends on the a1a fields yet (the environment map and clouds come with S36a1c)
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
    atlas->setPyramid(space, pyramid);
}

void RenderScene::setTerrainTile(const RegionId& region, const TerrainTileData& tile)
{
    atlas->setTile(region, tile);
}

void RenderScene::removeTerrainTile(const RegionId& region)
{
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
    atlas->clear();
}

void RenderScene::patchTerrainTile(const RegionId& region, int x0, int z0, int w, int h, const int32_t* heights)
{
    atlas->patchTile(region, x0, z0, w, h, heights);
}

void RenderScene::patchTerrainPyramid(SpaceId space, int x0, int z0, int w, int h, const int32_t* heights)
{
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
                                SceneLighting& out) const
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
    out.studioShading              = vec4(studio.ambient, studio.hemisphere ? 1 : 0, studio.albedo, 0);
    const float cameraAboveBase    = float(camera.position.y() - fogBase);
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
        // The shadow light (D13; a2 adds the hysteresis): the body with the higher illuminance on the ground
        const float sunGround  = e.sunIlluminance * std::max(e.sunDir(1), 0.0f);
        const float moonGround = e.moonIlluminance * std::max(e.moonDir(1), 0.0f);
        if (sunGround > 0 && sunGround >= moonGround)
            out.shadowLight = vec4(e.sunDir(0), e.sunDir(1), e.sunDir(2), 1);
        else if (moonGround > 0)
            out.shadowLight = vec4(e.moonDir(0), e.moonDir(1), e.moonDir(2), 2);
        // The moon in daylight adds about 1e-6 of the sun's light: no transmittance lookup and GGX for it (the sky's
        // LUTs and the moon disc take the environment's values)
        if (moonGround < 1e-4f * sunGround) out.moon(3) = 0;
        out.minAmbient      = vec4(e.minAmbient(0), e.minAmbient(1), e.minAmbient(2), e.ambientScale);
        out.enclosedAmbient = vec4(e.enclosedAmbient(0), e.enclosedAmbient(1), e.enclosedAmbient(2), 0);
        // The cloud layer: the cloudiness through the noise's histogram (the layer's mean coverage follows the
        // weather's cloudiness)
        out.clouds =
            vec4(e.clouds.altitude, e.clouds.radius, clouds->effectiveCloudiness(e.cloudiness), e.clouds.shadowFade);
        // The coverage texture's position of the camera (double: the camera's xz and the wrapped offset)
        auto wrap = [](double v)
        {
            const double t = std::fmod(v, CloudLayer::kTile);
            return float(t < 0 ? t + CloudLayer::kTile : t);
        };
        const bool cloudsOn = physical && params.lighting.clouds && !e.enclosed && !enclosed && e.cloudiness > 0;
        out.cloudMotion     = vec4(wrap(camera.position.x() + double(e.cloudOffset(0))),
                                   wrap(camera.position.z() + double(e.cloudOffset(1))),
                                   clouds->meanCoverage(out.clouds(2)), cloudsOn ? 1.0f : 0.0f);
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

void RenderScene::writeLightingSet(vk::CommandBuffer cmd, uint32_t frameIndex, const SceneLighting& frameBlock)
{
    ambient->update(cmd, frameIndex, env);
    SceneLightingBindings b;
    b.ambient = ambient->bindings(frameIndex);
    if (foliage) b.foliage = foliage->getDescriptorInfo();
    b.brdfLut           = brdf->descriptorInfo();
    b.transmittanceLut  = atmosphere->transmittanceInfo();
    b.skyViewLut        = atmosphere->skyViewInfo();
    b.cloudCoverage     = clouds->coverageInfo();
    SceneLighting block = frameBlock;
    block.atmosphere(1) = ambient->usesEnvironmentMap() ? 1.0f : 0.0f;
    block.atmosphere(3) = atmosphere->skyViewHorizonElevation();
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
    st.environmentMap    = envMap->statistics();
    st.lutComputations   = atmosphere->lutComputations();
    st.clouds            = block.cloudMotion(3) > 0.5f;
    st.cloudMeanCoverage = block.cloudMotion(2);
    st.luminance         = debugImages->statistics(LightingImage::Luminance);
    st.luminanceFrame    = debugImages->statisticsFrame(LightingImage::Luminance);
    // The block and SH buffers per frame in flight, the BRDF LUT (RG16F), the previews, the debug lines, the LUTs, the
    // environment map and the clouds (not the dummy textures)
    st.memoryBytes = VulkanRenderer::framesInFlight * (sizeof(SceneLighting) + SkyAmbient::kShBytes) +
                     uint64_t(BrdfLut::kSize) * BrdfLut::kSize * 4 + debugImages->memoryBytes() + lines->memoryBytes() +
                     atmosphere->memoryBytes() + envMap->memoryBytes() + clouds->memoryBytes();
}

void RenderScene::latchDebug()
{
    frameDebugRaw = false;
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

void RenderScene::requestProbe(ivec2 pixel, std::function<void(const LightingProbe&)> callback)
{
    probeRequests.push_back(ProbeRequest{pixel, std::nullopt, std::move(callback)});
}

void RenderScene::requestProbe(const Vec3& world, std::function<void(const LightingProbe&)> callback)
{
    probeRequests.push_back(ProbeRequest{ivec2(-1, -1), world, std::move(callback)});
}

void RenderScene::finishProbe()
{
    std::unique_ptr<ProbeRun> run = std::move(probeRun);
    probeRun.reset();
    const mat4 invViewProj = inverse(run->viewProj);
    for (size_t k = 0; k < run->requests.size(); ++k)
    {
        LightingProbe p;
        p.pixel    = run->requests[k].pixel;
        p.frames   = kProbeFrames;
        p.physical = run->physical;
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
            // S36a1c: the cloud shadow (the shadow factor of a1; a2 adds the cascades and the horizon)
            p.cloudShadow  = value(7).color(0);
            p.shadowFactor = p.cloudShadow;
        }
        run->requests[k].callback(p);
    }
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

InstanceSetHandle RenderScene::addInstanceSet(const RegionId& region, const std::vector<StaticInstance>& instances,
                                              uint32_t capacity)
{
    return instanceBuffer->addSet(region, instances, capacity, *meshPool);
}

void RenderScene::updateInstances(InstanceSetHandle set, uint32_t first, const std::vector<StaticInstance>& instances)
{
    instanceBuffer->update(set, first, instances, *meshPool);
}

void RenderScene::removeInstance(InstanceSetHandle set, uint32_t index)
{
    instanceBuffer->removeInstance(set, index);
}

void RenderScene::removeInstanceSet(InstanceSetHandle set)
{
    instanceBuffer->removeSet(set);
}

InstanceHandle RenderScene::addDynamic(MeshId mesh)
{
    return instanceBuffer->addDynamic(mesh);
}

void RenderScene::setDynamic(InstanceHandle h, const ChunkId& chunk, const DynamicPose& prev, const DynamicPose& curr,
                             uint32_t boneOffset)
{
    instanceBuffer->setDynamic(h, chunk, prev, curr, boneOffset);
}

void RenderScene::removeDynamic(InstanceHandle h)
{
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
        stats          = r;
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
