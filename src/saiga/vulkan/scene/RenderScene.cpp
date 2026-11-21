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
#include "saiga/vulkan/scene/GrassPass.h"
#include "saiga/vulkan/scene/InstanceBuffer.h"
#include "saiga/vulkan/scene/MeshPool.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/SceneCulling.h"
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
        "transfer",        "cull",        "depth",     "depth.terrain", "scene", "scene.terrain",
        "scene.instances", "scene.grass", "scene.sky", "resolve",       "post",  "imgui"};
    return names;
}
std::vector<std::string> sectionNameStrings()
{
    std::vector<std::string> r;
    for (const char* n : sectionNames()) r.push_back(n);
    return r;
}
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
        postSet = postPipeline.createDescriptorSet();

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
    StaticDescriptorSet resolveSet, postSet;
    // Shared by the frames in flight. hdrMsaa only with MSAA (stored only for the custom resolve); hdrResolved is the
    // scene target with 1 sample. depthResolved: the SAMPLE_ZERO depth resolve, only with MSAA.
    Texture2D depthImage, depthResolved, hdrMsaa, hdrResolved;
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
    const bool msaa      = samples != vk::SampleCountFlagBits::e1;
    const bool linear    = s.sceneWritesLinear();
    const bool custom    = msaa && linear;  // the tone-map weighted resolve (Identity: the hardware average)
    const float scale    = s.frameExposure / s.framePreExposure;
    const bool drawImgui = imguiFrame(
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
            s.terrain->transfer(cmd, frame.frameIndex, *s.atlas, camera, proto, s.sceneTime, surfaceWidth,
                                SurfaceHeight);
        s.grass->update(*s.atlas, *s.terrain, camera, proto);
        s.slotStatistics[frame.frameIndex].terrainNodes = uint32_t(s.terrain->stats().nodes);
        SceneLighting lighting;
        s.frameLighting(camera, proto, enclosed, int(samples), lighting);
        s.culling->recordTransfer(cmd, frame.frameIndex, *s.meshPool, *s.instanceBuffer, camera, lighting);
        // ... and make the writes visible to this frame's culling and draws.
        vk::MemoryBarrier2 raw(Stage::eAllTransfer, Access::eTransferWrite,
                               Stage::eVertexInput | Stage::eVertexShader | Stage::eFragmentShader |
                                   Stage::eComputeShader | Stage::eDrawIndirect,
                               Access::eUniformRead | Access::eShaderRead | Access::eVertexAttributeRead |
                                   Access::eIndexRead | Access::eIndirectCommandRead);
        cmd.pipelineBarrier2(vk::DependencyInfo({}, raw, nullptr, nullptr));
    }
    timings.leaveSection("transfer", cmd);

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
        const auto prevColor  = Stage::eColorAttachmentOutput | Stage::eFragmentShader;
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
        // The environment's sky (S20): from setPrototypeEnvironment on, with or without terrain; not in an enclosed
        // camera space (S21: the background is the clear color)
        if (s.foliage && !enclosed) s.terrain->renderSky(cmd, frame.frameIndex, proto);
        subEnd("scene.sky", cmd);
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

    {
        // The resolved image -> post's sampled read; the output image after the acquire (the submit waits for
        // imageAvailable at COLOR_ATTACHMENT_OUTPUT; headless: the frame that last used it has completed)
        vk::Image output                                = colorImage(frame.imageIndex);
        std::array<vk::ImageMemoryBarrier2, 2> barriers = {
            vk::ImageMemoryBarrier2(Stage::eColorAttachmentOutput, Access::eColorAttachmentWrite,
                                    Stage::eFragmentShader, Access::eShaderSampledRead,
                                    vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, hdrResolved.image(),
                                    colorRange()),
            vk::ImageMemoryBarrier2(Stage::eColorAttachmentOutput, {}, Stage::eColorAttachmentOutput,
                                    Access::eColorAttachmentWrite | Access::eColorAttachmentRead,
                                    vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, output, colorRange())};
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, barriers));
    }

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
            PostPush push{scale, linear ? 1 : 0, isSrgbFormat(getColorFormat()) ? 1 : 0};
            postPipeline.bindDescriptorSet(cmd, postSet);
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
    std::unique_ptr<StagingBuffer> screenshot = beginScreenshots(cmd, output, callbacks);
    recordPresentTransition(cmd, output, screenshot != nullptr);
    submitCommands(cmd, frame);
    finishScreenshots(frame, screenshot.get(), callbacks);
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
    culling = std::make_unique<SceneCulling>();
    culling->init(base(), formats(), params);
    protoSettings = std::make_unique<PrototypeSettings>();
    // The static instance buffer is bound whole: it grows up to the storage buffer limit
    instanceBuffer = std::make_unique<InstanceBuffer>();
    instanceBuffer->init(base(), params.staticInstanceCapacity, culling->storageLimit());
    atlas = std::make_unique<TerrainAtlas>();
    atlas->init(base(), params, culling->storageLimit());
    terrain = std::make_unique<TerrainPass>();
    terrain->init(base(), formats());
    grass = std::make_unique<GrassPass>();
    grass->init(base(), formats());
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
    instanceBuffer.reset();
    meshPool.reset();
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
    const bool userImgui  = renderer->should_render_imgui;
    const bool overridden = imguiOverride.has_value() && renderer->screenshotRequested();
    if (overridden) renderer->should_render_imgui = *imguiOverride;
    renderer->render(info);
    if (overridden) renderer->should_render_imgui = userImgui;
    transferFn = nullptr;
    renderFn   = nullptr;
    guiFn      = nullptr;
    if (renderer->submittedFrames() == before) return false;
    // The scene of the next frame is written with this frame's exposure
    lastExposure = frameExposure;
    frameCount++;
    submittedFrames.push_back({timeline, frameCount});
    if (submittedFrames.size() > 32) submittedFrames.pop_front();
    return true;
}

void RenderScene::latchFrame()
{
    // Once per recorded frame, before any callback (setters apply from the next frame)
    frameToneMap  = params.toneMap;
    frameExposure = frameToneMap == ToneMap::Identity ? 1.0f : params.exposure;
    // The previous frame's exposure; this frame's after a change by more than kMaxPreExposureChange (fp16: the scaled
    // values would clip at 65504 or lose precision, a one-frame flash). Clay mode ignores exposure: 1.
    float pre = lastExposure;
    if (frameExposure > pre * kMaxPreExposureChange || frameExposure * kMaxPreExposureChange < pre) pre = frameExposure;
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
    params.toneMap = toneMap;
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
    SAIGA_ASSERT(data.pipelineClass != uint8_t(PipelineClass::PrototypeFoliage) || foliage,
                 "addMaterial: PipelineClass::PrototypeFoliage needs the foliage texture of setPrototypeEnvironment");
    meshPool->addMaterial(id, name, data);
}

void RenderScene::setMaterial(MaterialId id, const MaterialData& data)
{
    SAIGA_ASSERT(data.pipelineClass != uint8_t(PipelineClass::PrototypeFoliage) || foliage,
                 "setMaterial: PipelineClass::PrototypeFoliage needs the foliage texture of setPrototypeEnvironment");
    meshPool->setMaterial(id, data);
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

void RenderScene::setPrototypeEnvironment(const PrototypeSettings& settings, uint32_t foliageSeed)
{
    SAIGA_ASSERT(!transferFn, "setPrototypeEnvironment: not inside renderFrame");
    // The frames in flight may use the previous texture
    renderer->waitIdle();
    culling->setFoliageTexture(nullptr);
    terrain->setFoliageTexture(nullptr);
    foliage = std::make_unique<Texture2DArray>();
    {
        auto layers = FoliageGenerator::foliageTextures(512, foliageSeed);
        std::vector<const Image*> images;
        for (auto& l : layers) images.push_back(&l);
        foliage->fromImages(base(), images, vk::ImageUsageFlagBits::eSampled, false, true);
    }
    culling->setFoliageTexture(foliage.get());
    terrain->setFoliageTexture(foliage.get());
    *protoSettings = settings;
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
    if (spaceRenderSettings.size() <= space) spaceRenderSettings.resize(size_t(space) + 1);
    spaceRenderSettings[space] = settings;
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
    // An enclosed camera space (S21): the fixed enclosed lighting, independent of the game's environment
    if (enclosed) prototypeLighting(kEnclosedLighting, sceneTime, 0.0f, out);
    // The new path's environment (S20): the fog's base is the terrain set's sea level
    else if (foliage)
        prototypeLighting(proto, sceneTime, atlas->seaLevel() ? float(double(*atlas->seaLevel()) * kHeightUnit) : 0.0f,
                          out);
    const vec3 p = camera.position.cast<float>();
    out.camera   = vec4(p(0), p(1), p(2), float(samples));
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
