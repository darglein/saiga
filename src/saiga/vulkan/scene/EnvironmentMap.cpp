/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/EnvironmentMap.h"

#include "saiga/core/util/assert.h"
#include "saiga/vulkan/pipeline/DescriptorSetLayout.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

namespace
{
constexpr vk::Format kFormat = vk::Format::eR16G16B16A16Sfloat;

struct CapturePush
{
    int32_t size[4];
    float camera[4];      // x: the camera's radius (km)
    float uniformSky[4];  // w > 0: a uniform radiance
};
struct DebugPush
{
    int32_t size[4];  // the image's size, the mode, the mip
};
struct PrefilterPush
{
    int32_t size[4];  // this mip's size, the capture's size, the mip, the last mip
};
struct ShPush
{
    int32_t size[4];  // the SH source's face size, its mip
};

vk::Buffer handle(const Buffer& b)
{
    Memory::SafeAccessor acc(*b.memoryLocation());
    return b.memoryLocation()->data;
}

float angleDeg(const vec3& a, const vec3& b)
{
    const float na = a.norm(), nb = b.norm();
    if (na < 1e-6f || nb < 1e-6f) return na < 1e-6f && nb < 1e-6f ? 0.0f : 180.0f;
    const double c = std::clamp(double(a.dot(b)) / (double(na) * nb), -1.0, 1.0);
    return float(std::acos(c) * 57.29577951308232);
}

bool relativeChange(float a, float b, float tolerance)
{
    return std::abs(a - b) > tolerance * std::max(std::max(std::abs(a), std::abs(b)), 1e-6f);
}

void memoryBarrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
{
    vk::MemoryBarrier2 b(srcStage, srcAccess, dstStage, dstAccess);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, b, nullptr, nullptr));
}
}  // namespace

void EnvironmentMap::init(VulkanBase& b, int resolution)
{
    destroy();
    base = &b;
    SamplerParameters sp;
    sp.setFilter(vk::Filter::eLinear);
    sp.setAddressMode(vk::SamplerAddressMode::eClampToEdge);
    linear.create(b, sp);

    const auto c = vk::ShaderStageFlagBits::eCompute;
    {
        DescriptorSetLayout l;
        l.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageImage, 1, c));
        l.addBinding(1, vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, c));
        for (auto* p : {&capturePipeline, &prefilterPipeline})
        {
            p->init(b, 1);
            p->addDescriptorSetLayout(l);
            p->addPushConstantRange(
                {c, 0, uint32_t(p == &capturePipeline ? sizeof(CapturePush) : sizeof(PrefilterPush))});
        }
        capturePipeline.shaderPipeline.loadCompute(b.device, "vulkan/scene/lighting/env_capture.comp");
        capturePipeline.create();
        prefilterPipeline.shaderPipeline.loadCompute(b.device, "vulkan/scene/lighting/env_prefilter.comp");
        prefilterPipeline.create();
    }
    {
        DescriptorSetLayout l;
        l.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, 1, c));
        l.addBinding(1, vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, c));
        shPipeline.init(b, 1);
        shPipeline.addDescriptorSetLayout(l);
        shPipeline.addPushConstantRange({c, 0, sizeof(ShPush)});
        shPipeline.shaderPipeline.loadCompute(b.device, "vulkan/scene/lighting/env_sh.comp");
        shPipeline.create();
    }
    {
        DescriptorSetLayout l;
        l.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageImage, 1, c));
        l.addBinding(1, vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, c));
        l.addBinding(2, vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, c));
        debugPipeline.init(b, 1);
        debugPipeline.addDescriptorSetLayout(l);
        debugPipeline.addPushConstantRange({c, 0, sizeof(DebugPush)});
        debugPipeline.shaderPipeline.loadCompute(b.device, "vulkan/scene/lighting/env_debug.comp");
        debugPipeline.create();
        for (auto& slot : debugSets)
            for (auto& set : slot) set = debugPipeline.createRawDescriptorSet();
    }
    for (Buffer& sh : shBuffers)
    {
        sh.createBuffer(b, kShBytes, vk::BufferUsageFlagBits::eStorageBuffer,
                        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        std::memset(sh.getMappedPointer(), 0, kShBytes);
    }
    createImages(resolution);
}

void EnvironmentMap::createImages(int resolution)
{
    SAIGA_ASSERT(base);
    size             = std::clamp(resolution, int(kShSize), 512);
    const auto usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
    capture.create(*base, uint32_t(size), kFormat, usage, Texture::fullMipLevels(uint32_t(size), uint32_t(size), 1));
    for (TextureCube& cube : cubes) cube.create(*base, uint32_t(size), kFormat, usage, kMips);
    auto view = [&](Texture& t, uint32_t mip)
    {
        vk::ImageViewCreateInfo info({}, t.image(), vk::ImageViewType::e2DArray, kFormat, {},
                                     vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, mip, 1, 0, 6));
        return base->device.createImageView(info);
    };
    captureMip0 = view(capture, 0);
    for (int i = 0; i < 2; ++i)
        for (int m = 0; m < kMips; ++m) mipViews[size_t(i)][size_t(m)] = view(cubes[size_t(i)], uint32_t(m));

    // The descriptor sets of the images (the capture's sky-view input is written in record)
    captureSet = capturePipeline.createRawDescriptorSet();
    vk::DescriptorImageInfo src(linear, capture.view(), vk::ImageLayout::eShaderReadOnlyOptimal);
    std::vector<vk::WriteDescriptorSet> writes;
    std::vector<vk::DescriptorImageInfo> images;
    images.reserve(2 * kMips + 2);
    std::array<vk::DescriptorBufferInfo, 2> buffers;
    images.emplace_back(nullptr, captureMip0, vk::ImageLayout::eGeneral);
    writes.emplace_back(captureSet, 0, 0, 1, vk::DescriptorType::eStorageImage, &images.back());
    for (int i = 0; i < 2; ++i)
    {
        for (int m = 0; m < kMips; ++m)
        {
            vk::DescriptorSet& set = prefilterSets[size_t(i)][size_t(m)];
            set                    = prefilterPipeline.createRawDescriptorSet();
            images.emplace_back(nullptr, mipViews[size_t(i)][size_t(m)], vk::ImageLayout::eGeneral);
            writes.emplace_back(set, 0, 0, 1, vk::DescriptorType::eStorageImage, &images.back());
            writes.emplace_back(set, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &src);
        }
        shSets[size_t(i)]  = shPipeline.createRawDescriptorSet();
        buffers[size_t(i)] = vk::DescriptorBufferInfo(handle(shBuffers[size_t(i)]), 0, kShBytes);
        writes.emplace_back(shSets[size_t(i)], 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &src);
        writes.emplace_back(shSets[size_t(i)], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
                            &buffers[size_t(i)]);
    }
    base->device.updateDescriptorSets(writes, nullptr);

    // Sampled before the first capture completes (the dummy is not used: valid() is false): cleared, read only
    auto& q               = base->mainQueue;
    vk::CommandBuffer cmd = q.commandPool.createAndBeginOneTimeBuffer();
    for (Texture* t :
         {static_cast<Texture*>(&capture), static_cast<Texture*>(&cubes[0]), static_cast<Texture*>(&cubes[1])})
    {
        t->transitionImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
        cmd.clearColorImage(t->image(), vk::ImageLayout::eTransferDstOptimal,
                            vk::ClearColorValue(std::array<float, 4>{0, 0, 0, 1}),
                            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, t->mipLevels, 0, 6));
        t->transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    cmd.end();
    q.submitAndWait(cmd);
    q.commandPool.freeCommandBuffer(cmd);
    front          = 0;
    frontValid     = false;
    step           = 0;
    haveKey        = false;
    skyViewWritten = false;
}

void EnvironmentMap::destroyImages()
{
    if (!base) return;
    auto* pool   = &base->descriptorPool;
    auto device  = base->device;
    auto freeSet = [&](vk::DescriptorSet& s)
    {
        vk::DescriptorSet set = s;
        if (set) base->deferDelete([pool, set]() { pool->freeDescriptorSet(set); });
        s = nullptr;
    };
    auto freeView = [&](vk::ImageView& v)
    {
        vk::ImageView view = v;
        if (view) base->deferDelete([device, view]() { device.destroyImageView(view); });
        v = nullptr;
    };
    freeSet(captureSet);
    for (auto& sets : prefilterSets)
        for (auto& s : sets) freeSet(s);
    for (auto& s : shSets) freeSet(s);
    freeView(captureMip0);
    for (auto& views : mipViews)
        for (auto& v : views) freeView(v);
    capture.destroy();
    for (TextureCube& cube : cubes) cube.destroy();
    frontValid = false;
    step       = 0;
}

void EnvironmentMap::destroy()
{
    if (!base) return;
    destroyImages();
    auto* pool = &base->descriptorPool;
    for (auto& slot : debugSets)
        for (auto& set : slot)
        {
            vk::DescriptorSet d = set;
            if (d) base->deferDelete([pool, d]() { pool->freeDescriptorSet(d); });
            set = nullptr;
        }
    for (Texture2D& t : debugImages) t.destroy();
    capturePipeline.destroy();
    prefilterPipeline.destroy();
    shPipeline.destroy();
    debugPipeline.destroy();
    for (Buffer& sh : shBuffers) sh.destroy();
    linear.destroy();
    base = nullptr;
}

vk::DescriptorImageInfo EnvironmentMap::frontCubeInfo()
{
    return vk::DescriptorImageInfo(linear, cubes[size_t(front)].view(), vk::ImageLayout::eShaderReadOnlyOptimal);
}

vk::Buffer EnvironmentMap::frontSh() const
{
    return handle(shBuffers[size_t(front)]);
}

uint64_t EnvironmentMap::memoryBytes() const
{
    if (!base) return 0;
    // RGBA16F: the capture with its full chain (4/3) and the two cubes with their mips
    const uint64_t face = uint64_t(size) * uint64_t(size) * 8 * 6;
    return face * 4 / 3 + 2 * face * 4 / 3 + 2 * kShBytes;
}

bool EnvironmentMap::due(const Key& k, const LightingParameters& p) const
{
    if (!haveKey) return true;
    const Key& c    = current;
    const float deg = p.environmentUpdateDegrees;
    if (angleDeg(k.sunDir, c.sunDir) > deg || angleDeg(k.moonDir, c.moonDir) > deg) return true;
    if (relativeChange(k.sunE, c.sunE, 0.01f) || relativeChange(k.moonE, c.moonE, 0.01f)) return true;
    if (std::abs(k.cloudiness - c.cloudiness) > 0.01f) return true;
    if (std::abs(k.cameraHeight - c.cameraHeight) > 200.0f) return true;
    const AtmosphereSettings &a = k.atmosphere, &b = c.atmosphere;
    if (a.rayleighScale != b.rayleighScale || a.mieScale != b.mieScale || a.ozoneScale != b.ozoneScale ||
        a.groundAlbedo != b.groundAlbedo)
        return true;
    const CloudSettings &x = k.clouds, &y = c.clouds;
    if (x.altitude != y.altitude || x.radius != y.radius || x.seed != y.seed) return true;
    // The cloud drift: a fraction of the coverage texture's feature size (step a1c.3)
    if (std::abs(k.cloudOffsetX - c.cloudOffsetX) + std::abs(k.cloudOffsetY - c.cloudOffsetY) > 500.0f) return true;
    return k.enclosed != c.enclosed || k.cloudsOn != c.cloudsOn;
}

void EnvironmentMap::harvest()
{
    if (!shPending || base->completedFrameValue() < shReadyValue) return;
    float data[9][4];
    std::memcpy(data, shBuffers[size_t(front)].getMappedPointer(), sizeof(data));
    for (int i = 0; i < 9; ++i) frontShValues[size_t(i)] = vec3(data[i][0], data[i][1], data[i][2]);
    shPending = false;
}

void EnvironmentMap::recordCapture(vk::CommandBuffer cmd, vk::DescriptorImageInfo skyView, float cameraHeight)
{
    if (!skyViewWritten)
    {
        // The sky-view LUT is the same texture for the scene's life: written once (the set is not in use yet)
        base->device.updateDescriptorSets(
            vk::WriteDescriptorSet(captureSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &skyView), nullptr);
        skyViewWritten = true;
    }
    capture.transitionImageLayout(cmd, vk::ImageLayout::eGeneral);
    if (capturePipeline.bind(cmd))
    {
        CapturePush push{};
        push.size[0]   = size;
        push.camera[0] = std::clamp(6360.0f + cameraHeight * 1e-3f, 6360.001f, 6459.999f);
        if (captureOverride)
            for (int i = 0; i < 3; ++i) push.uniformSky[i] = (*captureOverride)(i);
        push.uniformSky[3] = captureOverride ? 1.0f : 0.0f;
        capturePipeline.bindRawDescriptorSet(cmd, captureSet);
        capturePipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
        cmd.dispatch(uint32_t(size + 7) / 8, uint32_t(size + 7) / 8, 6);
    }
    // The box mips (the prefilter's filtered importance sampling and the SH source)
    capture.recordGenerateMipmaps(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    // The back cube: its earlier reads (it was the front) before the writes
    cubes[size_t(1 - front)].transitionImageLayout(cmd, vk::ImageLayout::eGeneral);
}

void EnvironmentMap::recordPrefilter(vk::CommandBuffer cmd, int mip)
{
    const int back = 1 - front;
    if (!prefilterPipeline.bind(cmd)) return;
    PrefilterPush push{};
    const int n  = std::max(size >> mip, 1);
    push.size[0] = n;
    push.size[1] = size;
    push.size[2] = mip;
    push.size[3] = kMips - 1;
    prefilterPipeline.bindRawDescriptorSet(cmd, prefilterSets[size_t(back)][size_t(mip)]);
    prefilterPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
    cmd.dispatch(uint32_t(n + 7) / 8, uint32_t(n + 7) / 8, 6);
}

void EnvironmentMap::recordSh(vk::CommandBuffer cmd)
{
    const int back = 1 - front;
    // The back SH buffer's earlier reads (it was the front) before the write
    memoryBarrier(cmd, Stage::eFragmentShader | Stage::eComputeShader | Stage::eHost,
                  Access::eShaderStorageRead | Access::eHostRead, Stage::eComputeShader, Access::eShaderStorageWrite);
    if (shPipeline.bind(cmd))
    {
        ShPush push{};
        push.size[0] = kShSize;
        push.size[1] = int32_t(std::lround(std::log2(double(size) / kShSize)));
        shPipeline.bindRawDescriptorSet(cmd, shSets[size_t(back)]);
        shPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
        cmd.dispatch(1, 1, 1);
    }
    memoryBarrier(cmd, Stage::eComputeShader, Access::eShaderStorageWrite,
                  Stage::eFragmentShader | Stage::eComputeShader | Stage::eHost,
                  Access::eShaderStorageRead | Access::eHostRead);
    // The back cube is complete: read only
    cubes[size_t(back)].transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
}

Texture2D* EnvironmentMap::recordDebug(vk::CommandBuffer cmd, uint32_t frameIndex, int mode, int mip)
{
    if (!base) return nullptr;
    mip              = std::clamp(mip, 0, kMips - 1);
    const uint32_t n = uint32_t(std::max(size >> mip, 1));
    const uint32_t w = mode == 0 ? 4 * n : 128;
    const uint32_t h = mode == 0 ? 3 * n : 64;
    Texture2D& t     = debugImages[size_t(mode)];
    if (!t.memoryLocation || t.width != w || t.height != h)
    {
        t.destroy();
        t.create(*base, w, h, kFormat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);
    }
    vk::DescriptorSet set = debugSets[frameIndex][size_t(mode)];
    vk::DescriptorImageInfo dst(nullptr, t.view(), vk::ImageLayout::eGeneral);
    vk::DescriptorImageInfo cube = frontCubeInfo();
    vk::DescriptorBufferInfo sh(frontSh(), 0, kShBytes);
    std::array<vk::WriteDescriptorSet, 3> writes = {
        vk::WriteDescriptorSet(set, 0, 0, 1, vk::DescriptorType::eStorageImage, &dst),
        vk::WriteDescriptorSet(set, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &cube),
        vk::WriteDescriptorSet(set, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sh)};
    base->device.updateDescriptorSets(writes, nullptr);
    t.transitionImageLayout(cmd, vk::ImageLayout::eGeneral);
    if (debugPipeline.bind(cmd))
    {
        DebugPush push{{int32_t(w), int32_t(h), mode, mip}};
        debugPipeline.bindRawDescriptorSet(cmd, set);
        debugPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
        cmd.dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }
    t.transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    return &t;
}

void EnvironmentMap::record(vk::CommandBuffer cmd, const EnvironmentState& e, float cameraHeight,
                            const LightingParameters& p, vk::DescriptorImageInfo skyView, uint64_t frame)
{
    const int wanted = std::clamp(p.environmentResolution, int(kShSize), 512);
    if (wanted != size)
    {
        destroyImages();
        createImages(wanted);
    }
    harvest();
    Key k;
    k.sunDir       = e.sunDir;
    k.moonDir      = e.moonDir;
    k.sunE         = e.enclosed ? 0.0f : e.sunIlluminance;
    k.moonE        = e.enclosed ? 0.0f : e.moonIlluminance;
    k.cloudiness   = e.cloudiness;
    k.cameraHeight = cameraHeight;
    k.cloudOffsetX = e.cloudOffset(0);
    k.cloudOffsetY = e.cloudOffset(1);
    k.atmosphere   = e.atmosphere;
    k.clouds       = e.clouds;
    k.enclosed     = e.enclosed;
    k.cloudsOn     = p.clouds;
    stats.steps    = kMips + 2;

    // A new capture when due and none is in progress (a running capture finishes first: no mixed captures); the first
    // capture and forceFullUpdates do all steps in this frame
    const bool full = p.forceFullUpdates || !frontValid;
    if (p.forceFullUpdates) step = 0;
    if (step == 0 && (p.forceFullUpdates || due(k, p)))
    {
        current = k;
        haveKey = true;
        recordCapture(cmd, skyView, cameraHeight);
        stats.captures++;
        step = 1;
        if (!full)
        {
            stats.progress = 1;
            return;
        }
    }
    if (step == 0)
    {
        stats.progress = 0;
        return;
    }
    // The prefilter, one mip per frame (all with full), then the SH and the swap
    while (step >= 1 && step <= kMips)
    {
        recordPrefilter(cmd, step - 1);
        step++;
        if (!full) break;
    }
    if (step == kMips + 1 && (full || stats.progress == kMips + 1))
    {
        recordSh(cmd);
        front                   = 1 - front;
        frontValid              = true;
        step                    = 0;
        shPending               = true;
        shReadyValue            = base->currentFrameSignalValue();
        stats.lastCompleteFrame = frame;
        stats.progress          = 0;
        stats.valid             = true;
        return;
    }
    stats.progress = step;
}

}  // namespace Vulkan
}  // namespace Saiga
