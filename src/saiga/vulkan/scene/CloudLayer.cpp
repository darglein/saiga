/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/CloudLayer.h"

#include "saiga/vulkan/pipeline/DescriptorSetLayout.h"
#include "saiga/vulkan/scene/SceneLightingSet.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

namespace
{
// The noise of the last seed generated in the process (R8, kSize^2): a new scene's first texture is a copy (the
// generation cost about 22 ms per RenderScene on lavapipe; the tests create a scene per test). Deterministic data of
// the seed, as BrdfLut's cache
struct NoiseCache
{
    std::mutex mutex;
    std::optional<uint32_t> seed;
    std::vector<uint8_t> bytes;
};
NoiseCache& noiseCache()
{
    static NoiseCache cache;
    return cache;
}

struct NoisePush
{
    int32_t size[4];  // size, mode, seed, blob
    float blob[4];
    float clouds[4];
    float origin[4];
    float light[4];
};
struct LayerPush
{
    mat4 viewProj;
    vec4 params;  // pre-exposure, rings, segments
};

vk::Buffer handle(const Buffer& b)
{
    Memory::SafeAccessor acc(*b.memoryLocation());
    return b.memoryLocation()->data;
}

float smoothstep(float a, float b, float x)
{
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3 - 2 * t);
}
}  // namespace

float CloudLayer::coverageFromNoise(float n, float cloudiness)
{
    return cloudiness <= 0 ? 0.0f : smoothstep(1 - cloudiness - 0.1f, 1 - cloudiness + 0.1f, n);
}

float CloudLayer::edgeAlpha(float r01)
{
    return 1 - smoothstep(0.7f, 1.0f, r01);
}

void CloudLayer::init(VulkanBase& b, const RenderingFormats& formats, uint32_t seed)
{
    destroy();
    base = &b;
    coverage.create(b, kSize, kSize, vk::Format::eR8Unorm, vk::ImageUsageFlagBits::eSampled,
                    Texture::fullMipLevels(kSize, kSize, 1));
    SamplerParameters sp;
    sp.setFilter(vk::Filter::eLinear);
    sp.setAddressMode(vk::SamplerAddressMode::eRepeat);
    linearRepeat.create(b, sp);
    noise.createBuffer(b, size_t(kSize) * kSize,
                       vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    const auto c = vk::ShaderStageFlagBits::eCompute;
    DescriptorSetLayout l;
    l.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, c));
    l.addBinding(1, vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, c));
    l.addBinding(2, vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageImage, 1, c));
    noisePipeline.init(b, 1);
    noisePipeline.addDescriptorSetLayout(l);
    noisePipeline.addPushConstantRange({c, 0, sizeof(NoisePush)});
    noisePipeline.shaderPipeline.loadCompute(b.device, "vulkan/scene/lighting/cloud_noise.comp");
    noisePipeline.create();
    for (auto& s : noiseSets) s = noisePipeline.createRawDescriptorSet();
    {
        // The generation's set: the buffer, and 1x1 dummies for the preview bindings (statically used by the shader)
        dummySampled.create(b, 1, 1, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eSampled);
        dummyStorage.create(b, 1, 1, vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eStorage);
        auto& q                 = b.mainQueue;
        vk::CommandBuffer setup = q.commandPool.createAndBeginOneTimeBuffer();
        dummySampled.transitionImageLayout(setup, vk::ImageLayout::eTransferDstOptimal);
        setup.clearColorImage(dummySampled.image(), vk::ImageLayout::eTransferDstOptimal,
                              vk::ClearColorValue(std::array<float, 4>{0, 0, 0, 0}),
                              vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
        dummySampled.transitionImageLayout(setup, vk::ImageLayout::eShaderReadOnlyOptimal);
        dummyStorage.transitionImageLayout(setup, vk::ImageLayout::eGeneral);
        setup.end();
        q.submitAndWait(setup);
        q.commandPool.freeCommandBuffer(setup);
        vk::DescriptorBufferInfo buf(handle(noise), 0, VK_WHOLE_SIZE);
        vk::DescriptorImageInfo src(linearRepeat, dummySampled.view(), vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorImageInfo dst(nullptr, dummyStorage.view(), vk::ImageLayout::eGeneral);
        std::array<vk::WriteDescriptorSet, 3> writes = {
            vk::WriteDescriptorSet(noiseSets[0], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buf),
            vk::WriteDescriptorSet(noiseSets[0], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &src),
            vk::WriteDescriptorSet(noiseSets[0], 2, 0, 1, vk::DescriptorType::eStorageImage, &dst)};
        b.device.updateDescriptorSets(writes, nullptr);
    }

    // The layer: set 0 unused (an empty layout), set 1 the lighting set; premultiplied alpha, depth test (reversed-Z),
    // no depth write, both faces
    PipelineInfo info;
    info.rasterizationState.cullMode              = vk::CullModeFlagBits::eNone;
    info.depthStencilState.depthTestEnable        = VK_TRUE;
    info.depthStencilState.depthWriteEnable       = VK_FALSE;
    info.depthStencilState.depthCompareOp         = vk::CompareOp::eGreaterOrEqual;
    info.blendAttachmentState.blendEnable         = VK_TRUE;
    info.blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
    info.blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    info.blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    info.blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    layerPipeline.init(b, 2);
    layerPipeline.addDescriptorSetLayout(DescriptorSetLayout(), 0);
    layerPipeline.addDescriptorSetLayout(SceneLightingSet::layout(), 1);
    layerPipeline.addPushConstantRange(
        {vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(LayerPush)});
    layerPipeline.shaderPipeline.load(
        b.device, {"vulkan/scene/lighting/cloud_layer.vert", "vulkan/scene/lighting/cloud_layer.frag"});
    layerPipeline.create(formats, info);

    // The first texture at once (blocking): sampled from the first frame
    auto& q               = b.mainQueue;
    vk::CommandBuffer cmd = q.commandPool.createAndBeginOneTimeBuffer();
    generate(cmd, seed, true);
    cmd.end();
    q.submitAndWait(cmd);
    q.commandPool.freeCommandBuffer(cmd);
    pendingValue = 0;
    harvest();
}

void CloudLayer::destroy()
{
    if (!base) return;
    auto* pool = &base->descriptorPool;
    for (auto& s : noiseSets)
    {
        vk::DescriptorSet set = s;
        if (set) base->deferDelete([pool, set]() { pool->freeDescriptorSet(set); });
        s = nullptr;
    }
    noisePipeline.destroy();
    layerPipeline.destroy();
    coverage.destroy();
    for (Texture2D& t : debugImages) t.destroy();
    dummySampled.destroy();
    dummyStorage.destroy();
    linearRepeat.destroy();
    noise.destroy();
    generatedSeed.reset();
    base = nullptr;
}

uint64_t CloudLayer::memoryBytes() const
{
    return base ? uint64_t(kSize) * kSize * 4 / 3 + uint64_t(kSize) * kSize : 0;
}

void CloudLayer::generate(vk::CommandBuffer cmd, uint32_t seed, bool fromCache)
{
    bool cached = false;
    if (fromCache && !testBlob)
    {
        NoiseCache& cache = noiseCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.seed == seed && cache.bytes.size() == size_t(kSize) * kSize)
        {
            std::memcpy(noise.getMappedPointer(), cache.bytes.data(), cache.bytes.size());
            cached = true;
        }
    }
    // The previous copy's reads of the buffer and the host's read before the writes
    vk::MemoryBarrier2 before(Stage::eAllTransfer | Stage::eHost, Access::eTransferRead | Access::eHostRead,
                              Stage::eComputeShader, Access::eShaderStorageWrite);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, before, nullptr, nullptr));
    if (!cached && noisePipeline.bind(cmd))
    {
        NoisePush push{};
        push.size[0] = int32_t(kSize);
        push.size[1] = 0;
        push.size[2] = int32_t(seed);
        push.size[3] = testBlob ? 1 : 0;
        if (testBlob)
            for (int i = 0; i < 3; ++i) push.blob[i] = (*testBlob)(i);
        noisePipeline.bindRawDescriptorSet(cmd, noiseSets[0]);
        noisePipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
        cmd.dispatch(kSize / 4 / 8, kSize / 8, 1);
    }
    vk::MemoryBarrier2 after(Stage::eComputeShader, Access::eShaderStorageWrite, Stage::eAllTransfer | Stage::eHost,
                             Access::eTransferRead | Access::eHostRead);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, after, nullptr, nullptr));
    coverage.transitionImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
    vk::BufferImageCopy region(0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), {0, 0, 0},
                               {kSize, kSize, 1});
    cmd.copyBufferToImage(handle(noise), coverage.image(), vk::ImageLayout::eTransferDstOptimal, region);
    coverage.recordGenerateMipmaps(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    generatedSeed    = seed;
    generatedBlob    = testBlob;
    histogramPending = true;
    pendingValue     = base->currentFrameSignalValue();
    generationCount++;
}

void CloudLayer::record(vk::CommandBuffer cmd, uint32_t seed)
{
    harvest();
    const bool blobChanged =
        testBlob.has_value() != generatedBlob.has_value() || (testBlob && generatedBlob && *testBlob != *generatedBlob);
    if (generatedSeed && *generatedSeed == seed && !blobChanged) return;
    generate(cmd, seed);
}

void CloudLayer::harvest()
{
    if (!histogramPending || (pendingValue != 0 && base->completedFrameValue() < pendingValue)) return;
    histogram.fill(0);
    const uint8_t* bytes = static_cast<const uint8_t*>(noise.getMappedPointer());
    for (size_t i = 0; i < size_t(kSize) * kSize; ++i) histogram[bytes[i]]++;
    histogramTotal   = kSize * kSize;
    histogramPending = false;
    if (generatedSeed && !generatedBlob)
    {
        NoiseCache& cache = noiseCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.seed != generatedSeed)
        {
            cache.seed = generatedSeed;
            cache.bytes.assign(bytes, bytes + size_t(kSize) * kSize);
        }
    }
}

float CloudLayer::effectiveCloudiness(float cloudiness) const
{
    // The test blob is not noise: its cloudiness is the coverage threshold itself
    if (histogramTotal == 0 || cloudiness <= 0 || generatedBlob) return std::max(cloudiness, 0.0f);
    if (cloudiness >= 1) return 1;
    // The noise value below which (1 - cloudiness) of the texels lie (linear inside the bin)
    const double target = (1.0 - double(cloudiness)) * histogramTotal;
    double cumulative   = 0;
    for (int b = 0; b < 256; ++b)
    {
        const double next = cumulative + histogram[size_t(b)];
        if (next >= target && histogram[size_t(b)] > 0)
        {
            const double t = (target - cumulative) / histogram[size_t(b)];
            return std::clamp(1.0f - float((b - 0.5 + t) / 255.0), 0.0f, 1.0f);
        }
        cumulative = next;
    }
    return 0;
}

float CloudLayer::meanCoverage(float cloudiness) const
{
    if (histogramTotal == 0) return 0;
    double sum = 0;
    for (int b = 0; b < 256; ++b)
        sum += double(histogram[size_t(b)]) * coverageFromNoise(float(b) / 255.0f, cloudiness);
    return float(sum / histogramTotal);
}

void CloudLayer::draw(vk::CommandBuffer cmd, vk::DescriptorSet lightingSet, const mat4& viewProj, float preExposure)
{
    if (!layerPipeline.bind(cmd)) return;
    LayerPush push{viewProj, vec4(preExposure, float(kRings), float(kSegments), 0)};
    layerPipeline.bindRawDescriptorSet(cmd, lightingSet, 1);
    layerPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, sizeof(push),
                               &push);
    cmd.draw(uint32_t(kRings * kSegments * 6), 1, 0, 0);
}

Texture2D* CloudLayer::recordDebug(vk::CommandBuffer cmd, uint32_t frameIndex, int mode, const DebugInput& in)
{
    if (!base || mode < 1 || mode > 2) return nullptr;
    const uint32_t n = mode == 1 ? 512 : 256;
    Texture2D& t     = debugImages[size_t(mode - 1)];
    if (!t.memoryLocation)
        t.create(*base, n, n, vk::Format::eR16G16B16A16Sfloat,
                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);
    // One set per frame slot and mode (the slot's previous frame has completed)
    vk::DescriptorSet set = noiseSets[1 + frameIndex * 2 + uint32_t(mode - 1)];
    vk::DescriptorBufferInfo buf(handle(noise), 0, VK_WHOLE_SIZE);
    vk::DescriptorImageInfo src = coverageInfo();
    vk::DescriptorImageInfo dst(nullptr, t.view(), vk::ImageLayout::eGeneral);
    std::array<vk::WriteDescriptorSet, 3> writes = {
        vk::WriteDescriptorSet(set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buf),
        vk::WriteDescriptorSet(set, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &src),
        vk::WriteDescriptorSet(set, 2, 0, 1, vk::DescriptorType::eStorageImage, &dst)};
    base->device.updateDescriptorSets(writes, nullptr);
    t.transitionImageLayout(cmd, vk::ImageLayout::eGeneral);
    if (noisePipeline.bind(cmd))
    {
        NoisePush push{};
        push.size[0] = int32_t(n);
        push.size[1] = mode;
        push.blob[0] = in.clouds(3);
        for (int i = 0; i < 4; ++i) push.clouds[i] = in.clouds(i);
        push.origin[0] = in.origin(0);
        push.origin[1] = in.origin(1);
        push.origin[2] = in.mean;
        push.origin[3] = in.height;
        for (int i = 0; i < 3; ++i) push.light[i] = in.light(i);
        noisePipeline.bindRawDescriptorSet(cmd, set);
        noisePipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
        cmd.dispatch(n / 8, n / 8, 1);
    }
    t.transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    return &t;
}

}  // namespace Vulkan
}  // namespace Saiga
