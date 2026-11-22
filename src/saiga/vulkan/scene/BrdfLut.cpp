/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/BrdfLut.h"

#include "saiga/core/util/assert.h"
#include "saiga/vulkan/buffer/StagingBuffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/pipeline/DescriptorSetLayout.h"
#include "saiga/vulkan/scene/LightingDebug.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

namespace Saiga
{
namespace Vulkan
{
namespace
{
struct LutPush
{
    int32_t size;
    uint32_t samples;
};

// The texels of the first LUT computed in the process (RG16F, row-major) and the number of GPU computations
struct LutCache
{
    std::mutex mutex;
    std::vector<uint8_t> bytes;
    int computations = 0;
};
LutCache& lutCache()
{
    static LutCache cache;
    return cache;
}

double visibility(double NoV, double NoL, double a)
{
    const double a2 = a * a;
    const double gv = NoL * std::sqrt(NoV * NoV * (1 - a2) + a2);
    const double gl = NoV * std::sqrt(NoL * NoL * (1 - a2) + a2);
    return 0.5 / std::max(gv + gl, 1e-7);
}
}  // namespace

int BrdfLut::computations()
{
    std::lock_guard<std::mutex> lock(lutCache().mutex);
    return lutCache().computations;
}

void BrdfLut::init(VulkanBase& base)
{
    SamplerParameters sp;
    sp.setAddressMode(vk::SamplerAddressMode::eClampToEdge);
    sampler.create(base, sp);
    LutCache& cache = lutCache();
    std::vector<uint8_t> bytes;
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        bytes = cache.bytes;
    }
    if (!bytes.empty()) return upload(base, bytes);
    compute(base);
    bytes = texture.download();
    SAIGA_ASSERT(bytes.size() == size_t(kSize) * kSize * 4);
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.computations++;
    if (cache.bytes.empty()) cache.bytes = std::move(bytes);
}

void BrdfLut::upload(VulkanBase& base, const std::vector<uint8_t>& bytes)
{
    texture.create(base, kSize, kSize, vk::Format::eR16G16Sfloat,
                   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage);
    StagingBuffer staging;
    staging.init(base, bytes.size(), bytes.data());
    auto& q               = base.mainQueue;
    vk::CommandBuffer cmd = q.commandPool.createAndBeginOneTimeBuffer();
    texture.transitionImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
    vk::BufferImageCopy region(0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                               vk::Offset3D(0, 0, 0), vk::Extent3D(kSize, kSize, 1));
    staging.copyTo(cmd, texture.image(), vk::ImageLayout::eTransferDstOptimal, region);
    texture.transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    cmd.end();
    q.submitAndWait(cmd);
    q.commandPool.freeCommandBuffer(cmd);
    staging.destroyImmediately();
}

void BrdfLut::compute(VulkanBase& base)
{
    texture.create(base, kSize, kSize, vk::Format::eR16G16Sfloat,
                   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage);
    ComputePipeline pipeline;
    pipeline.init(base, 1);
    DescriptorSetLayout layout;
    layout.addBinding(
        0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute));
    pipeline.addDescriptorSetLayout(layout);
    pipeline.addPushConstantRange({vk::ShaderStageFlagBits::eCompute, 0, sizeof(LutPush)});
    pipeline.shaderPipeline.loadCompute(base.device, "vulkan/scene/lighting/brdf_lut.comp");
    pipeline.create();
    vk::DescriptorSet set = pipeline.createRawDescriptorSet();
    vk::DescriptorImageInfo image(nullptr, texture.view(), vk::ImageLayout::eGeneral);
    base.device.updateDescriptorSets(vk::WriteDescriptorSet(set, 0, 0, 1, vk::DescriptorType::eStorageImage, &image),
                                     nullptr);

    auto& q               = base.mainQueue;
    vk::CommandBuffer cmd = q.commandPool.createAndBeginOneTimeBuffer();
    texture.transitionImageLayout(cmd, vk::ImageLayout::eGeneral);
    if (pipeline.bind(cmd))
    {
        LutPush push{kSize, kSamples};
        pipeline.bindRawDescriptorSet(cmd, set);
        pipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
        cmd.dispatch((kSize + 7) / 8, (kSize + 7) / 8, 1);
    }
    texture.transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    cmd.end();
    q.submitAndWait(cmd);
    q.commandPool.freeCommandBuffer(cmd);
    base.descriptorPool.freeDescriptorSet(set);
    pipeline.destroy();
}

std::vector<vec2> BrdfLut::read()
{
    std::vector<uint8_t> bytes = texture.download();
    SAIGA_ASSERT(bytes.size() == size_t(kSize) * kSize * 4);
    std::vector<vec2> out(size_t(kSize) * kSize);
    for (size_t i = 0; i < out.size(); ++i)
    {
        uint16_t h[2];
        std::memcpy(h, bytes.data() + i * 4, 4);
        out[i] = vec2(LightingDebug::halfToFloat(h[0]), LightingDebug::halfToFloat(h[1]));
    }
    return out;
}

vec2 BrdfLut::integrate(float NdotV, float roughness, uint32_t samples)
{
    const double pi  = 3.14159265358979323846;
    const double NoV = NdotV, a = double(roughness) * roughness;
    const double vx = std::sqrt(1 - NoV * NoV), vz = NoV;
    double A = 0, B = 0;
    for (uint32_t i = 0; i < samples; ++i)
    {
        // Hammersley (radical inverse base 2)
        uint32_t b      = i;
        b               = (b << 16u) | (b >> 16u);
        b               = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
        b               = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
        b               = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
        b               = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
        const double x1 = double(i) / samples, x2 = double(b) * 2.3283064365386963e-10;
        const double phi = 2 * pi * x1;
        const double ct  = std::sqrt((1 - x2) / (1 + (a * a - 1) * x2));
        const double st  = std::sqrt(1 - ct * ct);
        // The half vector (hx, hy, hz) with v in the xz plane: hy does not enter VoH or l.z
        const double hx = st * std::cos(phi), hz = ct;
        const double VoH = vx * hx + vz * hz;
        const double lz  = 2 * VoH * hz - vz;
        const double NoL = std::max(lz, 0.0), NoH = std::max(hz, 0.0), vh = std::max(VoH, 0.0);
        if (NoL > 0)
        {
            const double w  = visibility(NoV, NoL, a) * 4 * NoL * vh / NoH;
            const double fc = std::pow(1 - vh, 5.0);
            A += (1 - fc) * w;
            B += fc * w;
        }
    }
    return vec2(float(A / samples), float(B / samples));
}

}  // namespace Vulkan
}  // namespace Saiga
