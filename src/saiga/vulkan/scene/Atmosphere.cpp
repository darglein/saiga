/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/Atmosphere.h"

#include "saiga/vulkan/pipeline/DescriptorSetLayout.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

namespace
{
// The push constants of atmosphere_*.comp
struct AtmospherePush
{
    float scales[4];  // Rayleigh, Mie, ozone, ground albedo
    float sun[4];     // towards the sun, illuminance (lux)
    float moon[4];
    float camera[4];  // x: the camera's radius (km), y: the cloudiness
    int32_t size[4];
};

constexpr float kGroundRadiusKm = 6360.0f, kTopRadiusKm = 6460.0f;
constexpr float kPiF = 3.14159265358979f;

bool sameSettings(const AtmosphereSettings& a, const AtmosphereSettings& b)
{
    return a.rayleighScale == b.rayleighScale && a.mieScale == b.mieScale && a.ozoneScale == b.ozoneScale &&
           a.groundAlbedo == b.groundAlbedo;
}

// The barriers around a compute write of a whole image: earlier reads (fragment, compute) -> the write (the content is
// discarded), the write -> fragment and compute reads
void beginWrite(vk::CommandBuffer cmd, Texture2D& t)
{
    vk::ImageMemoryBarrier2 b(Stage::eFragmentShader | Stage::eComputeShader, Access::eShaderSampledRead,
                              Stage::eComputeShader, Access::eShaderStorageWrite, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eGeneral, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, t.image(),
                              vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, b));
}
void endWrite(vk::CommandBuffer cmd, Texture2D& t)
{
    vk::ImageMemoryBarrier2 b(Stage::eComputeShader, Access::eShaderStorageWrite,
                              Stage::eFragmentShader | Stage::eComputeShader, Access::eShaderSampledRead,
                              vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
                              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, t.image(),
                              vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, b));
    t.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
}
}  // namespace

void Atmosphere::init(VulkanBase& b)
{
    destroy();
    base             = &b;
    const auto usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
    transmittanceLut.create(b, kTransmittanceWidth, kTransmittanceHeight, vk::Format::eR16G16B16A16Sfloat, usage);
    multiScatteringLut.create(b, kMultiScatteringSize, kMultiScatteringSize, vk::Format::eR16G16B16A16Sfloat, usage);
    skyViewLut.create(b, kSkyViewWidth, kSkyViewHeight, vk::Format::eR16G16B16A16Sfloat, usage);
    SamplerParameters sp;
    sp.setFilter(vk::Filter::eLinear);
    sp.setAddressMode(vk::SamplerAddressMode::eClampToEdge);
    linearClamp.create(b, sp);
    sp.addressU = vk::SamplerAddressMode::eRepeat;
    skyViewSampler.create(b, sp);

    // One layout for the three passes: 0 the output image, 1 the transmittance, 2 the multiple scattering
    DescriptorSetLayout layout;
    const auto c = vk::ShaderStageFlagBits::eCompute;
    layout.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageImage, 1, c));
    layout.addBinding(1, vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, c));
    layout.addBinding(2, vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eCombinedImageSampler, 1, c));
    auto create = [&](ComputePipeline& p, const char* shader)
    {
        p.init(b, 1);
        p.addDescriptorSetLayout(layout);
        p.addPushConstantRange({c, 0, sizeof(AtmospherePush)});
        p.shaderPipeline.loadCompute(b.device, shader);
        p.create();
    };
    create(transmittancePipeline, "vulkan/scene/lighting/atmosphere_transmittance.comp");
    create(multiScatteringPipeline, "vulkan/scene/lighting/atmosphere_multiscatter.comp");
    create(skyViewPipeline, "vulkan/scene/lighting/atmosphere_skyview.comp");
    transmittanceSet   = transmittancePipeline.createRawDescriptorSet();
    multiScatteringSet = multiScatteringPipeline.createRawDescriptorSet();
    skyViewSet         = skyViewPipeline.createRawDescriptorSet();
    auto write         = [&](vk::DescriptorSet set, Texture2D& out)
    {
        vk::DescriptorImageInfo dst(nullptr, out.view(), vk::ImageLayout::eGeneral);
        vk::DescriptorImageInfo t(linearClamp, transmittanceLut.view(), vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorImageInfo m(linearClamp, multiScatteringLut.view(), vk::ImageLayout::eShaderReadOnlyOptimal);
        std::array<vk::WriteDescriptorSet, 3> writes = {
            vk::WriteDescriptorSet(set, 0, 0, 1, vk::DescriptorType::eStorageImage, &dst),
            vk::WriteDescriptorSet(set, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &t),
            vk::WriteDescriptorSet(set, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &m)};
        b.device.updateDescriptorSets(writes, nullptr);
    };
    write(transmittanceSet, transmittanceLut);
    write(multiScatteringSet, multiScatteringLut);
    write(skyViewSet, skyViewLut);

    // The LUTs are sampled before their first computation (the lighting set binds them): cleared, read only
    auto& q               = b.mainQueue;
    vk::CommandBuffer cmd = q.commandPool.createAndBeginOneTimeBuffer();
    for (Texture2D* t : {&transmittanceLut, &multiScatteringLut, &skyViewLut})
    {
        t->transitionImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);
        cmd.clearColorImage(t->image(), vk::ImageLayout::eTransferDstOptimal,
                            vk::ClearColorValue(std::array<float, 4>{0, 0, 0, 1}),
                            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
        t->transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    cmd.end();
    q.submitAndWait(cmd);
    q.commandPool.freeCommandBuffer(cmd);
    valid = false;
}

void Atmosphere::destroy()
{
    if (!base) return;
    auto* pool = &base->descriptorPool;
    for (vk::DescriptorSet* s : {&transmittanceSet, &multiScatteringSet, &skyViewSet})
    {
        vk::DescriptorSet set = *s;
        if (set) base->deferDelete([pool, set]() { pool->freeDescriptorSet(set); });
        *s = nullptr;
    }
    transmittancePipeline.destroy();
    multiScatteringPipeline.destroy();
    skyViewPipeline.destroy();
    transmittanceLut.destroy();
    multiScatteringLut.destroy();
    skyViewLut.destroy();
    linearClamp.destroy();
    skyViewSampler.destroy();
    base = nullptr;
}

float Atmosphere::radiusOf(float heightM)
{
    return std::clamp(kGroundRadiusKm + heightM * 1e-3f, kGroundRadiusKm + 1e-3f, kTopRadiusKm - 1e-3f);
}

float Atmosphere::skyViewHorizon(float radiusKm)
{
    return -std::acos(std::clamp(kGroundRadiusKm / radiusKm, 0.0f, 1.0f));
}

vec2 Atmosphere::skyViewUv(const vec3& dir, float horizon)
{
    const float halfPi = 0.5f * kPiF;
    const float el     = std::asin(std::clamp(dir(1), -1.0f, 1.0f));
    float v;
    if (el >= horizon)
        v = 0.5f + 0.5f * std::sqrt(std::clamp((el - horizon) / (halfPi - horizon), 0.0f, 1.0f));
    else
        v = 0.5f - 0.5f * std::sqrt(std::clamp((horizon - el) / (horizon + halfPi), 0.0f, 1.0f));
    const float u = std::atan2(dir(2), dir(0)) / (2.0f * kPiF) + 0.5f;
    return vec2(u, v);
}

vec3 Atmosphere::skyViewDir(const vec2& uv, float horizon)
{
    const float halfPi = 0.5f * kPiF;
    float el;
    if (uv(1) >= 0.5f)
    {
        const float t = (uv(1) - 0.5f) * 2.0f;
        el            = horizon + t * t * (halfPi - horizon);
    }
    else
    {
        const float t = (0.5f - uv(1)) * 2.0f;
        el            = horizon - t * t * (horizon + halfPi);
    }
    const float az = (uv(0) - 0.5f) * 2.0f * kPiF;
    return vec3(std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az));
}

uint64_t Atmosphere::memoryBytes() const
{
    return 8ull * (kTransmittanceWidth * kTransmittanceHeight + kMultiScatteringSize * kMultiScatteringSize +
                   kSkyViewWidth * kSkyViewHeight);
}

void Atmosphere::record(vk::CommandBuffer cmd, const EnvironmentState& e, float cameraHeight)
{
    const AtmosphereSettings& a = e.atmosphere;
    AtmospherePush push{};
    push.scales[0] = a.rayleighScale;
    push.scales[1] = a.mieScale;
    push.scales[2] = a.ozoneScale;
    push.scales[3] = a.groundAlbedo;
    for (int i = 0; i < 3; ++i)
    {
        push.sun[i]  = e.sunDir(i);
        push.moon[i] = e.moonDir(i);
    }
    push.sun[3]    = e.enclosed ? 0.0f : e.sunIlluminance;
    push.moon[3]   = e.enclosed ? 0.0f : e.moonIlluminance;
    push.camera[0] = radiusOf(cameraHeight);
    push.camera[1] = e.cloudiness;
    auto dispatch  = [&](ComputePipeline& p, vk::DescriptorSet set, Texture2D& out)
    {
        push.size[0] = int32_t(out.width);
        push.size[1] = int32_t(out.height);
        beginWrite(cmd, out);
        if (p.bind(cmd))
        {
            p.bindRawDescriptorSet(cmd, set);
            p.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
            cmd.dispatch((out.width + 7) / 8, (out.height + 7) / 8, 1);
        }
        endWrite(cmd, out);
    };
    if (!valid || !sameSettings(a, computedFor))
    {
        dispatch(transmittancePipeline, transmittanceSet, transmittanceLut);
        dispatch(multiScatteringPipeline, multiScatteringSet, multiScatteringLut);
        valid       = true;
        computedFor = a;
        computations++;
    }
    dispatch(skyViewPipeline, skyViewSet, skyViewLut);
    skyViewHorizonRad = skyViewHorizon(push.camera[0]);
}

}  // namespace Vulkan
}  // namespace Saiga
