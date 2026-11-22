/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/scene/LightingTypes.h"
#include "saiga/vulkan/texture/Sampler.h"
#include "saiga/vulkan/texture/Texture.h"

namespace Saiga
{
namespace Vulkan
{
struct EnvironmentState;

/**
 * The physically based atmosphere of RenderScene (S36, A4 of docs/stages/S36_lighting.plan.md, D2; Hillaire 2020):
 * the precomputed LUTs of the Earth's Rayleigh, Mie and ozone medium (AtmosphereSettings' scales;
 * shader/vulkan/scene/lighting/atmosphere_*.comp): the transmittance LUT (256 x 64) and the multiple-scattering LUT
 * (32 x 32), recomputed only when the parameters change, and the sky-view LUT (192 x 108, the sun's and the moon's
 * in-scattered radiance times kSkyLutScale) every frame at the camera height. Bindings 6 and 7 of the lighting set; the
 * sky pass, the fog colour, the environment map and the direct light's transmittance read them. Recorded in the frame's
 * "environment" section.
 */
class SAIGA_VULKAN_API Atmosphere
{
   public:
    // One constant per declaration: MSVC rejects a second declarator of a static member of a dll interface class
    // (C2487)
    static constexpr uint32_t kTransmittanceWidth  = 256;
    static constexpr uint32_t kTransmittanceHeight = 64;
    static constexpr uint32_t kMultiScatteringSize = 32;
    static constexpr uint32_t kSkyViewWidth        = 192;
    static constexpr uint32_t kSkyViewHeight       = 108;
    // The sky-view LUT and the environment cube store radiance times this (shader: kSkyLutScale)
    static constexpr float kSkyLutScale = 1.0f / 16.0f;

    void init(VulkanBase& base);
    void destroy();
    ~Atmosphere() { destroy(); }

    // Records the LUTs of this frame: the transmittance and multiple scattering when the settings changed (or the first
    // time), the sky-view LUT for the environment's sun and moon at the camera height (m above the sea level). The LUTs
    // are in eShaderReadOnlyOptimal afterwards, visible to the fragment and compute shaders
    void record(vk::CommandBuffer cmd, const EnvironmentState& environment, float cameraHeight);

    Texture2D& transmittance() { return transmittanceLut; }
    Texture2D& multiScattering() { return multiScatteringLut; }
    Texture2D& skyView() { return skyViewLut; }
    vk::DescriptorImageInfo transmittanceInfo() { return transmittanceLut.getDescriptorInfo(linearClamp); }
    vk::DescriptorImageInfo skyViewInfo() { return skyViewLut.getDescriptorInfo(skyViewSampler); }
    // The LUT recomputations so far (tests)
    uint64_t lutComputations() const { return computations; }
    // The sky-view LUT's horizon elevation (rad; the block's atmosphere.w): of the radius it was computed at
    float skyViewHorizonElevation() const { return skyViewHorizonRad; }

    // The CPU copy of the sky-view LUT's mapping (atmosphere_common.glsl; tests): the horizon elevation (rad, below 0)
    // seen from the radius (km), the uv of a direction (x the azimuth atan(z, x), y the elevation: the horizon at 0.5,
    // more rows near it) and back
    static float skyViewHorizon(float radiusKm);
    static vec2 skyViewUv(const vec3& dir, float horizon);
    static vec3 skyViewDir(const vec2& uv, float horizon);
    // The radius (km) of a height above the sea level (m) inside the atmosphere (atmosphereRadius)
    static float radiusOf(float heightM);
    uint64_t memoryBytes() const;

   private:
    VulkanBase* base = nullptr;
    Texture2D transmittanceLut, multiScatteringLut, skyViewLut;
    Sampler linearClamp, skyViewSampler;
    ComputePipeline transmittancePipeline, multiScatteringPipeline, skyViewPipeline;
    vk::DescriptorSet transmittanceSet, multiScatteringSet, skyViewSet;
    bool valid = false;
    AtmosphereSettings computedFor;
    uint64_t computations   = 0;
    float skyViewHorizonRad = 0;
};

}  // namespace Vulkan
}  // namespace Saiga
