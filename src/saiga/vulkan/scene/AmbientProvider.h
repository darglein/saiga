/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/vulkan/svulkan.h"

#include <array>

namespace Saiga
{
namespace Vulkan
{
struct EnvironmentState;

// What the ambient light of a frame is (statistics, the Lighting tab, get_lighting): the irradiance SH9 (lux, rgb per
// coefficient in the order of ambient.glsl) and its dominant colour (the SH's band 0)
struct AmbientStatistics
{
    std::array<vec3, 9> sh = {};
    vec3 dominant          = vec3(0, 0, 0);
};

// The ambient's resources of set 1 (bound by SceneLightingSet::write): the SH buffer (binding 2) and the prefiltered
// environment cube (binding 4; null until the environment map of a1c: the dummy)
struct AmbientBindings
{
    vk::Buffer sh;
    vk::DeviceSize shBytes = 0;
    vk::DescriptorImageInfo environment;
};

/**
 * The ambient light of RenderScene (A3 of the S36 plan, docs/engine_design.md §10.7): per frame the SH9 irradiance and
 * the specular environment the lighting reads through ambient.glsl. SkyAmbient is the only implementation (the sky);
 * probes or occlusion methods replace it later without shader changes outside ambient.glsl.
 */
class AmbientProvider
{
   public:
    virtual ~AmbientProvider() = default;
    // The frame slot's resources for this frame's environment (main thread, the slot's previous frame has completed)
    virtual void update(vk::CommandBuffer cmd, uint32_t frameIndex, const EnvironmentState& environment) = 0;
    virtual AmbientBindings bindings(uint32_t frameIndex)                                                = 0;
    virtual AmbientStatistics statistics() const                                                         = 0;
};

}  // namespace Vulkan
}  // namespace Saiga
