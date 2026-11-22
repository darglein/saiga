/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/scene/AmbientProvider.h"

#include <optional>

namespace Saiga
{
namespace Vulkan
{
/**
 * The sky's ambient light (A3 of the S36 plan): the SH9 irradiance of the sky (lux) in a host-visible buffer per frame
 * slot. S36a1b: the SH of a uniform sky whose radiance follows the sun and the moon (skyRadiance: a blue sky of 0.08
 * of the bodies' horizontal illuminance over pi; a1c: the environment map's SH and its prefiltered cube); at least the
 * environment's minAmbient is applied in the shader. setUniformRadiance replaces the sky by a uniform radiance (tests:
 * the white furnace; inspection).
 */
class SAIGA_VULKAN_API SkyAmbient : public AmbientProvider
{
   public:
    void init(VulkanBase& base);
    void destroy();
    ~SkyAmbient() override { destroy(); }

    void update(vk::CommandBuffer cmd, uint32_t frameIndex, const EnvironmentState& environment) override;
    AmbientBindings bindings(uint32_t frameIndex) override;
    AmbientStatistics statistics() const override { return stats; }

    // A uniform sky radiance (cd/m^2) instead of the sky; nullopt: the sky
    void setUniformRadiance(std::optional<vec3> radiance) { uniform = radiance; }
    // The uniform sky radiance of an environment (a1b)
    static vec3 skyRadiance(const EnvironmentState& environment);

    static constexpr size_t kShBytes = 9 * 16;

   private:
    VulkanBase* base = nullptr;
    std::array<Buffer, 2> buffers;
    std::optional<vec3> uniform;
    AmbientStatistics stats;
};

}  // namespace Vulkan
}  // namespace Saiga
