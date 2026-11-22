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
class EnvironmentMap;

/**
 * The sky's ambient light (A3 of the S36 plan): the SH9 irradiance of the sky (lux) and the specular environment. With
 * a valid environment map (S36a1c, setEnvironmentMap) its front SH and its prefiltered cube; before its first capture
 * (and in an enclosed space, whose ambient the shader takes from the block) a host-visible buffer per frame slot with
 * the floor minAmbient as band 0; at least the environment's minAmbient is applied in the shader.
 * setUniformRadiance replaces the sky by a uniform radiance (tests: the white furnace; inspection): its SH, no cube.
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
    bool uniformRadiance() const { return uniform.has_value(); }
    // The environment map whose SH and cube are the sky's (RenderScene's; nullptr: the floor)
    void setEnvironmentMap(EnvironmentMap* map) { environmentMap = map; }
    // The environment map is used this frame (valid, no uniform radiance)
    bool usesEnvironmentMap() const;

    static constexpr size_t kShBytes = 9 * 16;

   private:
    VulkanBase* base = nullptr;
    std::array<Buffer, 2> buffers;
    std::optional<vec3> uniform;
    EnvironmentMap* environmentMap = nullptr;
    AmbientStatistics stats;
};

}  // namespace Vulkan
}  // namespace Saiga
