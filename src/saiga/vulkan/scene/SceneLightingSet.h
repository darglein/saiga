/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/DescriptorSetLayout.h"
#include "saiga/vulkan/scene/AmbientProvider.h"
#include "saiga/vulkan/scene/SceneLighting.h"
#include "saiga/vulkan/texture/Sampler.h"
#include "saiga/vulkan/texture/Texture.h"

#include <array>
#include <optional>

namespace Saiga
{
namespace Vulkan
{
// The resources of set 1 in a frame (SceneLightingSet::write); a missing one is bound to its 1x1 dummy. a1b: the
// foliage texture, the ambient and the BRDF LUT; a1c adds the cloud coverage and the atmosphere's LUTs, a2 the cascades
// and the horizon map, b the lights, clusters and the atlas
struct SceneLightingBindings
{
    std::optional<vk::DescriptorImageInfo> foliage;
    AmbientBindings ambient;
    std::optional<vk::DescriptorImageInfo> cloudCoverage, brdfLut, transmittanceLut, skyViewLut;
};

/**
 * The owner of set 1 of the scene draws (S36, A3 of docs/stages/S36_lighting.plan.md; finding 11 of review 2): the
 * set-1 layout (final from S36a1b, so later parts add no layout change to every pipeline), the lighting block's buffer
 * and one descriptor set per frame slot, the 1x1 dummies of the bindings a part does not fill yet. Bindings (vertex,
 * fragment and compute stages): 0 the block (SceneLighting, uniform), 1 the foliage texture (sampler2DArray), 2 the
 * ambient SH (storage), 3 the cloud coverage (sampler2D), 4 the prefiltered environment cube (samplerCube), 5 the BRDF
 * LUT, 6 the transmittance LUT, 7 the sky-view LUT (sampler2D), 8 the cascade shadow map (sampler2DArrayShadow), 9 the
 * horizon maps (sampler2DArray), 10 the lights, 11 the cluster table, 12 the light indices (storage), 13 the shadow
 * atlas (sampler2DArrayShadow), 14 the atlas tile table (storage). RenderScene writes it once per frame; SceneCulling,
 * TerrainPass and GrassPass create their pipelines with layout() and bind set(frameIndex) as set 1.
 */
class SAIGA_VULKAN_API SceneLightingSet
{
   public:
    static constexpr uint32_t kBindingCount = 15;
    // The storage buffers of the set (per stage, next to the scene set's)
    static constexpr uint32_t kStorageBuffers = 5;

    void init(VulkanBase& base);
    void destroy();
    ~SceneLightingSet() { destroy(); }

    // The layout's description for a pipeline (identically defined: compatible with the sets of set())
    static DescriptorSetLayout layout();
    // The frame slot's block and bindings (its previous frame has completed)
    void write(uint32_t frameIndex, const SceneLighting& block, const SceneLightingBindings& bindings);
    vk::DescriptorSet& set(uint32_t frameIndex) { return frames[frameIndex].set; }
    // The block of the frame slot's last write
    const SceneLighting& block(uint32_t frameIndex) const { return frames[frameIndex].block; }

   private:
    struct Frame
    {
        Buffer buffer;
        vk::DescriptorSet set;
        SceneLighting block;
    };
    VulkanBase* base = nullptr;
    DescriptorSetLayout ownLayout;
    std::array<Frame, 2> frames;
    // 1x1 dummies: a color 2D array (foliage, horizon), a 2D texture, a cube, a depth 2D array with a comparison
    // sampler, a storage buffer
    Texture2DArray dummyArray, dummyDepth;
    Texture2D dummy2D;
    TextureCube dummyCube;
    Sampler compareSampler;
    Buffer dummyStorage;
};

}  // namespace Vulkan
}  // namespace Saiga
