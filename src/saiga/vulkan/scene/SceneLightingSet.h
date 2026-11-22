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
    // S36a2: the cascade shadow map (a depth 2D array with its comparison sampler) and the horizon maps
    std::optional<vk::DescriptorImageInfo> cascades, horizon;
    // S36b: the frame's lights (10), the cluster counts (11) and light indices (12), the shadow atlas (13, a D16 2D
    // array with the comparison sampler) and its tile table (14)
    std::optional<vk::DescriptorBufferInfo> lights, clusterCounts, clusterIndices, atlasTiles;
    std::optional<vk::DescriptorImageInfo> atlas;
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
 *
 * The shadow and atlas pipelines (S36a2, b) use the small shadowLayout() (0 the block, 1 the foliage texture) and
 * shadowSet(frameIndex): a depth image that is the current attachment must not be bound through a set of the same draw.
 */
class SAIGA_VULKAN_API SceneLightingSet
{
   public:
    static constexpr uint32_t kBindingCount = 15;
    // The storage buffers of the set (per stage, next to the scene set's)
    static constexpr uint32_t kStorageBuffers = 5;
    // The pipeline stages that sample the set's images and buffers: the scene's vertex shaders too (the grass and the
    // terrain evaluate light terms per vertex: vertex_light.glsl, the grass's shadows), the fragment and the compute
    // shaders (previews, the environment capture). The barriers around a write of a bound resource use them
    static constexpr vk::PipelineStageFlags2 kReaderStages = vk::PipelineStageFlagBits2::eVertexShader |
                                                             vk::PipelineStageFlagBits2::eFragmentShader |
                                                             vk::PipelineStageFlagBits2::eComputeShader;

    void init(VulkanBase& base);
    void destroy();
    ~SceneLightingSet() { destroy(); }

    // The layout's description for a pipeline (identically defined: compatible with the sets of set())
    static DescriptorSetLayout layout();
    // The shadow pipelines' set 1 (S36a2): the block and the foliage texture only
    static DescriptorSetLayout shadowLayout();
    vk::DescriptorSet& shadowSet(uint32_t frameIndex) { return frames[frameIndex].shadowSet; }
    // The comparison sampler of the cascades (linear where D32 supports linear filtering) and whether it is linear
    Sampler& shadowSampler() { return compareSampler; }
    // The atlas's comparison sampler (S36b; linear where D16 supports linear filtering) and whether it is linear
    Sampler& atlasSampler() { return atlasCompareSampler; }
    bool linearAtlasFilter() const { return linearAtlasCompare; }
    bool linearShadowFilter() const { return linearCompare; }
    // The frame slot's block and bindings (its previous frame has completed)
    void write(uint32_t frameIndex, const SceneLighting& block, const SceneLightingBindings& bindings);
    vk::DescriptorSet& set(uint32_t frameIndex) { return frames[frameIndex].set; }
    // The block of the frame slot's last write
    const SceneLighting& block(uint32_t frameIndex) const { return frames[frameIndex].block; }

   private:
    struct Frame
    {
        Buffer buffer;
        vk::DescriptorSet set, shadowSet;
        SceneLighting block;
    };
    VulkanBase* base = nullptr;
    DescriptorSetLayout ownLayout, ownShadowLayout;
    bool linearCompare = false;
    std::array<Frame, 2> frames;
    // 1x1 dummies: a color 2D array (foliage, horizon), a 2D texture, a cube, a depth 2D array with a comparison
    // sampler (D32: the cascades; D16: the atlas, S36b), a storage buffer
    Texture2DArray dummyArray, dummyDepth, dummyDepth16;
    Texture2D dummy2D;
    TextureCube dummyCube;
    Sampler compareSampler, atlasCompareSampler;
    bool linearAtlasCompare = false;
    Buffer dummyStorage;
};

}  // namespace Vulkan
}  // namespace Saiga
