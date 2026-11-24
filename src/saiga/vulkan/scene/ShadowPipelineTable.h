/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <array>
#include <optional>
#include <string>

namespace Saiga
{
namespace Vulkan
{
/**
 * The shadow pipelines of the instance pipeline classes (S36a2, A5 of docs/stages/S36_lighting.plan.md): one entry per
 * class (MeshPool::pipelineClassIndex), depth only (D32, LESS_OR_EQUAL, not reversed), the rasterizer's slope bias as
 * dynamic state, back faces culled (cull none for the two-sided classes). The shaders place the instance as
 * instance.vert does and pancake casters nearer than the cascade's near plane onto depth 0 (gl_Position.z = max(z, 0):
 * no depthClamp feature). The default entries: opaque and two-sided shadow_depth.vert alone; the alpha-masked classes
 * shadow_depth.vert + shadow_alpha.frag (the material's alpha test); PrototypeFoliage shadow_foliage.vert/.frag (the
 * wind of the frame's phases, the foliage texture's alpha cutoff). A class without an entry casts nothing: S27's
 * impostor class and S34's skinned class register theirs with setEntry. Set 0 is SceneCulling's, set 1 the lighting
 * set's shadowLayout() (the block and the foliage texture).
 */
class SAIGA_VULKAN_API ShadowPipelineTable
{
   public:
    struct Entry
    {
        std::string vertexShader;
        std::string fragmentShader;  // empty: depth only
        bool twoSided = false;
    };
    static constexpr vk::Format kDepthFormat = vk::Format::eD32Sfloat;
    // The default entry of a class index (0..pipelineClassCount - 1)
    static Entry defaultEntry(int klass);

    // Creates the default entries
    void init(VulkanBase& base);
    void destroy();
    ~ShadowPipelineTable() { destroy(); }

    // Replaces (or with nullopt removes) a class's entry; outside of a frame's recording (the old pipeline is released
    // after the frames in flight)
    void setEntry(int klass, std::optional<Entry> entry);
    bool hasEntry(int klass) const { return entries[size_t(klass)].has_value(); }
    // Binds the class's pipeline with the sets, the push constants (the cascade's camera-relative view-projection, the
    // interpolation) and the slope bias; false without an entry
    bool bind(vk::CommandBuffer cmd, int klass, vk::DescriptorSet sceneSet, vk::DescriptorSet shadowSet,
              const mat4& viewProj, float alpha, float slopeBias);
    uint64_t pipelineCreations() const { return creations; }

   private:
    VulkanBase* base = nullptr;
    std::array<std::optional<Entry>, pipelineClassCount> entries;
    std::array<std::unique_ptr<Pipeline>, pipelineClassCount> pipelines;
    uint64_t creations = 0;
    void create(int klass);
};

}  // namespace Vulkan
}  // namespace Saiga
