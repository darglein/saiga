/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/buffer/UniformBuffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/texture/Texture.h"
#include "saiga/vulkan/world/FoliageGenerator.h"
#include "saiga/vulkan/world/HeightField.h"
#include "saiga/vulkan/world/WorldCommon.h"

#include <array>

namespace Saiga
{
namespace Vulkan
{
/**
 * Up to a few million instanced low-poly trees, culled and LOD-selected on the GPU:
 *
 * - All meshes (species x levels of detail) are in one vertex and one index buffer. The opaque core and the alpha
 *   tested leaf cards of a mesh are drawn together (no back face culling, alpha to coverage).
 * - The instances (position, rotation, scale, species) are in a storage buffer.
 * - Every frame a compute shader tests each instance against the frustum, selects the level of detail by distance
 *   and appends the instance to the visible list of its bucket (species, level), counting with atomics in the
 *   instanceCount of the bucket's indirect draw command.
 * - One vkCmdDrawIndexedIndirect with a command per bucket draws all trees.
 */
class SAIGA_VULKAN_API ForestRenderer
{
   public:
    static constexpr int lodCount = 3;

    struct TreeInstance
    {
        // xyz: position of the stem base, w: rotation around y (radians)
        vec4 posYaw;
        // x: height (scale of the unit mesh), y: species, zw: unused
        vec4 scaleSpecies;
    };

    // Places count trees on the terrain (deterministic for the seed).
    static std::vector<TreeInstance> placeTrees(const HeightField& heightField, int count, uint32_t seed,
                                                const std::vector<TreeSpecies>& species);

    void init(VulkanBase& base, const RenderingFormats& formats, UniformBuffer& worldUniforms,
              Texture2DArray& foliageTexture, const HeightField& heightField, int treeCount, uint32_t seed);
    void destroy();

    // Records the culling (outside of the rendering pass). The world uniforms must be visible to the compute shader.
    void cull(vk::CommandBuffer cmd);
    void render(vk::CommandBuffer cmd, bool wireframe);

    int treeCount() const { return int(instances.size()); }
    const std::vector<TreeInstance>& treeInstances() const { return instances; }
    const std::vector<TreeSpecies>& treeSpecies() const { return species; }

    struct Stats
    {
        // Visible trees per level of detail (of a frame that completed recently)
        std::array<int, lodCount> visible = {};
        int triangles                     = 0;
        // Visible trees per bucket (species * lodCount + lod)
        std::vector<int> bucketVisible;
    };
    const Stats& stats() const { return statistics; }

    // Triangles of the mesh of each bucket (species * lodCount + lod)
    const std::vector<int>& bucketTriangleCounts() const { return bucketTriangles; }
    // Indirect draw commands per frame (one vkCmdDrawIndexedIndirect with all of them)
    int drawCommands() const { return int(commandTemplate.size()); }
    // Draw calls render() issues: one multi-draw-indirect, or one indirect draw per bucket without multiDrawIndirect
    int drawCalls() const;
    // GPU memory (meshes, instances, visible list, commands)
    size_t memoryBytes() const;

   private:
    VulkanBase* base = nullptr;
    std::vector<TreeSpecies> species;
    std::vector<TreeInstance> instances;

    Buffer vertexBuffer, indexBuffer;
    Buffer instanceBuffer, visibleBuffer, commandBuffer;
    std::vector<vk::DrawIndexedIndirectCommand> commandTemplate;
    std::vector<int> bucketTriangles;

    // Readback of the commands (instance counts) for the statistics, one per frame in flight
    struct Readback
    {
        Buffer buffer;
        uint64_t frameValue = 0;
    };
    std::array<Readback, 3> readbacks;
    uint32_t readbackIndex = 0;
    Stats statistics;

    ComputePipeline cullPipeline;
    StaticDescriptorSet cullSet;
    Pipeline pipeline, wirePipeline;
    StaticDescriptorSet drawSet;

    void readStats();
};

}  // namespace Vulkan
}  // namespace Saiga
