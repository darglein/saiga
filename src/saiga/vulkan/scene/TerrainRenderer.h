/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/buffer/IndexBuffer.h"
#include "saiga/vulkan/buffer/UniformBuffer.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/texture/Texture.h"
#include "saiga/vulkan/world/HeightField.h"
#include "saiga/vulkan/world/WorldCommon.h"

#include <array>

namespace Saiga
{
namespace Vulkan
{
/**
 * Terrain with continuous distance dependent level of detail (CDLOD, Strugar 2010):
 *
 * - The height map is a R32Sfloat texture, the shaders sample it like HeightField::height().
 * - A quadtree over the world, leaf nodes of leafSize meters. Level l has nodes of leafSize * 2^l meters, each drawn
 *   as one instance of a grid of gridResolution x gridResolution cells, so the vertex spacing doubles per level.
 * - The CPU selects the nodes each frame by distance (level l is used up to range(l) = range0 * 2^l) and frustum.
 *   Where a node is only partly refined, the remaining quarters are drawn with a grid of half the resolution at the
 *   parent's spacing.
 * - In the vertex shader the vertices morph into the grid of the next level before the level changes, so there are
 *   no cracks or pops.
 *
 * The water plane is drawn with the same pipeline (an instance with level -1).
 *
 * S19a: drawn by RenderScene (PrototypeWorld) until S20 replaces it; the depth pre-pass draws it with terrain.vert
 * alone (Mode::Depth, invariant gl_Position) and the scene pass then with EQUAL and no depth write (Mode::Equal).
 */
class SAIGA_VULKAN_API TerrainRenderer
{
   public:
    static constexpr int gridResolution = 32;
    // Max number of selected nodes per frame (the instances are uploaded with vkCmdUpdateBuffer: <= 64 KB)
    static constexpr int maxInstances = 4000;

    // The pipeline of a draw: the scene pass (depth test and write), the depth pre-pass (depth only), the scene pass
    // after the pre-pass (EQUAL, no depth write)
    enum class Mode
    {
        Scene,
        Depth,
        Equal
    };

    // leafSize: size of the finest nodes in meters. gridResolution cells of a leaf should match the height map
    // spacing. formats: the scene pass (its depth format and sample count also make the pre-pass pipeline)
    void init(VulkanBase& base, const RenderingFormats& formats, UniformBuffer& worldUniforms,
              const HeightField& heightField, float leafSize = 64.0f);
    void destroy();

    /**
     * Selects the nodes for the camera position and frustum, and records the upload of the instances (outside of the
     * rendering pass). range0: the range of level 0 in meters, it is enlarged if it is too small for the node size and
     * the morph area (morphStart: the start of the morph as a fraction of a level's range, see terrain.vert).
     */
    void update(vk::CommandBuffer cmd, const vec3& cameraPos, const std::array<vec4, 6>& frustum, float range0,
                float morphStart, bool water);

    // terrain = false draws only the water nodes (the water plane without the terrain)
    void render(vk::CommandBuffer cmd, Mode mode, bool terrain = true);

    // The range of level 0 actually used (see update)
    float range0() const { return usedRange0; }
    int maxLevel() const { return levels - 1; }
    // Size of the nodes of a level in meters (the vertex spacing is nodeSize / gridResolution)
    float nodeSize(int level) const { return leafSize * float(1 << level); }

    // Draw calls of render(cmd, mode, terrain)
    int drawCalls(bool terrain = true) const
    {
        if (!terrain) return int(!waterNodes.empty());
        return int(!fullNodes.empty()) + int(!quarterNodes.empty() || !waterNodes.empty());
    }
    // GPU memory (height map, meshes, instances)
    size_t memoryBytes() const;

    // Min and max terrain height in the xz-rectangle (conservative, from the leaf nodes).
    vec2 heightRange(float x0, float z0, float x1, float z1) const;

    Texture2D& heightTexture() { return heightMap; }

    struct Stats
    {
        int nodes             = 0;
        int quarterNodes      = 0;
        int waterNodes        = 0;
        int triangles         = 0;
        int nodesPerLevel[16] = {};
        // Triangles per level (full nodes: gridResolution^2 * 2, quarter nodes: a quarter of it)
        int trianglesPerLevel[16] = {};
    };
    const Stats& stats() const { return statistics; }

   private:
    struct NodeInstance
    {
        // origin x, origin z, size, level (-1: water)
        vec4 data;
    };

    VulkanBase* base      = nullptr;
    const HeightField* hf = nullptr;
    float leafSize        = 64;
    float worldSize       = 0;
    int levels            = 1;
    int leavesPerSide     = 1;
    float usedRange0      = 300;

    // Min/max height of the nodes of each level, level 0 = leaves (row major, leavesPerSide >> level per side)
    std::vector<std::vector<vec2>> minMax;

    Texture2D heightMap;
    Buffer gridVertices;
    IndexBuffer<uint16_t> fullIndices, quarterIndices;
    uint32_t quarterVertexOffset = 0;
    Buffer instanceBuffer;

    Pipeline pipeline, depthPipeline, equalPipeline;
    StaticDescriptorSet descriptorSet;

    std::vector<NodeInstance> fullNodes, quarterNodes, waterNodes;
    Stats statistics;

    // Recursive selection, returns false if the node is out of its range (the parent draws the area).
    bool select(int level, int x, int z, const vec3& cameraPos, const std::array<vec4, 6>& frustum,
                const std::vector<float>& ranges);
    void nodeBounds(int level, int x, int z, vec3& bmin, vec3& bmax) const;
};

}  // namespace Vulkan
}  // namespace Saiga
