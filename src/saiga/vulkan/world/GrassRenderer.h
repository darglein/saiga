/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/buffer/UniformBuffer.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/texture/Texture.h"
#include "saiga/vulkan/world/TerrainRenderer.h"
#include "saiga/vulkan/world/WorldCommon.h"

#include <array>

namespace Saiga
{
namespace Vulkan
{
/**
 * Grass clumps around the camera, without stored instances: the vertex shader derives the position, rotation and size
 * of a clump from its cell in a world aligned grid (cell size from the world uniforms) and the height from the
 * height map. The grid around the camera is drawn in tiles x tiles tiles (one instanced draw each), which are culled
 * on the CPU. Clumps on steep slopes, under water or outside the radius get size 0, the density fades out towards
 * the radius.
 */
class SAIGA_VULKAN_API GrassRenderer
{
   public:
    static constexpr int tiles = 8;
    // Upper bound of the cells per tile side (instances per draw: maxTileCells^2)
    static constexpr int maxTileCells = 512;

    void init(VulkanBase& base, const RenderingFormats& formats, UniformBuffer& worldUniforms,
              Texture2DArray& foliageTexture, TerrainRenderer& terrain);
    void destroy();

    // Selects the visible tiles (CPU).
    void update(const vec3& cameraPos, const std::array<vec4, 6>& frustum, float radius, float cellSize);
    void render(vk::CommandBuffer cmd, bool wireframe);

    struct Stats
    {
        int tiles = 0;
        // Cells per tile side
        int tileCells = 0;
        // Clumps of the drawn tiles (including the ones that are discarded in the vertex shader)
        int clumps    = 0;
        int triangles = 0;
    };
    const Stats& stats() const { return statistics; }

    // One draw call per visible tile
    int drawCalls() const { return int(visibleTiles.size()); }
    size_t memoryBytes() const { return vertexBuffer.size() + indexBuffer.size(); }

   private:
    struct Tile
    {
        ivec2 origin;
    };

    VulkanBase* base         = nullptr;
    TerrainRenderer* terrain = nullptr;
    Buffer vertexBuffer, indexBuffer;
    uint32_t indexCount = 0;
    int tileCells       = 1;
    std::vector<Tile> visibleTiles;
    Stats statistics;

    Pipeline pipeline, wirePipeline;
    StaticDescriptorSet set;
};

}  // namespace Vulkan
}  // namespace Saiga
