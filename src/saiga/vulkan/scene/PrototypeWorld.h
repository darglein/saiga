/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/buffer/UniformBuffer.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/scene/GrassRenderer.h"
#include "saiga/vulkan/scene/TerrainRenderer.h"
#include "saiga/vulkan/texture/Texture.h"
#include "saiga/vulkan/world/HeightField.h"
#include "saiga/vulkan/world/WorldCommon.h"

#include <memory>

namespace Saiga
{
namespace Vulkan
{
struct SceneCamera;
struct SceneLighting;

/**
 * The settings of the prototype content of RenderScene (S19a, docs/engine_design.md §10.1; transitional until S20):
 * the game's WorldSettings map to it (R-MECH-1: the toggles, distances and light stay game settings). The tree LOD
 * distances and the wind's bounds padding are mesh data (RenderScene::setMeshLodDistances, PrototypeForest).
 */
struct PrototypeSettings
{
    // The passes (racing turns the grass off); terrain = false keeps the water plane
    bool terrain = true, grass = true, sky = true, water = true;
    // The depth pre-pass of the opaque content (terrain, water, opaque instances; §10.5 pass 2). Also read without
    // setPrototypeTerrain (the opaque instances). Off by default: the prototype's terrain already draws first, so the
    // pre-pass only removes terrain self-overdraw and costs a second CDLOD vertex pass (measured slower in S19a); the
    // mechanism stays for S37
    bool depthPrePass = false;
    bool fog          = true;
    float fogDensity  = 0.00015f;
    float fogFalloff  = 0.0012f;
    // Degrees
    float sunAzimuth   = 35.0f;
    float sunElevation = 38.0f;
    float sunIntensity = 1.0f;
    float ambient      = 0.25f;
    float albedo       = 0.55f;
    // The wind strength of the trees' and the grass's shaders (the trees' bounds padding is mesh data)
    float wind = 1.0f;
    // CDLOD: the range of level 0 (m) and the morph start as a fraction of a level's range
    float terrainLodRange = 300.0f;
    float morphStart      = 0.7f;
    float grassRadius     = 60.0f;
    float grassCellSize   = 0.3f;
    float grassDensity    = 0.85f;
    // The foliage's alpha cutoff (trees and grass)
    float alphaCutoff = 0.45f;
    // 0: off, 1: tint by LOD (terrain levels, the trees' LOD bits)
    int debugMode    = 0;
    bool flatShading = true;
    // The skirts of the S20 terrain nodes (TerrainPass); off only for debugging and the crack test
    bool terrainSkirts = true;
};

// The lighting block of the instance draws from the prototype settings (sun, hemisphere ambient, fog with its base
// height, wind phases of the time, cutoff, tint): PrototypeWorld (S19a) and RenderScene's environment (S20)
SAIGA_VULKAN_API void prototypeLighting(const PrototypeSettings& settings, double time, float fogBase,
                                        SceneLighting& out);

// The color word of a PrototypeFoliage vertex (S19a): the foliage texture layer in bits 0-7 (a negative layer: opaque
// parts, 0xFF), the AO in bits 8-15 and the wind weight in bits 16-23 (unorm8; both asserted in [0, 1])
SAIGA_VULKAN_API uint32_t packPrototypeFoliageColor(int layer, float ao, float wind);

/**
 * The prototype world content drawn by RenderScene (S19a, until S20 replaces it): the CDLOD terrain with the water
 * plane (a mode of the terrain draw in the opaque MSAA pass), the grass around the camera, the sky gradient, the
 * in-shader height fog, and the foliage texture of the PrototypeFoliage pipeline class. The uniforms (WorldUniforms of
 * WorldCommon.h) are written in the transfer section from the frame's camera (camera-relative matrices, the float
 * camera position), the settings and the time's wind phases.
 */
class SAIGA_VULKAN_API PrototypeWorld
{
   public:
    ~PrototypeWorld() { destroy(); }

    // formats: the scene pass (RenderScene::formats()); the height field is shared; foliageSeed: the foliage texture
    // (WorldParameters::seed)
    void init(VulkanBase& base, const RenderingFormats& formats, std::shared_ptr<const HeightField> heights,
              uint32_t foliageSeed);
    void destroy();

    // The transfer section: the terrain node selection (records the instance upload), the grass tiles and the uniforms
    void transfer(vk::CommandBuffer cmd, const SceneCamera& camera, const PrototypeSettings& settings, double time,
                  int width, int height);
    // Inside the depth pre-pass: the terrain and the water plane (depth only)
    void renderDepth(vk::CommandBuffer cmd, const PrototypeSettings& settings);
    // Inside the scene pass: the terrain and the water plane (EQUAL after the pre-pass), the grass, the sky
    void renderTerrain(vk::CommandBuffer cmd, const PrototypeSettings& settings, bool afterPrepass);
    void renderGrass(vk::CommandBuffer cmd, const PrototypeSettings& settings);
    void renderSky(vk::CommandBuffer cmd, const PrototypeSettings& settings);

    // The lighting block of the instance draws from the settings (sun, hemisphere ambient, fog, wind, cutoff, tint)
    void lighting(const PrototypeSettings& settings, double time, SceneLighting& out) const;

    Texture2DArray& foliageTexture() { return foliage; }
    const HeightField& heightField() const { return *hf; }

    struct Stats
    {
        // terrainTriangles: the selected nodes with the water plane (0 with the terrain off), as the prototype
        int terrainNodes = 0, quarterNodes = 0, waterNodes = 0, terrainTriangles = 0, waterTriangles = 0;
        int terrainMaxLevel               = 0;
        std::array<int, 16> nodesPerLevel = {}, trianglesPerLevel = {};
        int grassTiles = 0, grassClumps = 0, grassTriangles = 0;
        int drawCalls = 0;
        // CPU time of the terrain node selection and the grass tile culling (ms)
        double cpuTerrainSelectMs = 0, cpuGrassCullMs = 0;
    };
    // The statistics of the last transfer
    const Stats& stats() const { return statistics; }
    // GPU memory (height map, grids, nodes, grass, foliage texture, uniforms)
    size_t memoryBytes() const;

    // The wind phases fmod(time * f, 2 pi) in double (§10.1: no absolute time in a shader)
    static float windPhase(double time, double frequency);

   private:
    VulkanBase* base = nullptr;
    std::shared_ptr<const HeightField> hf;
    int samples = 1;
    UniformBuffer uniformBuffer;
    WorldUniforms uniforms;
    Texture2DArray foliage;
    TerrainRenderer terrain;
    GrassRenderer grass;
    Pipeline skyPipeline;
    StaticDescriptorSet skySet;
    Stats statistics;
    // The last transfer had a camera (its node lists and uniforms are this frame's)
    bool frameValid  = false;
    bool initialized = false;
};

}  // namespace Vulkan
}  // namespace Saiga
