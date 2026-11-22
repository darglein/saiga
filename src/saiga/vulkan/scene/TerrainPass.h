/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <array>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
class TerrainAtlas;
class SceneLightingSet;
struct SceneCamera;
struct PrototypeSettings;
struct Texture2DArray;

// TerrainFrame (std140, shader/vulkan/scene/terrain_common.glsl): the per-frame uniforms of the terrain, water, grass
// and sky passes, relative to the camera or the frame anchor
struct TerrainFrameUniforms
{
    mat4 viewProjRel    = mat4::Identity();
    mat4 invViewProjRel = mat4::Identity();
    // xyz: cameraRel = float(camera - (anchor.x, refHeight * kHeightUnit, anchor.z)), computed in double
    vec4 cameraRel = vec4(0, 0, 0, 0);
    // x, y: the anchor's x and z (m), z: refHeight (height units), w: the sea level (height units)
    ivec4 anchor = ivec4(0, 0, 0, 0);
    // x: samples per tile side, y: streamed levels, z: pyramid samples per side, w: base spacing (m)
    ivec4 grid = ivec4(0, 0, 0, 0);
    // x: a sea level exists, y: the root level, z: debug mode, w: unused
    ivec4 flags = ivec4(0, 0, 0, 0);
    // Per streamed level: x the page-table offset, y the tiles per side
    std::array<ivec4, 8> levels;
    vec4 sunDir   = vec4(0, 1, 0, 1);
    vec4 shading  = vec4(0, 0, 0, 0);
    vec4 fog      = vec4(0, 0, 0, 0);  // w: the camera height above the fog base (m)
    vec4 fogColor = vec4(0, 0, 0, 1);
    vec4 skyColor = vec4(0, 0, 0, 1);
    // x: the range of level 0 used (m), y: morph start, z: the LOD metric's height (the camera height above the
    // surface under it, m)
    vec4 lod   = vec4(0, 0, 0, 0);
    vec4 grass = vec4(0, 0, 0, 0);
    // x, y: the grass wind phases, z: wind, w: alpha cutoff
    vec4 grassPhases = vec4(0, 0, 0, 0);
    vec4 viewport    = vec4(0, 0, 0, 0);
};

// A selected node (TerrainPass::selectedNodes, tests): in instance order
struct TerrainNodeInfo
{
    int level = 0;
    // The node's origin in the space (integer metres)
    int32_t originX = 0, originZ = 0;
    // The global sample index of the origin at the node's level
    int32_t sampleX = 0, sampleZ = 0;
    bool quarter = false, water = false;
};

// A probe of TerrainPass::probeVertices: the node's instance index and a grid sample (0..32, 0..16 for a quarter);
// normal: the vertex normal and the level out instead of the position and the morph factor
struct TerrainProbe
{
    int node = 0, gridX = 0, gridZ = 0;
    bool normal = false;
};

/**
 * The CDLOD terrain of RenderScene over the GPU terrain set (S20, docs/engine_design.md §10.1, §12.2), replacing the
 * prototype's TerrainRenderer:
 * - nodes of level k have vertex spacing baseSpacing * 2^k and size 32 * baseSpacing * 2^k (gridN 32; quarter grids of
 *   16 at the parent's spacing); level 0 has one vertex per sample; levels below streamedLevels read their own tile
 *   level, coarser levels the pyramid. The roots are the 4096 m nodes, always considered.
 * - Selection in double: node boxes from the atlas's hierarchical bounds, relative to the camera, culled with the box
 *   reaching down by the skirt; the LOD distance is relief independent, sqrt(dxz^2 + hc^2) with the box's horizontal
 *   distance dxz and one height per frame hc (the camera height above the surface under it, TerrainFrame lod.z), the
 *   same metric as the vertex morph, so the CDLOD invariant (no morph on the coarse side of a level border, a fully
 *   morphed fine side) holds for any relief with ranges terrainLodRange * 2^k above the morph minimum; a node is
 *   refined only if its children's tile is GPU-resident; a child out of range is drawn as a quarter at the parent's
 *   spacing.
 * - The frame anchor: the camera's xz rounded down to 64 m and refHeight (the camera height rounded down to 64 m in
 *   height units, clamped to the limit); node origins are integers relative to the anchor, cameraRel is rounded once.
 * - Skirts (the outer ring of each grid, max(node range, 2 spacings) deep) close the T-junctions of residency-driven
 *   refinement; the water plane (400 km, centred on the camera snapped to 4096 m, at the sea level) is a node with the
 *   water flag, culled with its own box.
 * The per-frame uniform buffer TerrainFrame is written every frame the pass draws (also read by grass and sky).
 */
class SAIGA_VULKAN_API TerrainPass
{
   public:
    static constexpr int gridN = 32;

    enum class Mode
    {
        Scene,
        Depth,
        Equal
    };

    struct Stats
    {
        // Per level: the selected nodes (full and quarter), the quarters
        std::array<int, 16> nodesPerLevel{}, quartersPerLevel{};
        int nodes = 0, quarters = 0;
        // Refinements refused because the children's tile was not GPU-resident
        int refusedResidency = 0;
        bool water           = false;
        // Nodes beyond TerrainAtlas::maxNodes (clipped)
        int clipped   = 0;
        int triangles = 0;
    };

    // lighting: set 1 of the terrain, water and sky pipelines (S36; RenderScene's, outlives this); pbr: the pipeline
    // variant (LightingParameters::pbr)
    void init(VulkanBase& base, const RenderingFormats& formats, SceneLightingSet& lighting, bool pbr);
    void destroy();
    ~TerrainPass() { destroy(); }
    // The pipelines of the light model (S36a1b): recreated on a change (the old ones released deferred)
    void setPbr(bool pbr);

    // The transfer section: the node selection, the frame's node instances and TerrainFrame, the descriptor set.
    // Draws nothing (and returns false) without a GPU-resident pyramid of the camera's space
    bool transfer(vk::CommandBuffer cmd, uint32_t frameIndex, TerrainAtlas& atlas, const SceneCamera& camera,
                  const PrototypeSettings& settings, double time, int width, int height);
    // Nothing this frame (no set, no camera)
    void skipFrame();
    bool drawsThisFrame() const { return frameValid; }
    // Inside a pass: the terrain (settings.terrain) and the water (settings.water)
    void render(vk::CommandBuffer cmd, uint32_t frameIndex, Mode mode, const PrototypeSettings& settings);
    // The sky (settings.sky) of the environment, last in the scene pass (EQUAL at depth 0); TerrainFrame is written
    // for every frame with a camera, also without a terrain set
    void renderSky(vk::CommandBuffer cmd, uint32_t frameIndex, const PrototypeSettings& settings);

    const Stats& stats() const { return statistics; }
    // The last frame's nodes in instance order (tests)
    const std::vector<TerrainNodeInfo>& selectedNodes() const { return nodeInfos; }
    // Runs terrainVertex for the probes with the last frame's node data (waits for the device); camera-relative
    // positions
    std::vector<vec4> probeVertices(const std::vector<TerrainProbe>& probes);

    // The frame's uniforms (the grass and the sky bind the set of the frame slot)
    const TerrainFrameUniforms& frameUniforms() const { return uniforms; }
    vk::DescriptorSet descriptorSet(uint32_t frameIndex) const { return frames[frameIndex].set; }
    Buffer& uniformBuffer(uint32_t frameIndex) { return frames[frameIndex].uniforms; }
    // The descriptor set layout of the terrain passes (set 0): grass pipelines are created with it (S36: the foliage
    // texture moved to binding 1 of the lighting set)
    static DescriptorSetLayout setLayout();
    // The range of level 0 actually used (the morph minimum)
    double usedRange0() const { return range0; }
    // The LOD metric's height of the last frame (m): distances are sqrt(dxz^2 + lodHeight^2)
    double usedLodHeight() const { return lodHeight; }
    // The anchor of the last frame: x, z (m) and refHeight (height units)
    ivec4 anchor() const { return uniforms.anchor; }

    // The GPU layout of a node instance (32 bytes)
    struct GpuNode
    {
        int32_t originX = 0, originZ = 0;
        int32_t sampleX = 0, sampleZ = 0;
        uint32_t levelFlags = 0;
        float skirtDepth    = 0;
        uint32_t pad0 = 0, pad1 = 0;
    };

   private:
    struct Frame
    {
        Buffer uniforms;
        vk::DescriptorSet set;
    };
    VulkanBase* base              = nullptr;
    SceneLightingSet* lightingSet = nullptr;
    RenderingFormats sceneFormats;
    bool pbrVariant = false;
    Pipeline scenePipeline, depthPipeline, equalPipeline, skyPipeline;
    // The layouts the frame and sky sets are allocated from (independent of the pipelines, which setPbr recreates)
    DescriptorSetLayout frameLayout, skyLayout;
    void createPipelines();
    // The sky's sets (binding 0: the frame's uniform buffer), written once
    std::array<vk::DescriptorSet, 2> skySets;
    bool skyValid = false;
    ComputePipeline probePipeline;
    vk::DescriptorSet probeSet;
    TerrainAtlas* lastAtlas = nullptr;
    // The atlas set of the last frame (TerrainAtlas::Stats::rebuilds): probes need the frame's set
    uint64_t lastSetRebuilds = 0;
    int samples              = 1;
    uint32_t numFull = 0, numQuarter = 0, numWater = 0;
    TerrainNodeInfo waterInfo;
    // Index buffers in the (n + 3)^2 vertex layout: full and quarter grids with and without the skirt ring
    Buffer indices;
    struct Range
    {
        uint32_t first = 0, count = 0;
    };
    Range fullSkirt, fullPlain, quarterSkirt, quarterPlain;
    std::array<Frame, 2> frames;
    TerrainFrameUniforms uniforms;
    bool initialized   = false;
    bool frameValid    = false;
    uint32_t lastFrame = 0;
    double range0      = 300;
    // The LOD metric's height (TerrainFrame lod.z): the camera height above the surface under it
    double lodHeight = 0;
    // The frame's nodes: full, quarter, water (their instance ranges)
    std::vector<GpuNode> fullNodes, quarterNodes, waterNodes;
    std::vector<TerrainNodeInfo> fullInfos, quarterInfos, nodeInfos;
    Stats statistics;

    void select(const TerrainAtlas& atlas, const SceneCamera& camera, int32_t anchorX, int32_t anchorZ);
    bool selectNode(const TerrainAtlas& atlas, const SceneCamera& camera, int level, int32_t nx, int32_t nz,
                    int32_t anchorX, int32_t anchorZ, const std::vector<double>& ranges);
    void addNode(const TerrainAtlas& atlas, int level, int32_t ox, int32_t oz, bool quarter, int32_t anchorX,
                 int32_t anchorZ, double skirt);
    void writeDescriptors(Frame& f, TerrainAtlas& atlas, uint32_t frameIndex);
};

}  // namespace Vulkan
}  // namespace Saiga
