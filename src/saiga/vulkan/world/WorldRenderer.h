/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/world/ForestRenderer.h"
#include "saiga/vulkan/world/GrassRenderer.h"
#include "saiga/vulkan/world/HeightField.h"
#include "saiga/vulkan/world/TerrainRenderer.h"
#include "saiga/vulkan/world/WorldCommon.h"

#include <chrono>
#include <functional>

namespace Saiga
{
namespace Vulkan
{
struct SAIGA_VULKAN_API WorldParameters
{
    HeightField::Parameters terrain;
    int treeCount = 200000;
    uint32_t seed = 1;
};

// The shading mode of the world. Clay: uniform gray with simple lighting, to judge the geometry (also the debug mode of
// the engine's renderer later, see docs/engine_design.md §10.5).
enum class WorldShading
{
    Clay
};

/**
 * The settings that can be changed at runtime (GUI and MCP, see toJson / fromJson).
 */
struct SAIGA_VULKAN_API WorldSettings
{
    WorldShading shading = WorldShading::Clay;
    // Keeps the animation time at 0 (golden images, benchmarks); otherwise it advances with the wall clock
    bool freezeTime = false;

    bool terrain = true, trees = true, grass = true, water = true, sky = true;
    bool wireframe = false;
    // 0: off, 1: tint by level of detail
    int debugMode      = 0;
    bool flatShading   = true;
    bool freezeCulling = false;

    float fovy = 60.0f;

    bool fog           = true;
    float fogDensity   = 0.00015f;
    float fogFalloff   = 0.0012f;
    float sunAzimuth   = 35.0f;
    float sunElevation = 38.0f;
    float sunIntensity = 1.0f;
    float ambient      = 0.25f;
    float albedo       = 0.55f;
    float wind         = 1.0f;

    float terrainLodRange = 300.0f;
    float morphStart      = 0.7f;

    float treeLod1        = 70.0f;
    float treeLod2        = 300.0f;
    float treeMaxDistance = 30000.0f;

    float grassRadius   = 60.0f;
    float grassCellSize = 0.3f;
    float grassDensity  = 0.85f;
    float alphaCutoff   = 0.45f;

    json11::Json toJson() const;
    // Applies the members present in the json object and clamps all values (clamp()). Returns the names of unknown
    // keys and of keys with an invalid value (wrong JSON type, unknown shading mode); those are not applied.
    std::vector<std::string> fromJson(const json11::Json& json);
    // Limits all values to the ranges of the GUI.
    void clamp();
};

/**
 * A large procedural world for the forward renderer (requires reversed depth, ForwardRenderingParameters::reversedZ):
 * CDLOD terrain with a water plane, GPU culled instanced trees with alpha tested leaves, grass around the camera, a
 * sky gradient and height fog. Everything is a uniform gray ("clay") with simple lighting, to judge the geometry.
 *
 *   world.init(renderer, parameters);
 *   // VulkanForwardRenderingInterface::transfer:
 *   world.transfer(cmd, camera.view, camera.getPosition(), aspect);
 *   // VulkanForwardRenderingInterface::render:
 *   world.render(cmd);
 */
class SAIGA_VULKAN_API WorldRenderer
{
   public:
    WorldRenderer() = default;
    ~WorldRenderer() { destroy(); }

    void init(VulkanForwardRenderer& renderer, const WorldParameters& parameters);
    void destroy();

    // Records the uploads and the GPU culling (outside of the rendering pass).
    void transfer(vk::CommandBuffer cmd, const mat4& view, const vec3& cameraPosition, float aspect);
    // Records the draws (inside of the main rendering pass).
    void render(vk::CommandBuffer cmd);

    // The settings window
    void imgui();
    // The detailed statistics window (GPU time per pass, CPU times, geometry, levels of detail, memory)
    void imguiStatistics();
    bool showStatistics = true;

    WorldSettings settings;

    const HeightField& heightField() const { return hf; }
    const WorldParameters& parameters() const { return params; }
    TerrainRenderer& terrain() { return terrainRenderer; }
    ForestRenderer& forest() { return forestRenderer; }
    GrassRenderer& grass() { return grassRenderer; }

    mat4 projection(float aspect) const;

    // The GPU passes, measured with timestamps between them
    enum GpuPass
    {
        PASS_UPLOAD_CULL = 0,
        PASS_TERRAIN,
        PASS_TREES,
        PASS_GRASS,
        PASS_SKY,
        PASS_WIREFRAME,
        PASS_COUNT
    };
    static const char* gpuPassName(int pass);

    struct Stats
    {
        // Averages over the last frames
        double cpuFrameMs = 0;
        double fps        = 0;
        // GPU time of the world passes (culling + drawing), -1 if not available
        double gpuMs = -1;
        // GPU time per pass (averaged)
        std::array<double, PASS_COUNT> gpuPassMs = {};
        // CPU time of the world per frame (averaged): terrain node selection, grass tile culling, all of transfer()
        // (including the selection and culling) and render()
        double cpuTerrainSelectMs = 0, cpuGrassCullMs = 0, cpuTransferMs = 0, cpuRenderMs = 0;
        int drawCalls    = 0;
        int terrainNodes = 0, terrainTriangles = 0;
        std::array<int, 3> treesVisible = {};
        int treeTriangles               = 0;
        int grassClumps = 0, grassTriangles = 0;
        int frames = 0;
    };
    const Stats& stats() const { return statistics; }
    json11::Json statsJson() const;

    struct MemoryEntry
    {
        const char* name;
        size_t bytes;
    };
    // GPU memory of the world
    std::vector<MemoryEntry> memoryUsage() const;

    // Number of the frame that transfer() prepared last (counts from 1)
    int frameNumber() const { return statistics.frames; }

    struct FrameSample
    {
        // CPU time of frame cpuFrame (from its transfer() to the next one)
        double cpuMs = 0;
        int cpuFrame = 0;
        // GPU time of the world passes of frame gpuFrame (ms), -1 if no frame completed since the last sample. The
        // GPU times are read back a few frames later, so gpuFrame < cpuFrame.
        double gpuMs = -1;
        int gpuFrame = 0;
    };
    /**
     * Called once per frame in transfer() with the newest samples. Return true to be removed. Used e.g. for
     * measurements over several frames (compare the frame numbers with frameNumber() at the start to skip samples of
     * frames prepared before).
     */
    std::vector<std::function<bool(const FrameSample&)>> frameCallbacks;

    double time = 0;

   private:
    VulkanForwardRenderer* renderer = nullptr;
    VulkanBase* base                = nullptr;
    WorldParameters params;
    HeightField hf;

    UniformBuffer uniformBuffer;
    WorldUniforms uniforms;
    Texture2DArray foliageTexture;

    TerrainRenderer terrainRenderer;
    ForestRenderer forestRenderer;
    GrassRenderer grassRenderer;

    Pipeline skyPipeline;
    StaticDescriptorSet skySet;

    // Culling state (kept while freezeCulling is set)
    vec3 cullPosition = vec3(0, 0, 0);
    bool hasCullState = false;
    std::array<vec4, 6> cullFrustum;

    // GPU timestamps: begin of transfer() and end of render(), per frame in flight
    vk::QueryPool queryPool;
    double timestampPeriod = 1;
    struct TimestampSlot
    {
        uint64_t frameValue = 0;
        int frameNumber     = 0;
        bool written        = false;
    };
    std::array<TimestampSlot, 3> timestampSlots;
    // Timestamps per frame: begin, then one after each pass
    static constexpr uint32_t timestampsPerFrame = PASS_COUNT + 1;
    uint32_t timestampIndex                      = 0;
    uint32_t currentSlot                         = 0;
    bool timestampsSupported                     = false;
    double lastGpuMs                             = -1;

    std::chrono::steady_clock::time_point lastFrame;
    bool hasLastFrame = false;

    Stats statistics;
    bool initialized = false;

    // Frame times of the last frames (ring buffers for the plots)
    static constexpr int historySize = 300;
    std::vector<float> cpuHistory, gpuHistory;
    int historyIndex = 0;

    // Reads the timestamps of the newest completed frame. Returns the total time or -1.
    double readGpuTimes(std::array<double, PASS_COUNT>& passes, int& frameNumber);
    void writeTimestamp(vk::CommandBuffer cmd, uint32_t marker, vk::PipelineStageFlags2 stage);
    static void smooth(double& average, double value, int frames);
};

}  // namespace Vulkan
}  // namespace Saiga
