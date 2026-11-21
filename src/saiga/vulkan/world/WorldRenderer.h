/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/vulkan/world/HeightField.h"
#include "saiga/vulkan/world/PrototypeForest.h"
#include "saiga/vulkan/world/WorldSettings.h"

#include <chrono>
#include <deque>
#include <memory>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
class RenderScene;

// The camera rotation (RenderScene::setCamera: looks along -z, y up) of a view matrix (rotation and translation)
inline Quat viewRotation(const mat4& view)
{
    Mat3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r(i, j) = double(view(j, i));
    return Quat(r).normalized();
}

struct SAIGA_VULKAN_API WorldParameters
{
    HeightField::Parameters terrain;
    int treeCount = 200000;
    // The trees' placement and the foliage texture
    uint32_t seed = 1;
};

/**
 * The game side of the large procedural world prototype (S19a; S20 replaces it): the height field and the generators
 * (trees, foliage), the game settings (WorldSettings, GUI and MCP) and the statistics window. The rendering is
 * RenderScene's: attach() hands the terrain, water, grass and sky to the scene (setPrototypeTerrain) and creates the
 * trees as instance sets (PrototypeForest); update() applies the settings each frame. Everything is a uniform gray
 * ("clay") with simple lighting, to judge the geometry.
 *
 *   world.init(parameters);                      // CPU: the height field
 *   world.attach(scene);                         // renderHooks.init: the scene's prototype content and the trees
 *   per frame before the scene records it:       // e.g. the transfer hook
 *       world.update(); scene.setCamera(...); scene.setTime(world time);
 */
class SAIGA_VULKAN_API WorldRenderer
{
   public:
    WorldRenderer() = default;
    // It owns the trees' instance set handles (a copy would remove them twice)
    WorldRenderer(const WorldRenderer&)            = delete;
    WorldRenderer& operator=(const WorldRenderer&) = delete;
    ~WorldRenderer() { destroy(); }

    // Generates the height field
    void init(const WorldParameters& parameters);
    // With a height field generated elsewhere (e.g. once for the game's simulation and its render state; S13b): it is
    // shared, not copied; parameters.terrain is ignored
    void init(const WorldParameters& parameters, std::shared_ptr<const HeightField> heightField);
    // setPrototypeTerrain with the settings, then the trees' instance sets (outside of renderFrame)
    void attach(RenderScene& scene);
    // Forgets the scene (its instance sets are removed if the scene lives on) and the height field; the tree meshes and
    // material stay registered on the scene, and a later init + attach to the same scene reuses them by name
    void destroy();

    // Per frame, before the scene records the frame: the settings (setPrototypeSettings, the trees) and the statistics
    void update();

    // The settings window
    void imgui();
    // The detailed statistics window (RenderScene's GPU sections, CPU times, geometry, levels of detail, memory)
    void imguiStatistics();
    bool showStatistics = true;

    WorldSettings settings;

    const HeightField& heightField() const;
    // The height field (shared with the caller of the external init)
    const std::shared_ptr<const HeightField>& sharedHeightField() const { return hf; }
    const WorldParameters& parameters() const { return params; }
    PrototypeForest& forest() { return trees; }
    RenderScene* scene() const { return renderScene; }

    // The camera's projection of the settings' fovy (reversed-Z, infinite far plane, near 0.1)
    mat4 projection(float aspect) const;

    struct Stats
    {
        // Averages over the last frames: the frame interval (wall time), of it the render's waits for the frame slot,
        // the swap chain image and the present (vsync or the GPU; idle time), and the rest, the CPU's work
        double cpuFrameMs = 0;
        double cpuWaitMs  = 0;
        double cpuWorkMs  = 0;
        double fps        = 0;
        // RenderScene's GPU frame and sections (averaged, the app takes them), -1 if not available
        double gpuMs = -1;
        std::vector<double> gpuSectionMs;
        // CPU time per frame (averaged): the terrain node selection and the grass tile culling
        double cpuTerrainSelectMs = 0, cpuGrassCullMs = 0;
        int drawCalls    = 0;
        int terrainNodes = 0, terrainTriangles = 0;
        std::array<uint32_t, 3> treesVisible = {};
        uint64_t treeTriangles               = 0;
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

   private:
    WorldParameters params;
    std::shared_ptr<const HeightField> hf;
    RenderScene* renderScene = nullptr;
    PrototypeForest trees;

    std::chrono::steady_clock::time_point lastFrame;
    bool hasLastFrame     = false;
    uint64_t lastGpuFrame = 0;
    Stats statistics;

    static void smooth(double& average, double value, int frames);

   public:
    // The numbers the statistics window shows: means over the last kShownSeconds (buckets of kRefreshSeconds),
    // recomputed every kRefreshSeconds (numbers that change every frame cannot be read). The frame times themselves are
    // in the app's "Engine statistics".
    static constexpr double kShownSeconds   = 2.0;
    static constexpr double kRefreshSeconds = 0.5;
    struct Shown
    {
        double gpuMs = 0;
        std::vector<double> gpuSectionMs;  // RenderScene::gpuSectionNames(); empty without GPU times
        double terrainSelectMs = 0, grassCullMs = 0, recordingMs = 0;
        bool valid = false;
    };
    // The shown statistics (refreshed every kRefreshSeconds by update())
    const Shown& shownStatistics() const { return shown; }

   private:
    // The sums of one bucket of frames (each GPU frame once)
    struct Bucket
    {
        double gpuSum = 0;
        std::vector<double> gpuSectionSum;
        int gpuFrames           = 0;
        double terrainSelectSum = 0, grassCullSum = 0, recordingSum = 0;
        int cpuFrames = 0;
        std::chrono::steady_clock::time_point end;  // when the bucket was completed
    };
    std::deque<Bucket> buckets;
    size_t sectionCount = 0;  // the GPU section layout of the buckets
    Bucket current;
    uint64_t bucketGpuFrame = 0;
    std::chrono::steady_clock::time_point bucketStart;
    bool bucketStarted = false;
    Shown shown;
    void rotateBuckets();
};

}  // namespace Vulkan
}  // namespace Saiga
