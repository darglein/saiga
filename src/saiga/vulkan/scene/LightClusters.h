/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/scene/LightingTypes.h"
#include "saiga/vulkan/scene/LocalLights.h"
#include "saiga/vulkan/texture/Texture.h"

#include <array>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
// The cluster grid of a frame (B3): clusterX x clusterY screen tiles (pixel row 0 at the top), clusterZ depth slices
// along the view direction: slice 0 from the near plane to clusterNear, slices 1.. exponential from clusterNear to
// clusterFar (beyond it no local light)
struct SAIGA_VULKAN_API ClusterGrid
{
    int x = 16, y = 9, z = 24;
    int maxLightsPerCluster = 128;
    double nearPlane = 0.1, clusterNear = 2, clusterFar = 1000;
    int width = 1, height = 1;
    double fovy = 1, aspect = 1;
    int count() const { return x * y * z; }
    // The view-space depth range of a slice
    void sliceDepth(int slice, double& d0, double& d1) const;
    // The slice of a view-space depth (-1: beyond clusterFar)
    int sliceOf(double depth) const;
    // The cluster of a pixel and a view depth (-1: beyond clusterFar or outside the image)
    int clusterAt(double px, double py, double depth) const;
    bool operator==(const ClusterGrid& o) const;
    bool operator!=(const ClusterGrid& o) const { return !(*this == o); }
};

/**
 * The clustered forward light culling of RenderScene (S36b, B3 of docs/stages/S36_lighting.plan.md): the cluster
 * AABBs in view space (light_cluster_bounds.comp, recomputed when the grid, the size or the projection changes), then
 * per frame light_cluster.comp (section lights, after cull): one invocation per cluster loops the frame's lights in
 * importance order (the buffer is sorted) with a sphere vs AABB test (spots: plus a cone test against the AABB's
 * bounding sphere) and writes up to maxLightsPerCluster indices into the cluster's fixed slot (uint, no atomics: the
 * lists are deterministic), the count and the total (the excess is counted as overflow and dropped: the least
 * important). The fragment shaders find their cluster from gl_FragCoord.xy and the view depth (clusters.glsl). The
 * table (binding 11), the indices (binding 12) and the per-light cluster counts (statistics) belong to a frame slot;
 * the statistics (max, mean, overflowed clusters, clusters per light) are read when the slot's frame has completed.
 * computeCpu is the CPU reference of the assignment (tests, probe_lighting).
 */
class SAIGA_VULKAN_API LightClusters
{
   public:
    void init(VulkanBase& base);
    void destroy();
    ~LightClusters() { destroy(); }

    // The cull of the frame (in the lights section): the bounds when the grid changed, the assignment; view: the
    // camera rotation's view matrix (camera-relative world -> view), projection: the frame's
    void record(vk::CommandBuffer cmd, uint32_t frameIndex, const ClusterGrid& grid, const mat4& view,
                const mat4& projection, const LocalLights& lights);
    // The frame slot's table and indices (binding 11, 12; valid after init)
    vk::DescriptorBufferInfo tableInfo(uint32_t frameIndex);
    vk::DescriptorBufferInfo indexInfo(uint32_t frameIndex);
    // The slot's previous frame has completed: its statistics become statistics() and clustersOf()
    void harvest(uint32_t frameIndex);
    struct Statistics
    {
        uint64_t frame         = 0;
        uint32_t clusters      = 0;
        uint32_t maxLights     = 0;  // the most lights in a cluster (after the cap)
        double meanLights      = 0;  // over every cluster
        uint32_t overflowed    = 0;  // clusters with more lights than slots
        uint64_t droppedLights = 0;  // the lights dropped by the cap (summed over the clusters)
        uint64_t memoryBytes   = 0;
    };
    const Statistics& statistics() const { return stats; }
    // The clusters a light touched in the newest harvested frame (by handle ID; 0: none or unknown)
    uint32_t clustersOf(uint32_t handle) const;
    // The counts per cluster of the newest harvested frame (count after the cap; tests, the probe)
    const std::vector<uint32_t>& lastCounts() const { return counts; }
    const std::vector<uint32_t>& lastTotals() const { return totals; }
    const ClusterGrid& lastGrid() const { return harvestedGrid; }
    // Reads the frame slot's indices and (count, total) table (waits for the device; tests); the slot of the last
    // record
    std::vector<uint32_t> readIndicesForTests(uint32_t frameIndex);
    std::vector<uint32_t> readTableForTests(uint32_t frameIndex);
    uint32_t lastRecordedSlot() const { return lastSlot; }

    // The heat map of the lights per cluster (LightingImage::Clusters): one slice (or slice < 0 the maximum over the
    // slices) as an R32F image of kCellPixels per cluster, computed from the frame slot's table (after record)
    Texture2D* recordDebug(vk::CommandBuffer cmd, uint32_t frameIndex, int slice);
    static constexpr int kCellPixels = 16;

    // The CPU reference: the light indices of every cluster (in importance order, capped) for the frame's records
    static std::vector<std::vector<uint32_t>> computeCpu(const ClusterGrid& grid, const mat4& view,
                                                         const mat4& projection,
                                                         const std::vector<LocalLights::GpuLight>& lights);
    // The view-space AABB of a cluster (the CPU reference of light_cluster_bounds.comp)
    static void clusterBounds(const ClusterGrid& grid, const mat4& projection, int cluster, vec3& lo, vec3& hi);
    // The reference test of one light against a view-space AABB (sphere, plus the cone for spots)
    static bool lightTouches(const LocalLights::GpuLight& light, const mat4& view, const vec3& lo, const vec3& hi);

   private:
    VulkanBase* base = nullptr;
    ComputePipeline boundsPipeline, clusterPipeline, debugPipeline;
    vk::DescriptorSet boundsSet;
    std::array<vk::DescriptorSet, 2> clusterSets{}, debugSets{};
    Buffer bounds;
    struct Frame
    {
        Buffer table, indices, lightCounts;
        bool pending   = false;
        uint64_t frame = 0;
        ClusterGrid grid;
        std::vector<uint32_t> handles;
    };
    std::array<Frame, 2> frames;
    ClusterGrid boundsGrid;
    mat4 boundsProjection = mat4::Zero();
    bool boundsValid      = false;
    Texture2D debugImage;
    Statistics stats;
    ClusterGrid harvestedGrid;
    std::vector<uint32_t> counts, totals;
    std::vector<std::pair<uint32_t, uint32_t>> clusterCountsByHandle;
    uint64_t frameCounter = 0;
    uint32_t lastSlot     = 0;
    void ensure(Frame& f, const ClusterGrid& grid, size_t lights);
    bool pipelinesCreated = false;
    void createPipelines();
};

}  // namespace Vulkan
}  // namespace Saiga
