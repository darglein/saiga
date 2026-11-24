/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/LightClusters.h"

#include "saiga/vulkan/pipeline/DescriptorSetLayout.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

namespace
{
vk::Buffer handle(Buffer& b)
{
    Memory::SafeAccessor acc(*b.memoryLocation());
    return b.memoryLocation()->data;
}
// light_cluster_bounds.comp: Push
struct BoundsPush
{
    mat4 invProj;
    uint32_t grid[4];  // x, y, z, unused
    float depth[4];    // near plane, clusterNear, clusterFar, unused
};
// light_cluster.comp: Push
struct ClusterPush
{
    mat4 view;
    uint32_t grid[4];  // x, y, z, maxLightsPerCluster
    uint32_t lightCount;
    uint32_t pad[3];
};
// light_cluster.comp's debug mode (light_cluster_debug in the same file with a define): Push
struct DebugPush
{
    uint32_t grid[4];  // x, y, z, cell pixels
    int32_t slice;
    uint32_t pad[3];
};
static_assert(sizeof(BoundsPush) == 96 && sizeof(ClusterPush) == 96 && sizeof(DebugPush) == 32,
              "push constants of the cluster shaders");

void memoryBarrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
{
    vk::MemoryBarrier2 b(srcStage, srcAccess, dstStage, dstAccess);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, b, nullptr, nullptr));
}

// The ray of an NDC point through the projection's inverse, scaled to view depth 1 (-z = 1)
vec3 rayAt(const mat4& invProj, float x, float y)
{
    const vec4 h = invProj * vec4(x, y, 0.5f, 1.0f);
    const vec3 d(h(0) / h(3), h(1) / h(3), h(2) / h(3));
    return d / -d(2);
}
}  // namespace

//==================================================================================================
// ClusterGrid

void ClusterGrid::sliceDepth(int slice, double& d0, double& d1) const
{
    if (slice <= 0)
    {
        d0 = nearPlane;
        d1 = clusterNear;
        return;
    }
    const double r = clusterFar / clusterNear;
    d0             = clusterNear * std::pow(r, double(slice - 1) / double(z - 1));
    d1             = clusterNear * std::pow(r, double(slice) / double(z - 1));
}

int ClusterGrid::sliceOf(double depth) const
{
    if (depth > clusterFar) return -1;
    if (depth < clusterNear) return 0;
    const double k = std::log(depth / clusterNear) * double(z - 1) / std::log(clusterFar / clusterNear);
    return std::min(1 + int(std::floor(k)), z - 1);
}

int ClusterGrid::clusterAt(double px, double py, double depth) const
{
    const int s = sliceOf(depth);
    if (s < 0 || px < 0 || py < 0 || px >= width || py >= height) return -1;
    const int tx = std::min(int(px / double(width) * x), x - 1);
    const int ty = std::min(int(py / double(height) * y), y - 1);
    return (s * y + ty) * x + tx;
}

bool ClusterGrid::operator==(const ClusterGrid& o) const
{
    return x == o.x && y == o.y && z == o.z && maxLightsPerCluster == o.maxLightsPerCluster &&
           nearPlane == o.nearPlane && clusterNear == o.clusterNear && clusterFar == o.clusterFar && width == o.width &&
           height == o.height && fovy == o.fovy && aspect == o.aspect;
}

//==================================================================================================
// LightClusters

void LightClusters::init(VulkanBase& b)
{
    destroy();
    base = &b;
    // Valid (empty) buffers from the start: the lighting set binds them every frame; the pipelines on first use (a
    // scene without local lights compiles none)
    for (Frame& f : frames) ensure(f, ClusterGrid(), 1);
}

void LightClusters::createPipelines()
{
    if (pipelinesCreated) return;
    pipelinesCreated = true;
    const auto c     = vk::ShaderStageFlagBits::eCompute;
    {
        DescriptorSetLayout l;
        l.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, c));
        boundsPipeline.init(*base, 1);
        boundsPipeline.addDescriptorSetLayout(l);
        boundsPipeline.addPushConstantRange({c, 0, sizeof(BoundsPush)});
        boundsPipeline.shaderPipeline.loadCompute(base->device, "vulkan/scene/lighting/light_cluster_bounds.comp");
        boundsPipeline.create();
        for (auto& s : boundsSets) s = boundsPipeline.createRawDescriptorSet();
    }
    {
        DescriptorSetLayout l;
        for (uint32_t i = 0; i < 5; ++i)
            l.addBinding(i, vk::DescriptorSetLayoutBinding(i, vk::DescriptorType::eStorageBuffer, 1, c));
        clusterPipeline.init(*base, 1);
        clusterPipeline.addDescriptorSetLayout(l);
        clusterPipeline.addPushConstantRange({c, 0, sizeof(ClusterPush)});
        clusterPipeline.shaderPipeline.loadCompute(base->device, "vulkan/scene/lighting/light_cluster.comp");
        clusterPipeline.create();
        for (auto& s : clusterSets) s = clusterPipeline.createRawDescriptorSet();
    }
    {
        DescriptorSetLayout l;
        l.addBinding(0, vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, c));
        l.addBinding(1, vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageImage, 1, c));
        debugPipeline.init(*base, 1);
        debugPipeline.addDescriptorSetLayout(l);
        debugPipeline.addPushConstantRange({c, 0, sizeof(DebugPush)});
        debugPipeline.shaderPipeline.loadCompute(base->device, "vulkan/scene/lighting/light_cluster.comp",
                                                 "#define CLUSTER_DEBUG_IMAGE\n");
        debugPipeline.create();
        for (auto& s : debugSets) s = debugPipeline.createRawDescriptorSet();
    }
}

void LightClusters::destroy()
{
    if (!base) return;
    if (pipelinesCreated)
    {
        boundsPipeline.destroy();
        clusterPipeline.destroy();
        debugPipeline.destroy();
    }
    pipelinesCreated = false;
    auto* pool       = &base->descriptorPool;
    auto free        = [&](vk::DescriptorSet& s)
    {
        auto set = s;
        if (set) base->deferDelete([pool, set]() { pool->freeDescriptorSet(set); });
        s = nullptr;
    };
    for (auto& s : boundsSets) free(s);
    boundsSetVersions = {};
    for (auto& s : clusterSets) free(s);
    for (auto& s : debugSets) free(s);
    bounds.destroy();
    for (Frame& f : frames)
    {
        f.table.destroy();
        f.indices.destroy();
        f.lightCounts.destroy();
    }
    debugImage.destroy();
    boundsValid = false;
    base        = nullptr;
}

void LightClusters::ensure(Frame& f, const ClusterGrid& grid, size_t lights)
{
    const size_t n        = size_t(grid.count());
    const auto storage    = vk::BufferUsageFlagBits::eStorageBuffer;
    const auto host       = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    const size_t table    = n * 8;
    const size_t indices  = n * size_t(grid.maxLightsPerCluster) * 4;
    const size_t perLight = std::max<size_t>(lights, 1) * 4;
    if (f.table.size() < table)
    {
        f.table.createBuffer(*base, table, storage, host);
        std::memset(f.table.getMappedPointer(), 0, table);
    }
    if (f.indices.size() < indices)
        f.indices.createBuffer(*base, indices, storage | vk::BufferUsageFlagBits::eTransferSrc,
                               vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (f.lightCounts.size() < perLight)
        f.lightCounts.createBuffer(*base, std::max(perLight, size_t(1024)), storage, host);
}

void LightClusters::record(vk::CommandBuffer cmd, uint32_t frameIndex, const ClusterGrid& grid, const mat4& view,
                           const mat4& projection, const LocalLights& lights)
{
    createPipelines();
    Frame& f                = frames[frameIndex];
    const size_t lightCount = lights.frameRecords().size();
    ensure(f, grid, lightCount);
    const uint32_t n = uint32_t(grid.count());
    // The per-light counts (the slot's previous frame has completed)
    std::memset(f.lightCounts.getMappedPointer(), 0, std::max<size_t>(lightCount, 1) * 4);
    if (!bounds || bounds.size() < size_t(n) * 32)
    {
        bounds.createBuffer(*base, size_t(n) * 32, vk::BufferUsageFlagBits::eStorageBuffer,
                            vk::MemoryPropertyFlagBits::eDeviceLocal);
        boundsValid = false;
        boundsVersion++;
    }
    // The bounds when the grid or the projection changed (the previous frames' reads come before the write)
    if (!boundsValid || boundsGrid != grid || boundsProjection != projection)
    {
        memoryBarrier(cmd, Stage::eComputeShader, Access::eShaderStorageRead, Stage::eComputeShader,
                      Access::eShaderStorageWrite);
        // The slot's set (its previous frame has completed), only when it points to another buffer
        if (boundsSetVersions[frameIndex] != boundsVersion)
        {
            vk::DescriptorBufferInfo info(handle(bounds), 0, VK_WHOLE_SIZE);
            base->device.updateDescriptorSets(
                vk::WriteDescriptorSet(boundsSets[frameIndex], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
                                       &info),
                nullptr);
            boundsSetVersions[frameIndex] = boundsVersion;
        }
        if (boundsPipeline.bind(cmd))
        {
            BoundsPush push;
            push.invProj  = inverse(projection);
            push.grid[0]  = uint32_t(grid.x);
            push.grid[1]  = uint32_t(grid.y);
            push.grid[2]  = uint32_t(grid.z);
            push.grid[3]  = 0;
            push.depth[0] = float(grid.nearPlane);
            push.depth[1] = float(grid.clusterNear);
            push.depth[2] = float(grid.clusterFar);
            push.depth[3] = 0;
            boundsPipeline.bindRawDescriptorSet(cmd, boundsSets[frameIndex]);
            boundsPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
            cmd.dispatch((n + 63) / 64, 1, 1);
        }
        boundsGrid       = grid;
        boundsProjection = projection;
        boundsValid      = true;
    }
    // The bounds' write, and the previous frames' fragment reads of the shared index buffers (none: per slot) before
    // the assignment
    memoryBarrier(cmd, Stage::eComputeShader | Stage::eFragmentShader,
                  Access::eShaderStorageWrite | Access::eShaderStorageRead, Stage::eComputeShader,
                  Access::eShaderStorageRead | Access::eShaderStorageWrite);
    std::array<vk::DescriptorBufferInfo, 5> infos = {vk::DescriptorBufferInfo(handle(bounds), 0, VK_WHOLE_SIZE),
                                                     lights.bufferInfo(frameIndex),
                                                     vk::DescriptorBufferInfo(handle(f.table), 0, VK_WHOLE_SIZE),
                                                     vk::DescriptorBufferInfo(handle(f.indices), 0, VK_WHOLE_SIZE),
                                                     vk::DescriptorBufferInfo(handle(f.lightCounts), 0, VK_WHOLE_SIZE)};
    std::array<vk::WriteDescriptorSet, 5> writes;
    for (uint32_t i = 0; i < 5; ++i)
        writes[i] = vk::WriteDescriptorSet(clusterSets[frameIndex], i, 0, 1, vk::DescriptorType::eStorageBuffer,
                                           nullptr, &infos[i]);
    base->device.updateDescriptorSets(writes, nullptr);
    if (clusterPipeline.bind(cmd))
    {
        ClusterPush push;
        push.view       = view;
        push.grid[0]    = uint32_t(grid.x);
        push.grid[1]    = uint32_t(grid.y);
        push.grid[2]    = uint32_t(grid.z);
        push.grid[3]    = uint32_t(grid.maxLightsPerCluster);
        push.lightCount = uint32_t(lightCount);
        push.pad[0] = push.pad[1] = push.pad[2] = 0;
        clusterPipeline.bindRawDescriptorSet(cmd, clusterSets[frameIndex]);
        clusterPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
        cmd.dispatch((n + 63) / 64, 1, 1);
    }
    // The fragment shaders (and the debug image) read the lists; the host reads the table and the counts
    memoryBarrier(cmd, Stage::eComputeShader, Access::eShaderStorageWrite,
                  Stage::eFragmentShader | Stage::eComputeShader, Access::eShaderStorageRead);
    vk::MemoryBarrier2 host(Stage::eComputeShader, Access::eShaderStorageWrite, Stage::eHost, Access::eHostRead);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, host, nullptr, nullptr));
    f.pending = true;
    f.frame   = ++frameCounter;
    lastSlot  = frameIndex;
    f.grid    = grid;
    f.handles.clear();
    for (const FrameLightInfo& i : lights.frameLights()) f.handles.push_back(i.handle);
}

vk::DescriptorBufferInfo LightClusters::tableInfo(uint32_t frameIndex)
{
    return vk::DescriptorBufferInfo(handle(frames[frameIndex].table), 0, VK_WHOLE_SIZE);
}

vk::DescriptorBufferInfo LightClusters::indexInfo(uint32_t frameIndex)
{
    return vk::DescriptorBufferInfo(handle(frames[frameIndex].indices), 0, VK_WHOLE_SIZE);
}

void LightClusters::harvest(uint32_t frameIndex)
{
    Frame& f = frames[frameIndex];
    if (!f.pending) return;
    f.pending      = false;
    const size_t n = size_t(f.grid.count());
    const auto* t  = static_cast<const uint32_t*>(f.table.getMappedPointer());
    counts.assign(n, 0);
    totals.assign(n, 0);
    Statistics s;
    s.frame      = f.frame;
    s.clusters   = uint32_t(n);
    uint64_t sum = 0;
    for (size_t c = 0; c < n; ++c)
    {
        counts[c] = t[2 * c];
        totals[c] = t[2 * c + 1];
        sum += counts[c];
        s.maxLights = std::max(s.maxLights, counts[c]);
        if (totals[c] > counts[c])
        {
            s.overflowed++;
            s.droppedLights += totals[c] - counts[c];
        }
    }
    s.meanLights  = n ? double(sum) / double(n) : 0.0;
    s.memoryBytes = bounds.size();
    for (const Frame& o : frames) s.memoryBytes += o.table.size() + o.indices.size() + o.lightCounts.size();
    stats          = s;
    harvestedGrid  = f.grid;
    const auto* lc = static_cast<const uint32_t*>(f.lightCounts.getMappedPointer());
    clusterCountsByHandle.clear();
    for (size_t i = 0; i < f.handles.size(); ++i) clusterCountsByHandle.emplace_back(f.handles[i], lc[i]);
    std::sort(clusterCountsByHandle.begin(), clusterCountsByHandle.end());
}

void LightClusters::clearStatistics()
{
    Statistics s;
    s.memoryBytes = bounds.size();
    for (const Frame& o : frames) s.memoryBytes += o.table.size() + o.indices.size() + o.lightCounts.size();
    stats = s;
    counts.clear();
    totals.clear();
    clusterCountsByHandle.clear();
}

uint32_t LightClusters::clustersOf(uint32_t handle) const
{
    auto it = std::lower_bound(clusterCountsByHandle.begin(), clusterCountsByHandle.end(),
                               std::make_pair(handle, uint32_t(0)));
    return it != clusterCountsByHandle.end() && it->first == handle ? it->second : 0u;
}

std::vector<uint32_t> LightClusters::readIndicesForTests(uint32_t frameIndex)
{
    base->device.waitIdle();
    Frame& f = frames[frameIndex];
    std::vector<uint32_t> out(f.indices.size() / 4);
    f.indices.stagedDownload(out.data());
    return out;
}

std::vector<uint32_t> LightClusters::readTableForTests(uint32_t frameIndex)
{
    base->device.waitIdle();
    Frame& f = frames[frameIndex];
    std::vector<uint32_t> out(size_t(f.grid.count()) * 2);
    std::memcpy(out.data(), f.table.getMappedPointer(), out.size() * 4);
    return out;
}

Texture2D* LightClusters::recordDebug(vk::CommandBuffer cmd, uint32_t frameIndex, int slice)
{
    Frame& f             = frames[frameIndex];
    const ClusterGrid& g = f.grid;
    const uint32_t w = uint32_t(g.x * kCellPixels), h = uint32_t(g.y * kCellPixels);
    if (!debugImage.memoryLocation || debugImage.width != w || debugImage.height != h)
    {
        if (debugImage.memoryLocation) debugImage.destroy();
        debugImage.create(*base, w, h, vk::Format::eR32Sfloat,
                          vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);
    }
    vk::DescriptorBufferInfo table(handle(f.table), 0, VK_WHOLE_SIZE);
    vk::DescriptorImageInfo dst(nullptr, debugImage.view(), vk::ImageLayout::eGeneral);
    std::array<vk::WriteDescriptorSet, 2> writes = {
        vk::WriteDescriptorSet(debugSets[frameIndex], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &table),
        vk::WriteDescriptorSet(debugSets[frameIndex], 1, 0, 1, vk::DescriptorType::eStorageImage, &dst)};
    base->device.updateDescriptorSets(writes, nullptr);
    debugImage.transitionImageLayout(cmd, vk::ImageLayout::eGeneral);
    if (debugPipeline.bind(cmd))
    {
        DebugPush push;
        push.grid[0] = uint32_t(g.x);
        push.grid[1] = uint32_t(g.y);
        push.grid[2] = uint32_t(g.z);
        push.grid[3] = uint32_t(kCellPixels);
        push.slice   = slice;
        push.pad[0] = push.pad[1] = push.pad[2] = 0;
        debugPipeline.bindRawDescriptorSet(cmd, debugSets[frameIndex]);
        debugPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
        cmd.dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }
    debugImage.transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    return &debugImage;
}

//==================================================================================================
// The CPU reference

void LightClusters::clusterBounds(const ClusterGrid& grid, const mat4& projection, int cluster, vec3& lo, vec3& hi)
{
    const mat4 invProj = inverse(projection);
    const int tx = cluster % grid.x, ty = (cluster / grid.x) % grid.y, tz = cluster / (grid.x * grid.y);
    double d0, d1;
    grid.sliceDepth(tz, d0, d1);
    lo = vec3(1e30f, 1e30f, 1e30f);
    hi = vec3(-1e30f, -1e30f, -1e30f);
    for (int k = 0; k < 4; ++k)
    {
        const float x = -1.0f + 2.0f * float(tx + (k & 1)) / float(grid.x);
        const float y = -1.0f + 2.0f * float(ty + (k >> 1)) / float(grid.y);
        const vec3 r  = rayAt(invProj, x, y);
        for (double d : {d0, d1})
        {
            const vec3 p = r * float(d);
            lo           = lo.cwiseMin(p);
            hi           = hi.cwiseMax(p);
        }
    }
}

bool LightClusters::inRange(const vec4& range, int x, int y, int z)
{
    const int r[3] = {int(range(0)), int(range(1)), int(range(2))};
    const int t[3] = {x, y, z};
    for (int k = 0; k < 3; ++k)
        if (t[k] < (r[k] & 255) || t[k] > (r[k] >> 8)) return false;
    return true;
}

vec4 LightClusters::clusterRange(const ClusterGrid& grid, const mat4& view, const mat4& projection,
                                 const LocalLights::GpuLight& L)
{
    const vec4 cv  = view * vec4(L.posRange(0), L.posRange(1), L.posRange(2), 1);
    const double r = double(L.posRange(3)) * 1.001;
    const double d = -double(cv(2));
    // The slices of [d - r, d + r] (beyond clusterFar: the last slice; the light selection drops lights beyond it)
    const int z0 = std::max(grid.sliceOf(std::max(d - r, grid.nearPlane)), 0);
    int z1       = grid.sliceOf(d + r);
    if (z1 < 0) z1 = grid.z - 1;
    int x0 = 0, x1 = grid.x - 1, y0 = 0, y1 = grid.y - 1;
    if (d - r > grid.nearPlane)
    {
        // The AABB's corners in NDC (all in front of the near plane)
        double nx0 = 1e30, nx1 = -1e30, ny0 = 1e30, ny1 = -1e30;
        for (int k = 0; k < 8; ++k)
        {
            const vec4 p(cv(0) + float((k & 1) ? r : -r), cv(1) + float((k & 2) ? r : -r),
                         cv(2) + float((k & 4) ? r : -r), 1);
            const vec4 c   = projection * p;
            const double x = double(c(0)) / double(c(3)), y = double(c(1)) / double(c(3));
            nx0 = std::min(nx0, x);
            nx1 = std::max(nx1, x);
            ny0 = std::min(ny0, y);
            ny1 = std::max(ny1, y);
        }
        auto tile = [](double ndc, int n) { return std::clamp(int(std::floor((ndc * 0.5 + 0.5) * n)), 0, n - 1); };
        x0        = tile(nx0, grid.x);
        x1        = tile(nx1, grid.x);
        y0        = tile(ny0, grid.y);
        y1        = tile(ny1, grid.y);
    }
    return vec4(float(x0 + 256 * x1), float(y0 + 256 * y1), float(z0 + 256 * z1), 0);
}

bool LightClusters::lightTouches(const LocalLights::GpuLight& L, const mat4& view, const vec3& lo, const vec3& hi)
{
    const vec4 pv = view * vec4(L.posRange(0), L.posRange(1), L.posRange(2), 0);
    const vec3 p(pv(0), pv(1), pv(2));
    const float r = L.posRange(3);
    const vec3 d  = (lo - p).cwiseMax(vec3(0, 0, 0)).cwiseMax(p - hi);
    if (d.dot(d) > r * r) return false;
    if (L.dirKind(3) < 0.5f) return true;
    // The cone against the AABB's bounding sphere (Wronski)
    const vec4 dv = view * vec4(L.dirKind(0), L.dirKind(1), L.dirKind(2), 0);
    const vec3 dir(dv(0), dv(1), dv(2));
    const vec3 centre = (lo + hi) * 0.5f;
    const float rad   = (hi - lo).norm() * 0.5f;
    const vec3 v      = centre - p;
    const float vLen2 = v.dot(v);
    const float v1    = v.dot(dir);
    const float cosA  = L.cone(0);
    const float sinA  = std::sqrt(std::max(0.0f, 1.0f - cosA * cosA));
    const float dist  = cosA * std::sqrt(std::max(0.0f, vLen2 - v1 * v1)) - v1 * sinA;
    return !(dist > rad || v1 > rad + r || v1 < -rad);
}

std::vector<std::vector<uint32_t>> LightClusters::computeCpu(const ClusterGrid& grid, const mat4& view,
                                                             const mat4& projection,
                                                             const std::vector<LocalLights::GpuLight>& lights)
{
    std::vector<std::vector<uint32_t>> out(size_t(grid.count()));
    for (int c = 0; c < grid.count(); ++c)
    {
        vec3 lo, hi;
        clusterBounds(grid, projection, c, lo, hi);
        const int tx = c % grid.x, ty = (c / grid.x) % grid.y, tz = c / (grid.x * grid.y);
        for (uint32_t i = 0; i < uint32_t(lights.size()); ++i)
            if (inRange(lights[i].clusterRange, tx, ty, tz) && lightTouches(lights[i], view, lo, hi) &&
                out[size_t(c)].size() < size_t(grid.maxLightsPerCluster))
                out[size_t(c)].push_back(i);
    }
    return out;
}

}  // namespace Vulkan
}  // namespace Saiga
