/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <array>
#include <vector>

#include <unordered_map>

namespace Saiga
{
namespace Vulkan
{
/**
 * The local lights of RenderScene (S36b, B2 of docs/stages/S36_lighting.plan.md; docs/engine_design.md §10.7): the CPU
 * table of the lights (chunk + chunk-relative position: double-free, R-MATH-1), added one by one (entity lights of the
 * extract) or as one list per chunk (S30's piece lights, never entities). Handles are not reused.
 *
 * Per frame (prepareFrame, in the transfer section): the lights of the camera's space within clusterFar + range of the
 * camera whose sphere intersects the frustum, sorted by importance (priority * lumen / max(distance, range)^2, ties by
 * handle: deterministic) and capped at maxLights (the rest over budget), are written camera-relative (double -> float)
 * into the frame slot's light buffer (binding 10 of set 1; clusters.glsl's LocalLight): colour times the intensity in
 * candela (point lumen / 4 pi, spot lumen / pi) times the flicker (a sum of sines of the environment's flicker phase:
 * no absolute time in a shader).
 */
class SAIGA_VULKAN_API LocalLights
{
   public:
    // clusters.glsl's LocalLight (std430)
    struct GpuLight
    {
        vec4 posRange;   // xyz camera-relative, w range
        vec4 dirKind;    // xyz the spot's axis, w 0 point, 1 spot
        vec4 intensity;  // rgb colour * candela * flicker, w the first atlas tile-table entry (-1: unshadowed)
        vec4 cone;       // x cosOuter, y 1 / (cosInner - cosOuter), z 1 the atlas tiles are rendered, w unused
    };
    static_assert(sizeof(GpuLight) == 64, "LocalLight of clusters.glsl");
    // The buffer's header (uvec4: x the count) before the lights
    static constexpr size_t kHeaderBytes = 16;

    struct FrameInput
    {
        bool valid    = false;  // a camera
        SpaceId space = 0;
        Vec3 camera   = Vec3::Zero();
        // The camera's frustum planes (SceneCamera::planes: camera-relative, inward)
        std::array<Vec4, 5> planes = {};
        bool on                    = false;  // LightingParameters::localLights
        int maxLights              = 1024;
        double clusterFar          = 1000;
        float flickerPhase         = 0;  // EnvironmentState::phases[6]
        uint32_t highlight         = 0;  // a handle ID (0: none)
    };

    void init(VulkanBase& base);
    void destroy();
    ~LocalLights() { destroy(); }

    // The frame's selection and the frame slot's light buffer (its previous frame has completed)
    void prepareFrame(uint32_t frameIndex, const FrameInput& in);
    // The light buffer of the frame slot (the dummy-free binding: a buffer always exists after init)
    vk::DescriptorBufferInfo bufferInfo(uint32_t frameIndex) const;
    // The frame's lights in buffer order (importance), their GPU records, the highlighted light's index (-1: none)
    const std::vector<FrameLightInfo>& frameLights() const { return frameInfo; }
    std::vector<FrameLightInfo>& frameLightsMutable() { return frameInfo; }
    const std::vector<GpuLight>& frameRecords() const { return records; }
    int highlightIndex() const { return highlighted; }
    const LocalLightStatistics& statistics() const { return stats; }
    // The buffer index of a light in this frame (-1: not in the frame)
    int frameIndexOf(LightHandle h) const;
    // Rewrites the records' shadow fields (S36b step 4: the atlas's tile-table entries) into the frame slot's buffer
    void writeRecords(uint32_t frameIndex);
    // The intensity (candela) of a light, the flicker factor at a phase (the frame's: CPU, double)
    static float candela(const LightData& d);
    static float flickerFactor(const LightFlicker& f, float phase);
    // The CPU version of clusters.glsl's localLightRadiance (without the shadow and the energy compensation): the
    // radiance a record reflects at relPos towards v (probe_lighting)
    static vec3 reflectedRadiance(const GpuLight& light, const vec3& relPos, const vec3& n, const vec3& v,
                                  const vec3& albedo, float roughness, float metalness);
    // The importance of a light at a distance (B2)
    static double importance(const LightData& d, double distance);

    struct Light
    {
        LightHandle handle;
        ChunkId chunk;
        LightData data;
        // The list the light belongs to (0: a single light)
        uint32_t list = 0;
        // The world position (chunk origin + position, double)
        Vec3 worldPosition() const { return chunkOrigin(chunk) + data.position.cast<double>(); }
    };

    LightHandle add(const ChunkId& chunk, const LightData& light);
    // Asserts a live handle that is not part of a list
    void set(LightHandle h, const ChunkId& chunk, const LightData& light);
    void remove(LightHandle h);
    LightListHandle addList(const ChunkId& chunk, const std::vector<LightData>& lights);
    void removeList(LightListHandle list);

    // The live lights (unordered: a removal moves the last light into the gap)
    const std::vector<Light>& lights() const { return table; }
    const Light* find(LightHandle h) const;
    size_t size() const { return table.size(); }
    size_t listCount() const { return lists.size(); }
    // Changes with every add, set and remove (the shadow atlas's cache keys follow the lights themselves)
    uint64_t version() const { return changes; }

   private:
    VulkanBase* base = nullptr;
    std::array<Buffer, 2> buffers;
    std::vector<FrameLightInfo> frameInfo;
    std::vector<GpuLight> records;
    std::unordered_map<uint32_t, int> frameIndexByHandle;
    int highlighted = -1;
    LocalLightStatistics stats;
    std::vector<Light> table;
    std::unordered_map<uint32_t, size_t> indexOf;
    std::unordered_map<uint32_t, std::vector<LightHandle>> lists;
    uint32_t nextHandle = 1, nextList = 1;
    uint64_t changes = 0;
    void erase(LightHandle h);
};

}  // namespace Vulkan
}  // namespace Saiga
