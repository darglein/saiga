/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"
#include "saiga/vulkan/texture/Sampler.h"
#include "saiga/vulkan/texture/Texture.h"

#include <array>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
class TerrainAtlas;
struct SceneLighting;

/**
 * The far terrain horizon shadows of RenderScene (S36a2, A6 of docs/stages/S36_lighting.plan.md, D11, D25): a
 * world-anchored RG32F map of horizonResolution^2 texels over horizonExtent metres; per texel two shadow heights
 * relative to the map's reference height: R = max over t >= 64 m of (h(x + t d) - t tan(el)) and G the same over t >=
 * cascadeCasterDistance (d the horizontal direction towards the shadow light). A receiver at height y is lit by
 * smoothstep(H - s, H, y) with s = max(4 m, 2 texel tan(el)); inside a cascade rectangle (terrain in the cascades) it
 * uses G (the cascades shadow the nearer terrain), elsewhere R.
 * - The march (horizon_map.comp) is conservative over a max-height mip chain of the pyramid (horizon_maxmip.comp,
 *   rebuilt when the pyramid changes on the GPU).
 * - The anchor (the map's corner, integer metres snapped to the texel grid) moves only when the camera leaves the
 *   inner half of the extent; its reference height is the camera's rounded to 64 m (height units).
 * - Three maps (layers of one array: previous, front, back), each with its own transform (anchor, reference, light): a
 *   new map is computed into the back in row bands when the light moved by more than 0.1 degrees, the anchor moved,
 *   the pyramid changed or a recompute was requested; rows per frame = max(horizonRowsPerFrame, the rows that finish
 *   before the light moves 0.1 degrees). Receivers sample mix(previous, front, blend): blend follows the back's
 *   progress from its value at the back's start (and continues at the same rate when nothing is computed), on
 *   completion previous <- front <- back and blend restarts at 0, so the shadows move continuously. forceFullUpdates
 * (and requestRecompute(true)) computes a due map in the frame and shows it at once. Below the horizon (no shadow
 * light) nothing is updated and the factor is 1.
 */
class SAIGA_VULKAN_API HorizonShadows
{
   public:
    // The transform of one map
    struct MapTransform
    {
        bool valid      = false;
        int32_t cornerX = 0, cornerZ = 0;  // integer metres
        int64_t refUnits = 0;              // the reference height (height units)
        Vec3 light       = Vec3(0, 1, 0);  // towards the light, the map was computed for
        double texel = 32, extent = 32768;
        int resolution = 0;
    };
    struct Input
    {
        // The atlas when it holds the camera space's resident pyramid, else nullptr (nothing is updated)
        const TerrainAtlas* atlas = nullptr;
        Vec3 camera               = Vec3::Zero();
        // Towards the shadow light; none: nothing is updated
        bool hasLight = false;
        Vec3 light    = Vec3(0, 1, 0);
        LightingParameters parameters;
        bool force     = false;
        uint64_t frame = 0;
        // The frame slot (the compute passes' descriptor sets are per slot)
        uint32_t frameIndex = 0;
    };

    void init(VulkanBase& base);
    void destroy();
    ~HorizonShadows() { destroy(); }

    // The frame's update (in the shadow section, outside of a pass): the chain when the pyramid changed, the due rows
    // of the back map; returns the rows computed
    int update(vk::CommandBuffer cmd, const Input& in);
    // The block's horizon fields for the camera (no map or off: the factor is 1)
    void fillBlock(SceneLighting& block, const Vec3& camera, const LightingParameters& p, bool hasLight) const;
    // Binding 9 of the lighting set (the three maps in eGeneral); none before the first update
    bool hasImages() const { return maps != nullptr; }
    vk::DescriptorImageInfo descriptor() const;
    Texture2DArray* images() { return maps.get(); }
    int frontLayer() const { return front; }
    const MapTransform& transform(int layer) const { return transforms[size_t(layer)]; }
    // The next update computes a new map (fully in that frame with full)
    void requestRecompute(bool full)
    {
        requested     = true;
        requestedFull = requestedFull || full;
    }
    HorizonStatistics statistics() const;
    uint64_t memoryBytes() const;
    // Tests: a layer's texels (R, G), row-major (waits for the device)
    std::vector<vec2> readLayerForTests(int layer);

   private:
    VulkanBase* base = nullptr;
    ComputePipeline chainPipeline, mapPipeline;
    // Per frame slot (a set must not be updated while a pending frame uses it)
    std::array<vk::DescriptorSet, 2> chainSets{}, mapSets{};
    Buffer chain;
    std::vector<int32_t> levelOffsets, levelSizes;
    std::unique_ptr<Texture2DArray> maps;
    Sampler sampler;
    bool linearFilter = false;
    std::array<MapTransform, 3> transforms;
    int previous = 0, front = 1, back = 2;
    // The back map in progress, its rows done and the blend at its start; the blend between previous and front
    bool computing = false;
    int rowsDone   = 0;
    float blend = 1, blendStart = 1;
    int lastRows = 0;
    // The pyramid the chain was built for (the atlas's set rebuilds and pyramid version)
    uint64_t chainRebuilds = ~uint64_t(0), chainVersion = ~uint64_t(0);
    const void* chainPyramid = nullptr;
    // The light of the last update (the angular speed) and the recomputes so far
    Vec3 lastLight         = Vec3(0, 1, 0);
    double degreesPerFrame = 0;
    uint64_t recomputes = 0, completed = 0, lastComplete = 0, restarts = 0;
    bool requested = false, requestedFull = false;
    int levelAt10km = 0;

    void ensureImages(int resolution);
    void buildChain(vk::CommandBuffer cmd, const TerrainAtlas& atlas, uint32_t frameIndex);
    void recordRows(vk::CommandBuffer cmd, const MapTransform& t, int layer, int row0, int rows,
                    const TerrainAtlas& atlas, const LightingParameters& p, uint32_t frameIndex);
    // A new map's transform for the camera and the light (the anchor kept while the camera stays in the inner half)
    MapTransform targetTransform(const Vec3& camera, const Vec3& light, const LightingParameters& p) const;
};

}  // namespace Vulkan
}  // namespace Saiga
