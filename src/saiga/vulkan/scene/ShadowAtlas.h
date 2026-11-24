/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/scene/LightingTypes.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"
#include "saiga/vulkan/texture/Texture.h"

#include <array>
#include <functional>
#include <memory>
#include <vector>

#include <unordered_map>

namespace Saiga
{
namespace Vulkan
{
/**
 * The cached shadow atlas of the local lights (S36b, B4 of docs/stages/S36_lighting.plan.md; §10.7): a D16
 * Texture2DArray of two layers (atlasResolution^2): the static layer (static sets and the terrain; cached) and the
 * dynamic layer (dynamic instances only, re-rendered every frame for the tiles whose light sphere contains one);
 * receivers take the minimum visibility of both. Fixed tiles of atlasTile texels: a spot light takes 1 tile, a point
 * light 6 (cube faces, 90-degree perspective views). The faces store the linear depth distance(light, p) / range
 * (D16: range / 65536 per step at every distance; cleared to 1).
 *
 * Per frame (update, in the transfer section): the shadowed lights of the frame in importance order (a light that owns
 * tiles counts 20% more: lights near the cut do not alternate) take tiles until the atlas is full (the rest are
 * unshadowed and counted); tiles of lights that are no longer candidates stay cached until another light needs them
 * (oldest first). A static face is invalid when its key (the light's position, axis, cone, range and the terrain
 * selection under its sphere) changed or a dirty bound (markDirty: content changes; terrain tiles and patches)
 * intersects the light's sphere (also while the light is not a candidate: its cached faces see every dirty bound).
 * A light whose face count changed (a point light that became a spot or back) takes new tiles. The queue renders at
 * most facesPerFrame faces per frame (each a view of SceneCulling): the dynamic faces (a face whose frustum contains
 * a moving dynamic instance) oldest first, then the invalid static faces by importance and age, with a quarter of
 * the budget (at least 1) reserved for the static faces while any is due: neither queue starves; a skipped dynamic
 * face keeps its last image. A light is drawn unshadowed until each of its faces was rendered once. The tile table
 * (binding 14 of set 1) holds per face the camera-relative view-projection, the tile's rectangle and its texel factor,
 * written every frame (writeTable).
 */
class SAIGA_VULKAN_API ShadowAtlas
{
   public:
    // atlas_shadow.glsl's AtlasTile (std430, 96 bytes)
    struct GpuTile
    {
        mat4 viewProj;
        vec4 rect;    // xy the tile's origin in the atlas (uv), zw its size (uv)
        vec4 params;  // x 2 tan(half angle) / tile texels (a texel's size per m of distance), y 1 / tile texels, z 1
                      // the dynamic layer has content, w unused
    };
    static_assert(sizeof(GpuTile) == 96, "AtlasTile of atlas_shadow.glsl");

    // A shadowed light of the frame (in importance order)
    struct LightInput
    {
        uint32_t handle = 0;
        int index       = 0;  // the frame's light buffer index
        Vec3 world      = Vec3::Zero();
        LightData data;
        double importance = 0;
        // The moving dynamic instances whose sphere intersects the light's (world centre, radius): a face gets a
        // dynamic job only when one of them intersects its frustum (round 1, finding 4)
        std::vector<std::pair<Vec3, double>> movers;
        uint64_t terrainKey = 0;
    };
    // A face to render (or a dynamic tile to clear) this frame
    struct Face
    {
        int tile = 0, layer = 0, face = 0;
        size_t light   = 0;  // index into the input
        bool clearOnly = false;
        // Camera-relative (the frame's camera)
        mat4 viewProj              = mat4::Identity();
        std::array<Vec4, 6> planes = {};
        vec4 lightPush             = vec4(0, 0, 0, 1);
    };

    void init(VulkanBase& base);
    void destroy();
    ~ShadowAtlas() { destroy(); }

    // The image of the parameters (recreated through deferDelete on a change, every tile invalid); false: off
    void ensure(int resolution, int tile);
    void release();
    bool hasImage() const { return image != nullptr; }
    Texture2DArray* texture() { return image.get(); }
    vk::ImageView layerView(int layer) const { return layerViews[size_t(layer)]; }
    int resolution() const { return res; }
    int tileSize() const { return tileTexels; }
    int tilesPerSide() const { return res / std::max(tileTexels, 1); }

    // Content changes (world bounds, double): the static tiles whose light sphere intersects one are re-rendered
    void markDirty(const Vec3& lo, const Vec3& hi);
    void invalidateAll() { allDirty = true; }

    // Frees the tiles of the cached lights (not candidates of the frame) for which keep is false (removed lights,
    // lights without castShadows)
    void retain(const std::function<bool(uint32_t)>& keep);
    // The frame's allocation, cache and queue; camera: the frame's camera position (the faces' matrices)
    void update(const std::vector<LightInput>& lights, const Vec3& camera, int facesPerFrame, uint64_t frame,
                std::vector<Face>& faces);
    // The faces of update were recorded in this frame
    void rendered(const std::vector<Face>& faces, uint64_t frame);
    // The tile table of the frame slot (every allocated light's faces, camera-relative) and per input light the first
    // entry (-1: no tiles) and whether every face was rendered
    void writeTable(uint32_t frameIndex, const std::vector<LightInput>& lights, const Vec3& camera,
                    std::vector<int>& firstEntry, std::vector<bool>& ready);
    vk::DescriptorBufferInfo tableInfo(uint32_t frameIndex);

    const std::vector<AtlasTileInfo>& tiles() const { return tileState; }
    // The tiles of a light (empty: none)
    std::vector<int> tilesOf(uint32_t handle) const;
    const AtlasStatistics& statistics() const { return stats; }
    uint64_t memoryBytes() const;

    // The faces of a light (point 6, spot 1) and a face's camera-relative view-projection, planes and the texel factor
    static int faceCount(const LightData& d) { return d.kind == LightKind::Spot ? 1 : 6; }
    static void faceView(const LightData& d, const Vec3& lightRel, int face, mat4& viewProj,
                         std::array<Vec4, 6>& planes, double& tanHalf);

   private:
    struct Owner
    {
        std::vector<int> tiles;
        uint64_t lastUsed = 0;
        // The light's world position and range when it was last a candidate: the dirty bounds also reach the cached
        // owners that are not candidates (round 1, finding 2)
        Vec3 world   = Vec3::Zero();
        double range = 0;
    };
    VulkanBase* base = nullptr;
    std::unique_ptr<Texture2DArray> image;
    std::array<vk::ImageView, 2> layerViews{};
    int res = 0, tileTexels = 0;
    std::vector<AtlasTileInfo> tileState;
    std::vector<uint64_t> tileKeys;
    std::unordered_map<uint32_t, Owner> owners;
    std::vector<std::pair<Vec3, Vec3>> dirty;
    bool allDirty = false;
    std::array<Buffer, 2> tables;
    AtlasStatistics stats;
    static uint64_t faceKey(const LightInput& l, int face);
};

}  // namespace Vulkan
}  // namespace Saiga
