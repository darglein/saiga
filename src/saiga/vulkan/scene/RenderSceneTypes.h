/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/math.h"
#include "saiga/export.h"
#include "saiga/vulkan/scene/LightingTypes.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
// The tone map of RenderScene's post pass (docs/engine_design.md §10.5). Identity is the clay mode: the scene shaders
// write display values, which are shown as they are (no exposure). Aces: the scene shaders write linear radiance
// (pre-exposed), tone mapped with the ACES fit of Narkowicz and the sRGB curve.
enum class ToneMap : uint8_t
{
    Identity,
    Aces
};

// The parameters of the RenderScene frame (docs/engine_design.md §10.1, §17.9; S13: the frame shell; S17: the HDR
// frame). Tone map and exposure can also be changed at run time (RenderScene::setToneMap, setExposure).
struct RenderSceneParameters
{
    // MSAA samples of the scene pass, clamped to the largest supported count not above it (RenderScene::formats())
    int msaaSamples = 4;
    // A display value; converted to linear (and pre-exposed) when the tone map is not Identity
    vec4 clearColor = vec4(57, 57, 57, 255) / 255.0f;
    bool imgui      = true;
    ToneMap toneMap = ToneMap::Identity;
    // Fixed until auto exposure (S37); ignored (1) with Identity. Without an environment (setEnvironment never called)
    // the frame's exposure; with one (S36) a multiplier on top of its EV100 (exposureFromEv100: 2 is one stop brighter)
    float exposure = 1;
    // The lighting's parameters (S36; RenderScene::setLightingParameters changes them at run time)
    LightingParameters lighting;

    // The camera's near plane (reversed-Z with an infinite far plane, S18)
    float nearPlane = 0.1f;
    // The initial capacities of the scene content in elements (S18): the mesh pool's vertices and indices, the static
    // instances and the dynamic instances. Every buffer grows when needed (2x or the required size). The storage
    // buffers of the culling and the draws (static instances 32 bytes each, dynamics 80, compacted entries 8 per
    // static and dynamic instance, scratch 2 KiB per work chunk of 256 visited instances, offset table 32 per set and
    // chunk of a dynamic) are bound whole: each is limited to the device's maxStorageBufferRange (128 MiB on lavapipe
    // and many mobile GPUs, e.g. about 4.19M static instances; 4 GiB on desktop GPUs). Growth stops at the limit; a
    // content change that needs more is a fatal error (SAIGA_EXIT_ERROR, every build).
    uint32_t vertexCapacity         = 1 << 16;
    uint32_t indexCapacity          = 1 << 17;
    uint32_t staticInstanceCapacity = 1 << 14;
    uint32_t dynamicCapacity        = 1 << 10;
    // Lower limits than the device's for tests (0: the device's maxStorageBufferRange and maxComputeWorkGroupCount)
    uint64_t storageBufferLimit = 0;
    uint32_t workGroupLimit     = 0;
    // Draw a pipeline class's buckets with one multi-draw-indirect call if the device supports it; false draws them
    // one by one (tests compare both paths without a new device)
    bool allowMultiDrawIndirect = true;
    // The terrain's upload cap per frame (S20, TerrainAtlas): each frame records at most this many bytes of terrain
    // copies (tiles, patches, page-table entries, pyramid rows, grass masks); the terrain's staging buffers are created
    // once at this size (plus alignment) per frame in flight and never grow
    size_t terrainUploadBytesPerFrame = size_t(16) << 20;
};

//==================================================================================================
// Terrain (S20, docs/engine_design.md §10.1, §12.2): the data the RenderSink terrain calls carry. The engine's store
// keeps the same objects by shared_ptr (no copies between store, sink and the scene's upload queue).

// Fixed per GPU terrain set (D5): from the space template and the engine's TerrainViewSettings
struct TerrainLayout
{
    // From the SpaceTemplate: 1 m (257 x 257 tiles) or 2 m (129 x 129); the space size in chunks (multiple of 16)
    int32_t baseSpacing = 2, sizeChunks = 128;
    // From TerrainViewSettings: tiles per streamed level (= GPU atlas layers per level), grass mask layers
    int32_t tilesPerLevel = 48, grassMaskLayers = 16;
    // The pyramid's spacing in metres, for both base spacings
    static constexpr int32_t pyramidSpacing = 32;

    // 257 / 129: tiles share their borders
    int samplesPerSide() const { return 256 / baseSpacing + 1; }
    // The levels with a spacing below 32 m: 5 (1 m), 4 (2 m)
    int streamedLevels() const { return baseSpacing == 1 ? 5 : 4; }
    // The CDLOD level of the 4096 m root nodes: 7 (1 m), 6 (2 m)
    int rootLevel() const { return baseSpacing == 1 ? 7 : 6; }
    // The pyramid's samples per side: 8 * sizeChunks + 1 (1025 for 32 km)
    int pyramidSamplesPerSide() const { return 8 * sizeChunks + 1; }
    // Grass mask texels per side: one per base cell
    int grassMaskSize() const { return 256 / baseSpacing; }
    // The spacing of level L in metres
    int32_t levelSpacing(int level) const { return baseSpacing << level; }
    // Tiles per side of the space at a streamed level
    int32_t tilesPerSide(int level) const { return sizeChunks >> level; }
    bool operator==(const TerrainLayout& o) const
    {
        return baseSpacing == o.baseSpacing && sizeChunks == o.sizeChunks && tilesPerLevel == o.tilesPerLevel &&
               grassMaskLayers == o.grassMaskLayers;
    }
    bool operator!=(const TerrainLayout& o) const { return !(*this == o); }
};

// RegionId level L < streamedLevels: 2^L chunks at spacing baseSpacing * 2^L
struct TerrainTileData
{
    // Height units, of these samples
    int32_t minHeight = 0, maxHeight = 0;
    // samplesPerSide^2, row j = z, index j * n + i; sample (i, j) at regionOrigin + (i, 0, j) * spacing
    std::shared_ptr<const std::vector<int32_t>> heights;
};

// The whole space at 32 m: (8 * sizeChunks + 1)^2 samples (1025^2 for 32 km)
struct TerrainPyramidData
{
    TerrainLayout layout;
    // The water plane until S39's setWater (amendment A9)
    std::optional<int32_t> seaLevel;
    int32_t minHeight = 0, maxHeight = 0;
    std::shared_ptr<const std::vector<int32_t>> heights;
};

// One chunk: (256 / baseSpacing)^2 density texels, one per base cell (0..255)
struct GrassMaskData
{
    int32_t size = 0;
    std::shared_ptr<const std::vector<uint8_t>> density;
};

//==================================================================================================
// Scene content (S18, docs/engine_design.md §10.1, §10.2, §10.8)

// Explicit IDs (the World's name tables assign them, S19b); 0 = none
using MeshId     = uint16_t;
using MaterialId = uint16_t;

// The game range of mesh and material IDs, 1 .. kGameReservedIds - 1 (S19a): content the game registers directly on
// RenderScene (the prototype's tree meshes and their material, names prefixed "prototype/"; S19b: level geometry)
// takes its IDs through RenderScene::reserveIds; S19b's World name tables assign from kGameReservedIds upward; S30's
// generated meshes take IDs from the top
constexpr uint32_t kGameReservedIds = 256;
// The two ID spaces of RenderScene::reserveIds
enum class SceneIdKind : uint8_t
{
    Mesh,
    Material
};

// The vertex of the mesh pool (28 bytes)
struct MeshVertex
{
    vec3 position = vec3(0, 0, 0);
    // Octahedral encoding, two snorm16 (packNormalOct)
    uint32_t normal = 0;
    // sRGB8 RGB + linear alpha (packColor), or a palette index (MaterialData::usePalette); decoded in the shader
    uint32_t color = 0xFFFFFFFFu;
    vec2 uv        = vec2(0, 0);
};
static_assert(sizeof(MeshVertex) == 28, "MeshVertex is 28 bytes");

// One level of detail: used up to endDistance (distance from the camera to the instance's bounding sphere centre, not
// scaled by the instance scale), i.e. LOD i is the first with distance <= endDistance; beyond the last LOD the instance
// is culled. An endDistance equal to the previous LOD's makes this LOD empty: it is never selected, draws nothing
struct MeshLod
{
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t> indices;  // relative to the LOD's vertices
    float endDistance = 1000;
};

struct MeshLods
{
    // 1..4 LODs with non-decreasing endDistance >= 0 (asserted by RenderScene::addMesh; equal: an empty LOD)
    std::vector<MeshLod> lods;
    // The bounding sphere of all LODs (mesh space)
    vec3 sphereCenter  = vec3(0, 0, 0);
    float sphereRadius = 0;
    // Added to the radius (mesh units, scaled with the instance): wind sway etc. (R-MECH-1)
    float boundsPadding = 0;
};

enum class ShadingModel : uint8_t
{
    FlatColor,
    SmoothTextured
};

// The pipeline classes of the instance draws (MaterialData::pipelineClass): 0 derives the class from alphaMask and
// twoSided; an alphaMask material with Opaque or TwoSided is drawn in AlphaMasked or AlphaMaskedTwoSided (not in the
// depth pre-pass). PrototypeFoliage (S19a, transitional until S43): the prototype trees' shading (wind, the foliage
// texture array with alpha to coverage, LOD tint); its color word is packPrototypeFoliageColor (PrototypeWorld.h) and
// it needs the foliage texture of RenderScene::setFoliageTexture (asserted at addMaterial / setMaterial)
enum class PipelineClass : uint8_t
{
    Derived             = 0,
    Opaque              = 1,
    AlphaMasked         = 2,
    TwoSided            = 3,
    AlphaMaskedTwoSided = 4,
    PrototypeFoliage    = 5,
};
constexpr int pipelineClassCount = 5;

// The material data model (§10.8; textures, normal maps and wind in S43)
struct MaterialData
{
    ShadingModel model = ShadingModel::FlatColor;
    // The vertex color word is a palette index (else sRGB8 RGB + linear alpha, already containing any base color factor
    // of an import)
    bool usePalette = false;
    // Multiplied with the vertex color (1 for imported meshes)
    vec4 baseColor  = vec4(1, 1, 1, 1);
    float roughness = 0.8f, metalness = 0;
    vec3 emissive = vec3(0, 0, 0);
    // alphaMask: discarded where vertex alpha * baseColor.a < 0.5; twoSided: no back face culling
    bool alphaMask = false, twoSided = false;
    // PipelineClass (0: derived from alphaMask / twoSided; the class also selects how the vertex color word is
    // interpreted: PrototypeFoliage packs the texture layer, AO and the wind weight)
    uint8_t pipelineClass = 0;
};

// One entry of the game's palette (RenderScene::setPalette, at most 4096)
struct PaletteEntry
{
    ucvec4 color      = ucvec4(255, 255, 255, 255);  // sRGB8 + linear alpha
    uint8_t roughness = 204, metalness = 0;          // unorm8
};

// A static instance (32 bytes), region-relative (the set's RegionId); the GLSL struct is
// { float px, py, pz; uint rot01, rot23; float scale; uint meshFlags; uint key; }
struct StaticInstance
{
    float position[3] = {0, 0, 0};
    // The rotation (x, y, z, w) as snorm16 pairs (setRotation): rot01 = (x, y), rot23 = (z, w)
    uint32_t rot01 = 0, rot23 = 0x7FFF0000u;
    // 0: a tombstone (removeInstance)
    float scale    = 1;
    MeshId mesh    = 0;
    uint16_t flags = 0;  // reserved, 0
    // InstanceKey: generator slot in the upper 8 bits, counter in the lower 24
    uint32_t key = 0;

    void setPosition(const vec3& p)
    {
        position[0] = p(0);
        position[1] = p(1);
        position[2] = p(2);
    }
    vec3 getPosition() const { return vec3(position[0], position[1], position[2]); }
    // Encodes the (unit) quaternion
    SAIGA_VULKAN_API void setRotation(const quat& q);
    // The decoded rotation as the shaders see it (unpackSnorm2x16 + normalize)
    SAIGA_VULKAN_API quat getRotation() const;
};
static_assert(sizeof(StaticInstance) == 32, "StaticInstance is 32 bytes");
static_assert(offsetof(StaticInstance, rot01) == 12 && offsetof(StaticInstance, scale) == 20 &&
                  offsetof(StaticInstance, mesh) == 24 && offsetof(StaticInstance, flags) == 26 &&
                  offsetof(StaticInstance, key) == 28,
              "StaticInstance matches the std430 layout of the shaders");

// The pose of an instance: position, rotation, uniform scale
template <typename T>
struct InstancePose
{
    Vector<T, 3> position  = Vector<T, 3>::Zero();
    Quaternion<T> rotation = Quaternion<T>::Identity();
    T scale                = 1;
};

// The pose of a dynamic instance, relative to its chunk's origin
using DynamicPose = InstancePose<double>;

// The pose of a dynamic instance at alpha between prev (0) and curr (1) as the shaders compute it (dynamicPose and
// interpolateRotation of shader/vulkan/scene/instance_common.glsl): position and scale mixed linearly; the rotations
// normalized, curr negated if their dot product is < 0 (the shorter arc), slerp, nlerp (normalized mix) if the dot
// product is > 0.9995. The CPU reference of the GPU interpolation: in float as the shaders (tests), in double for the
// extract (S19b).
template <typename T>
InstancePose<T> interpolatePose(const InstancePose<T>& prev, const InstancePose<T>& curr, T alpha)
{
    using std::acos;
    using std::sin;
    InstancePose<T> r;
    r.position            = prev.position * (T(1) - alpha) + curr.position * alpha;
    r.scale               = prev.scale * (T(1) - alpha) + curr.scale * alpha;
    const Quaternion<T> a = prev.rotation.normalized();
    Quaternion<T> b       = curr.rotation.normalized();
    T d                   = a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
    if (d < T(0))
    {
        b = Quaternion<T>(-b.w(), -b.x(), -b.y(), -b.z());
        d = -d;
    }
    T wa = T(1) - alpha, wb = alpha;
    if (!(d > T(0.9995)))
    {
        const T theta = acos(d);
        const T s     = sin(theta);
        wa            = sin((T(1) - alpha) * theta) / s;
        wb            = sin(alpha * theta) / s;
    }
    r.rotation = Quaternion<T>(wa * a.w() + wb * b.w(), wa * a.x() + wb * b.x(), wa * a.y() + wb * b.y(),
                               wa * a.z() + wb * b.z())
                     .normalized();
    return r;
}

struct InstanceSetHandle
{
    uint32_t id = 0;  // 0: invalid
    explicit operator bool() const { return id != 0; }
};
struct InstanceHandle
{
    uint32_t id = 0;  // 0: invalid
    explicit operator bool() const { return id != 0; }
};

// ---- local lights (S36b, docs/engine_design.md §10.7; RenderScene::addLight, RenderSink::addLight)

// Count: visit's EnumNames convention (the engine's LightSource visits it)
enum class LightKind : uint8_t
{
    Point,
    Spot,
    Count
};
inline const char* lightKindName(LightKind k)
{
    return k == LightKind::Spot ? "spot" : "point";
}
// A render-only flicker of the intensity (§10.7: presentation): 1 + amplitude * a sum of sines of the environment's
// flicker phase (EnvironmentState::phases[6], computed in double; no absolute time in a shader) at about frequency Hz,
// shifted by seed (one light's flicker differs from the next one's)
struct LightFlicker
{
    float amplitude = 0;  // [0, 1]
    float frequency = 0;  // Hz
    uint32_t seed   = 0;
    bool operator==(const LightFlicker& o) const
    {
        return amplitude == o.amplitude && frequency == o.frequency && seed == o.seed;
    }
};
// A light of the scene: positions relative to its chunk's origin (float render data, R-MATH-1)
struct LightData
{
    LightKind kind   = LightKind::Point;
    vec3 position    = vec3(0, 0, 0);           // relative to the chunk's origin
    vec3 direction   = vec3(0, -1, 0);          // the spot's axis (normalized; unused for points)
    vec3 color       = vec3(1.0f, 0.9f, 0.8f);  // linear
    float lumen      = 800;                     // luminous flux (point: lumen / 4 pi cd, spot: lumen / pi cd)
    float range      = 10;                      // m, (0, 1000]; the light is 0 beyond it
    float cosInner   = 0.9063078f;              // cos(25 degrees)
    float cosOuter   = 0.8191520f;              // cos(35 degrees)
    bool castShadows = false;
    float priority   = 1;  // scales the importance (the budget order)
    LightFlicker flicker;
};
struct LightHandle
{
    uint32_t id = 0;  // 0: invalid
    explicit operator bool() const { return id != 0; }
    bool operator==(const LightHandle& o) const { return id == o.id; }
};
// The bulk path of many lights of one chunk (S30's piece lights): one list, no entities
struct LightListHandle
{
    uint32_t id = 0;  // 0: invalid
    explicit operator bool() const { return id != 0; }
};

// The statistics of a completed frame (RenderScene::statistics)
// The render settings of a space (S21, docs/engine_design.md §10.1, §12.4; RenderScene::setSpaceSettings, forwarded by
// the engine's World per SpaceId): an open-world space draws the terrain, grass, water and sky of the camera's space;
// an enclosed one none of them, clears to background (linear RGB) and lights the instances with kEnclosedLighting. A
// space without settings is drawn as open world. S36's environment per space replaces it.
struct SpaceRenderSettings
{
    bool openWorld  = true;
    vec3 background = vec3(0.02f, 0.02f, 0.025f);
};

// The environment of a frame (docs/engine_design.md §10.1, §10.7; S36): sun, moon, stars, weather, exposure and
// ambient of the view camera's space, extracted once per rendered frame by the engine (environmentState) and given to
// RenderScene::setEnvironment. Render data: float, directions unit vectors (y up, north = -z)
struct EnvironmentState
{
    vec3 sunDir               = vec3(0, 0.70710678f, 0.70710678f);  // towards the sun (noon at 45 degrees latitude)
    float sunIlluminance      = 120000;  // lux at the top of the atmosphere (the LUT attenuates)
    vec3 moonDir              = vec3(0, -0.70710678f, -0.70710678f);
    float moonIlluminance     = 0.27f;  // lux, scaled by the lit fraction of the phase
    float moonFullIlluminance = 0.27f;  // lux of the full moon (the disc's radiance; its phase is the geometry)
    float moonPhase           = 0.5f;
    float starIntensity       = 1;
    mat3 starRotation         = mat3::Identity();  // the hour angle about the celestial pole
    // Weather (§10.7)
    float cloudiness = 0.3f, rain = 0, fogDensity = 0.00015f, snowCover = 0;
    vec2 wind        = vec2(3, 1);
    vec2 cloudOffset = vec2(0, 0);  // already wrapped to the cloud tile (float exact below 2^24)
    // Exposure (A3): the resolved EV100 of this frame (curve or fixed value, compensation applied) and its source
    float ev100                   = 15;
    ExposureSource exposureSource = ExposureSource::Fixed;
    // Ambient (A3)
    float ambientScale = 1;
    vec3 minAmbient    = vec3(0.002f, 0.002f, 0.002f);  // lux-equivalent floor (night policy)
    // An enclosed space: its template's ambient (lux-equivalent colour) and EV100, no sun, sky or sky ambient
    bool enclosed        = false;
    vec3 enclosedAmbient = vec3(0, 0, 0);
    // Phases (radians, fmod in double on the CPU; no absolute time in a shader, §10.1): the trees' wind (0-2), the
    // grass's (3-4), the cloud drift remainder (5), the flicker base (6)
    std::array<float, 8> phases = {};
    // The presentation settings the renderer reads (S36a1c, from EnvironmentSettings): the cloud layer (altitude,
    // radius, seed, shadow fade, fog falloff) and the atmosphere (scales, ground albedo)
    CloudSettings clouds;
    AtmosphereSettings atmosphere;
};

struct RenderStatistics
{
    uint64_t frame = 0;  // RenderScene's frame number (0: none yet)
    // Drawn instances per LOD (static and dynamic)
    std::array<uint32_t, 4> instancesPerLod{};
    uint32_t terrainNodes = 0;  // the terrain's selected nodes incl. quarters (S19a: the prototype's; S20: TerrainPass)
    // The indirect draw calls issued in the scene pass (one per pipeline class and view, or one per bucket without
    // multiDrawIndirect); the depth pre-pass repeats those of the opaque classes (S19a), not counted
    uint32_t drawCommands = 0;
    // Buckets (view, pipeline class, mesh, LOD) with at least one instance
    uint32_t nonEmptyBuckets = 0;
    // The drawn instances per LOD of the tracked meshes (RenderScene::trackMeshStatistics), in the order of the list
    // that this frame used (0 for an unregistered mesh and beyond its LOD count)
    struct TrackedMesh
    {
        MeshId mesh = 0;
        std::array<uint32_t, 4> instancesPerLod{};
    };
    std::vector<TrackedMesh> trackedMeshes;
    // S36a2: the instances and triangles drawn into each shadow cascade (views 1..4 of the culling; 0 without
    // shadows). instancesPerLod and the tracked meshes stay the main view's
    std::array<uint32_t, 4> cascadeInstances{};
    std::array<uint32_t, 4> cascadeTriangles{};
};

// The views of SceneCulling (S36a2, docs/engine_design.md §10.2): view 0 the main camera, views 1..kMaxCascades the
// shadow cascades; the work chunks' 16-bit view masks leave room for part b's atlas faces
constexpr int kMaxCascades  = 4;
constexpr int kMaxCullViews = 16;

// A shadow view of the culling and the shadow passes (S36a2, A5 of docs/stages/S36_lighting.plan.md): a cascade's
// orthographic light-space volume, relative to the camera. Light space: x = axisX, y = axisY, z = axisZ (towards the
// light); a camera-relative point p has the coordinates (dot(axisX, p), dot(axisY, p), dot(axisZ, p)). The volume is
// the xy rectangle times the depth range shared by every cascade (D23): from the receivers' far end zMin to their near
// end zMax, extended towards the light by the casters' caps (casters nearer than zMax are pancaked onto depth 0)
struct ShadowView
{
    Vec3 axisX = Vec3(1, 0, 0), axisY = Vec3(0, 1, 0), axisZ = Vec3(0, 0, 1);
    Vec2 rectMin = Vec2::Zero(), rectMax = Vec2::Zero();
    double zMin = 0, zMax = 0;
    // The caps towards the light beyond zMax: instances (maxCasterHeight / sin(max(elevation, minimum))), terrain
    // (cascadeCasterDistance horizontally)
    double instanceCap = 0, terrainCap = 0;
    // De-duplication: a caster whose bounding sphere's xy footprint lies inside this rectangle and whose light depth
    // lies inside [dedupZMin, dedupZMax] (a part of the previous cascade's caster volume) is not drawn into this view
    // (its whole shadow falls into the previous cascade, which contains it)
    bool dedup    = false;
    Vec2 dedupMin = Vec2::Zero(), dedupMax = Vec2::Zero();
    double dedupZMin = 0, dedupZMax = 0;
    // The terrain's de-duplication interval ends at the previous cascade's terrain cap (<= dedupZMax; S36a2 review,
    // finding 8: the terrain nodes are de-duplicated like the instances)
    double dedupZMaxTerrain = 0;
    // Casters with a (scaled) bounding radius below this are skipped (LightingParameters::smallCasterTexels texels)
    double smallRadius = 0;
    // LOD levels added to the main view's LOD (clamped at the mesh's last LOD; D10: 0 for cascades 0-1)
    int lodBias = 0;
    // The camera-relative light view-projection: depth 0 at zMax, 1 at zMin (not reversed), xy over the rectangle
    mat4 viewProj = mat4::Identity();
    // S36b: a face of the shadow atlas (a perspective light view): its planes (camera-relative, inward) replace the
    // volume's; the casters it draws (the static layer: static sets and terrain, the dynamic layer: dynamics)
    bool perspective               = false;
    std::array<Vec4, 6> facePlanes = {};
    bool castStatic = true, castDynamic = true;

    // The volume's planes (inward normals, camera-relative: dot(n, p) + w >= -r is inside): the 4 sides, the far cap
    // (zMin) and the near cap (zMax + the instances' or the terrain's cap)
    // The light-space footprint of the camera-relative AABB [lo, hi] lies inside the de-duplication rectangle and its
    // light depth inside the de-duplication interval (of the instances, or of the terrain)
    bool dedupContains(const Vec3& lo, const Vec3& hi, bool terrain) const
    {
        if (!dedup) return false;
        double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300, z0 = 1e300, z1 = -1e300;
        for (int c = 0; c < 8; ++c)
        {
            const Vec3 p((c & 1) ? hi(0) : lo(0), (c & 2) ? hi(1) : lo(1), (c & 4) ? hi(2) : lo(2));
            const double x = axisX.dot(p), y = axisY.dot(p), z = axisZ.dot(p);
            x0 = std::min(x0, x);
            x1 = std::max(x1, x);
            y0 = std::min(y0, y);
            y1 = std::max(y1, y);
            z0 = std::min(z0, z);
            z1 = std::max(z1, z);
        }
        return x0 >= dedupMin(0) && x1 <= dedupMax(0) && y0 >= dedupMin(1) && y1 <= dedupMax(1) && z0 >= dedupZMin &&
               z1 <= (terrain ? dedupZMaxTerrain : dedupZMax);
    }
    std::array<Vec4, 6> planes(bool terrain) const
    {
        if (perspective) return facePlanes;
        const double cap = zMax + (terrain ? terrainCap : instanceCap);
        auto plane       = [](const Vec3& n, double w) { return Vec4(n(0), n(1), n(2), w); };
        return {plane(axisX, -rectMin(0)), plane(-axisX, rectMax(0)), plane(axisY, -rectMin(1)),
                plane(-axisY, rectMax(1)), plane(axisZ, -zMin),       plane(-axisZ, cap)};
    }
};

// The maximum length of RenderScene::trackMeshStatistics's list: the scan writes 4 counts per tracked mesh into the
// frame's readback buffer (one invocation each); enough for a game's species or vehicle meshes (S19a: the tree species)
constexpr int maxTrackedMeshes = 16;

// Encoders of the vertex words (the shaders decode them)
// Octahedral normal encoding as two snorm16 (x in the low 16 bits)
SAIGA_VULKAN_API uint32_t packNormalOct(const vec3& n);
SAIGA_VULKAN_API vec3 unpackNormalOct(uint32_t v);
// sRGB8 RGB (already encoded) + alpha (linear) in the bytes r, g, b, a (r in the low byte)
inline uint32_t packColor(const ucvec4& c)
{
    return uint32_t(c(0)) | (uint32_t(c(1)) << 8) | (uint32_t(c(2)) << 16) | (uint32_t(c(3)) << 24);
}
// GLSL's packSnorm2x16 / unpackSnorm2x16
SAIGA_VULKAN_API uint32_t packSnorm2x16(float a, float b);
SAIGA_VULKAN_API vec2 unpackSnorm2x16(uint32_t v);

// The GPU times of one completed frame (timestamps), matched to the frame by its number
struct FrameTimes
{
    uint64_t frame = 0;                // RenderScene's frame number (1, 2, ...)
    double gpuMs   = 0;                // the whole frame
    std::vector<double> gpuSectionMs;  // per section of RenderScene::gpuSectionNames
};

}  // namespace Vulkan
}  // namespace Saiga
