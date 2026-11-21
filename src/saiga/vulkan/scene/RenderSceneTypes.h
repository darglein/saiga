/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/math.h"
#include "saiga/export.h"

#include <array>
#include <cstddef>
#include <cstdint>
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
    // Fixed until auto exposure (S37); ignored (1) with Identity
    float exposure = 1;

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
// it needs the foliage texture of RenderScene::setPrototypeTerrain (asserted at addMaterial / setMaterial)
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

// The statistics of a completed frame (RenderScene::statistics)
struct RenderStatistics
{
    uint64_t frame = 0;  // RenderScene's frame number (0: none yet)
    // Drawn instances per LOD (static and dynamic)
    std::array<uint32_t, 4> instancesPerLod{};
    uint32_t terrainNodes = 0;  // the prototype terrain's selected nodes (S19a; S20: tiles)
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
