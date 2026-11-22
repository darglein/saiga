/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The scene content of RenderScene (S18, docs/engine_design.md §10.1, §10.2): the buffers shared by the culling passes
// (instance_count.comp, bucket_scan.comp, instance_write.comp) and the instance draws (instance.vert, flat.frag), one
// descriptor set with every binding (vulkan/scene/SceneCulling.cpp), and the placement functions that the count pass
// and the vertex shader share.

// A work chunk of the count and write passes: up to 256 instances [first, first + count) of the instance buffer with
// their offset slot, or of the dynamics (slot = dynamicChunk: each dynamic carries its chunk's slot); viewMask (S36a2):
// bits 0..15, the views the chunk is culled for (0: the main camera, 1..: the shadow cascades)
struct WorkChunk
{
    uint first;
    uint count;
    uint slot;
    uint viewMask;
};

// A view of the culling (S36a2; SceneCulling.cpp GpuCullView): six planes (camera-relative, inward: dot(n, p) + w >=
// -radius is inside), the light-space x and y axes of a shadow view, the de-duplication rectangle (xy min, zw max) and
// params (x: 1 de-duplication on, y: the small-caster radius, z: the LOD bias, w: 1 a shadow view)
struct CullView
{
    vec4 planes[6];
    vec4 axisX;
    vec4 axisY;
    vec4 dedup;
    vec4 params;
};
const uint maxCullViews = 16u;
const uint dynamicChunk = 0xFFFFFFFFu;

// Saiga::Vulkan::StaticInstance (32 bytes; region-relative position, quaternion as snorm16 pairs)
struct StaticInstance
{
    float px, py, pz;
    uint rot01, rot23;
    float scale;
    uint meshFlags;
    uint key;
};

// A dynamic instance (80 bytes): prev and curr pose relative to its chunk, scale in w
struct DynamicInstance
{
    vec4 prevPos;
    vec4 currPos;
    vec4 prevRot;
    vec4 currRot;
    uint mesh;
    uint slot;
    uint boneOffset;
    uint pad;
};

struct MeshLodRecord
{
    uint firstIndex;
    uint indexCount;
    int vertexOffset;
    float endDistance;
};

// The GPU record of a mesh (112 bytes); a zeroed record (lodCount 0: mesh 0, unregistered IDs) is rejected
struct MeshRecord
{
    uint material;
    uint lodCount;
    uint bucketBase;
    uint pipelineClass;
    vec4 sphere;   // xyz centre, w radius (mesh space)
    vec4 padding;  // x: boundsPadding
    MeshLodRecord lods[4];
};

// Saiga::Vulkan::MaterialData (48 bytes)
struct Material
{
    vec4 baseColor;
    vec4 emissiveRoughness;
    uint model;  // 0: flat color, 1: smooth textured
    uint flags;  // materialUsePalette | materialAlphaMask | materialTwoSided
    float metalness;
    uint pad;
};
const uint materialUsePalette = 1u;
const uint materialAlphaMask  = 2u;
const uint materialTwoSided   = 4u;

// The offset of a region or chunk: origin - cameraPosition (double on the CPU) split into float hi + lo; hi.w = 1 valid
struct OffsetEntry
{
    vec4 hi;
    vec4 lo;
};

layout(std430, set = 0, binding = 0) readonly buffer WorkChunks
{
    WorkChunk chunks[];
};
layout(std430, set = 0, binding = 1) readonly buffer Instances
{
    StaticInstance instances[];
};
layout(std430, set = 0, binding = 2) readonly buffer Dynamics
{
    DynamicInstance dynamics[];
};
layout(std430, set = 0, binding = 3) readonly buffer Meshes
{
    MeshRecord meshes[];
};
layout(std430, set = 0, binding = 4) readonly buffer Offsets
{
    OffsetEntry offsets[];
};
// Binding 5: the bucket counts, 6: the scratch entries, 7: the indirect commands, 8: the LOD of each bucket, 9: the
// statistics, 10: the compacted entries, 11: the materials, 12: the palette, 13: the views (uniform), 14: the work
// chunks' scratch bases (declared by the shaders that use them)

// The scratch entry of an instance that is not visible
const uint notVisible = 0xFFFFFFFFu;
// The high bit of a compacted entry's instance index: a dynamic instance
const uint dynamicBit = 0x80000000u;

// Octahedral decoding (Saiga::Vulkan::packNormalOct)
vec3 decodeOct(uint v)
{
    vec2 e = unpackSnorm2x16(v);
    vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

// Rotates v by the unit quaternion q (x, y, z, w)
vec3 rotateQuat(vec4 q, vec3 v)
{
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// The quaternion of a static instance (unpackSnorm2x16 + normalize)
vec4 instanceRotation(StaticInstance s)
{
    return normalize(vec4(unpackSnorm2x16(s.rot01), unpackSnorm2x16(s.rot23)));
}

// The interpolation of the dynamic rotations: the shorter arc (curr negated if dot < 0), slerp, nlerp near dot = 1. The
// CPU reference is interpolatePose of src/saiga/vulkan/scene/RenderSceneTypes.h (keep both in sync).
vec4 interpolateRotation(vec4 a, vec4 b, float t)
{
    float d = dot(a, b);
    if (d < 0.0)
    {
        b = -b;
        d = -d;
    }
    if (d > 0.9995) return normalize(mix(a, b, t));
    float theta = acos(d);
    float s     = sin(theta);
    return normalize((sin((1.0 - t) * theta) / s) * a + (sin(t * theta) / s) * b);
}

struct Pose
{
    vec3 position;
    vec4 rotation;
    float scale;
};

Pose staticPose(StaticInstance s)
{
    Pose p;
    p.position = vec3(s.px, s.py, s.pz);
    p.rotation = instanceRotation(s);
    p.scale    = s.scale;
    return p;
}

// The pose of a dynamic instance at alpha between prev (0) and curr (1)
Pose dynamicPose(DynamicInstance d, float alpha)
{
    Pose p;
    p.position = mix(d.prevPos.xyz, d.currPos.xyz, alpha);
    p.scale    = mix(d.prevPos.w, d.currPos.w, alpha);
    p.rotation = interpolateRotation(d.prevRot, d.currRot, alpha);
    return p;
}

// A point relative to the camera: ((position + hi) + lo) + local, without reassociation (precise): position and hi
// nearly cancel, so the sum keeps the precision of the local position (§9.1)
vec3 placePoint(vec3 position, OffsetEntry o, vec3 local)
{
    precise vec3 a = position + o.hi.xyz;
    precise vec3 b = a + o.lo.xyz;
    precise vec3 r = b + local;
    return r;
}
