#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The instance draws of RenderScene (S18, docs/engine_design.md §10.1): compacted[gl_InstanceIndex] -> the static or
// dynamic instance -> its pose, offset slot and mesh record -> the material. The vertex is placed camera-relative with
// placePoint (the count pass places the bounding sphere with the same function); the view matrix holds only the
// camera rotation. S19a: also the depth pre-pass of the opaque classes (this shader alone): invariant gl_Position, so
// the scene pass's EQUAL test sees bit-identical depths.

#include "instance_common.glsl"

layout(std430, set = 0, binding = 10) readonly buffer Compacted
{
    uvec2 compacted[];
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in uint inNormal;
layout(location = 2) in uint inColor;
layout(location = 3) in vec2 inUv;

layout(push_constant) uniform Push
{
    mat4 viewProj;
    float alpha;
    float preExposure;
    uint linear;
    uint pad;
}
pc;

layout(location = 0) flat out uint outMaterial;
layout(location = 1) flat out uint outColor;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec2 outUv;
// The position relative to the camera (S36: the lighting's fog, cloud shadow and specular need it in every fragment)
layout(location = 4) out vec3 outRelPos;

invariant gl_Position;

void main()
{
    uvec2 e = compacted[gl_InstanceIndex];
    Pose pose;
    uint mesh;
    if ((e.x & dynamicBit) != 0u)
    {
        DynamicInstance d = dynamics[e.x & ~dynamicBit];
        pose              = dynamicPose(d, pc.alpha);
        mesh              = d.mesh;
    }
    else
    {
        StaticInstance s = instances[e.x];
        pose             = staticPose(s);
        mesh             = s.meshFlags & 0xFFFFu;
    }
    OffsetEntry o = offsets[e.y & 0xFFFFFFu];
    vec3 p        = placePoint(pose.position, o, rotateQuat(pose.rotation, inPosition * pose.scale));

    outMaterial = meshes[mesh].material;
    outColor    = inColor;
    outNormal   = rotateQuat(pose.rotation, decodeOct(inNormal));
    outUv       = inUv;
    outRelPos = p;
    gl_Position = pc.viewProj * vec4(p, 1.0);
}
