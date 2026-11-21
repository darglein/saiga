#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The PrototypeFoliage pipeline class of RenderScene (S19a, transitional until S43): the prototype trees' tree.vert on
// the instance layout. The instance is placed as in instance.vert; the color word packs the foliage texture layer (bits
// 0-7, 0xFF: opaque parts), the AO (bits 8-15) and the wind weight (bits 16-23) (packPrototypeFoliageColor). Wind: the
// whole tree sways, the leaves flutter, with the phases of the lighting block (fmod(time * f, 2 pi) in double on the
// CPU, no absolute time) and a per-tree phase from its world xz. Transitional exception to the camera-relative rule
// (S20/S23 replace the prototype): the world xz is rebuilt in float as position + (hi + lo) + camera.xz (valid within
// the prototype's 8 km).

#include "instance_common.glsl"
#include "scene_lighting.glsl"

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

// Position relative to the camera
layout(location = 0) out vec3 outPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outUvLayer;
layout(location = 3) out float outAo;
layout(location = 4) flat out float outLod;

void main()
{
    uvec2 e = compacted[gl_InstanceIndex];
    Pose pose;
    if ((e.x & dynamicBit) != 0u)
        pose = dynamicPose(dynamics[e.x & ~dynamicBit], pc.alpha);
    else
        pose = staticPose(instances[e.x]);
    OffsetEntry o = offsets[e.y & 0xFFFFFFu];
    outLod        = float(e.y >> 24);

    uint layerBits = inColor & 0xFFu;
    float layer    = layerBits == 0xFFu ? -1.0 : float(layerBits);
    float ao       = float((inColor >> 8) & 0xFFu) / 255.0;
    float windW    = float((inColor >> 16) & 0xFFu) / 255.0;

    float scale = pose.scale;
    vec3 p      = rotateQuat(pose.rotation, inPosition * scale);
    vec3 n      = rotateQuat(pose.rotation, decodeOct(inNormal));

    // Wind: the whole tree sways, the leaves flutter (the prototype's tree.vert with time * f replaced by the phases)
    vec2 worldXz = pose.position.xz + (o.hi.xz + o.lo.xz) + lighting.camera.xz;
    float phase  = worldXz.x * 0.031 + worldXz.y * 0.027;
    vec2 sway    = vec2(sin(lighting.wind.x + phase), cos(lighting.wind.y + phase * 1.7)) * 0.012;
    float flutter =
        layer >= 0.0 ? sin(lighting.wind.z + phase * 5.0 + inPosition.x * 20.0 + inPosition.y * 13.0) * 0.004 : 0.0;
    p.xz += (sway + flutter) * scale * windW * lighting.wind.w;

    vec3 pos    = placePoint(pose.position, o, p);
    outPos      = pos;
    outNormal   = n;
    outUvLayer  = vec3(inUv, layer);
    outAo       = ao;
    gl_Position = pc.viewProj * vec4(pos, 1.0);
}
