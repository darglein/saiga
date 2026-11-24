#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The PrototypeFoliage class in the cascades (S36a2, A5): prototype_foliage.vert's placement and wind (the frame's
// phases of the lighting block; the prototype's world-xz exception of §10.1, valid within 8 km) with the cascade's
// camera-relative light view-projection and pancaking (shadow_depth.vert). Includes scene_block.glsl alone.

#include "../instance_common.glsl"
#include "scene_block.glsl"

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
    float pad0, pad1, pad2;
}
pc;

layout(location = 0) out vec3 outUvLayer;

void main()
{
    uvec2 e = compacted[gl_InstanceIndex];
    Pose pose;
    if ((e.x & dynamicBit) != 0u)
        pose = dynamicPose(dynamics[e.x & ~dynamicBit], pc.alpha);
    else
        pose = staticPose(instances[e.x]);
    OffsetEntry o = offsets[e.y & 0xFFFFFFu];

    uint layerBits = inColor & 0xFFu;
    float layer    = layerBits == 0xFFu ? -1.0 : float(layerBits);
    float windW    = float((inColor >> 16) & 0xFFu) / 255.0;
    float scale    = pose.scale;
    vec3 p         = rotateQuat(pose.rotation, inPosition * scale);

    // The wind of prototype_foliage.vert (keep both in sync)
    vec2 worldXz = pose.position.xz + (o.hi.xz + o.lo.xz) + lighting.camera.xz;
    float phase  = worldXz.x * 0.031 + worldXz.y * 0.027;
    vec2 sway    = vec2(sin(lighting.wind.x + phase), cos(lighting.wind.y + phase * 1.7)) * 0.012;
    float flutter =
        layer >= 0.0 ? sin(lighting.wind.z + phase * 5.0 + inPosition.x * 20.0 + inPosition.y * 13.0) * 0.004 : 0.0;
    p.xz += (sway + flutter) * scale * windW * lighting.wind.w;

    outUvLayer    = vec3(inUv, layer);
    gl_Position   = pc.viewProj * vec4(placePoint(pose.position, o, p), 1.0);
    gl_Position.z = max(gl_Position.z, 0.0);
}
