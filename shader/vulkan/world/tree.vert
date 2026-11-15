#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

struct TreeInstance
{
    // xyz: stem base, w: rotation around y
    vec4 posYaw;
    // x: height, y: species
    vec4 scaleSpecies;
};

layout(std430, set = 0, binding = 1) readonly buffer Instances
{
    TreeInstance trees[];
};

// Written by tree_cull.comp: instance index | (lod << 30), one region per draw command
layout(std430, set = 0, binding = 2) readonly buffer Visible
{
    uint visible[];
};

layout(location = 0) in vec4 inPosAo;
layout(location = 1) in vec4 inNormalWind;
layout(location = 2) in vec4 inUvLayer;

layout(location = 0) out vec3 outPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outUvLayer;
layout(location = 3) out float outAo;
layout(location = 4) flat out float outLod;

vec3 rotateY(vec3 v, float c, float s)
{
    return vec3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

void main()
{
    uint v         = visible[gl_InstanceIndex];
    TreeInstance t = trees[v & 0x3FFFFFFFu];
    outLod         = float(v >> 30);

    float scale = t.scaleSpecies.x;
    float c = cos(t.posYaw.w), s = sin(t.posYaw.w);
    vec3 p = rotateY(inPosAo.xyz * scale, c, s);
    vec3 n = rotateY(inNormalWind.xyz, c, s);

    // Wind: the whole tree sways, the leaves flutter
    float time  = world.cameraPos.w;
    float phase = t.posYaw.x * 0.031 + t.posYaw.z * 0.027;
    vec2 sway   = vec2(sin(time * 1.1 + phase), cos(time * 0.9 + phase * 1.7)) * 0.012;
    float flutter =
        inUvLayer.z >= 0.0 ? sin(time * 4.0 + phase * 5.0 + inPosAo.x * 20.0 + inPosAo.y * 13.0) * 0.004 : 0.0;
    p.xz += (sway + flutter) * scale * inNormalWind.w * world.terrain2.z;

    // Relative to the camera: the large instance position and the camera are subtracted first
    vec3 pos    = (t.posYaw.xyz - world.cameraPos.xyz) + p;
    outPos      = pos;
    outNormal   = n;
    outUvLayer  = inUvLayer.xyz;
    outAo       = inPosAo.w;
    gl_Position = world.viewProjRel * vec4(pos, 1.0);
}
