#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "heightmap.glsl"

// Grid vertex (integer coordinates 0..gridN)
layout(location = 0) in vec2 inGrid;
// x, z: origin, size, level (-1: water plane)
layout(location = 1) in vec4 inNode;

layout(push_constant) uniform PushConstants
{
    float gridN;
    float pad0;
    float pad1;
    int wire;
}
pc;

layout(location = 0) out vec3 outPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) flat out float outLod;

void main()
{
    float lod = inNode.w;
    outLod    = lod;
    if (lod < 0.0)
    {
        vec2 p      = inNode.xy + inGrid / pc.gridN * inNode.z;
        vec3 pos    = vec3(p.x, world.fog.w, p.y);
        outPos      = pos;
        outNormal   = vec3(0, 1, 0);
        gl_Position = world.viewProj * vec4(pos, 1.0);
        return;
    }

    float spacing = inNode.z / pc.gridN;
    vec2 p        = inNode.xy + inGrid * spacing;
    float h       = heightAt(p);

    // CDLOD morph: towards the grid of the next level when approaching the end of this level's range
    float k = 0.0;
    if (lod < world.terrain2.x)
    {
        float range = world.terrain.z * exp2(lod);
        float prev  = lod > 0.0 ? range * 0.5 : 0.0;
        float start = prev + (range - prev) * world.terrain.w;
        float dist  = distance(vec3(p.x, h, p.y), world.lodCameraPos.xyz);
        k           = clamp((dist - start) / (range - start), 0.0, 1.0);
    }
    // Odd vertices move onto their even neighbour
    p -= mod(inGrid, 2.0) * spacing * k;
    h = heightAt(p);

    vec3 pos    = vec3(p.x, h, p.y);
    outPos      = pos;
    outNormal   = normalAt(p, max(spacing, world.terrain.y));
    gl_Position = world.viewProj * vec4(pos, 1.0);
}
