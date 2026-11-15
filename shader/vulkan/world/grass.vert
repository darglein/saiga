#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "heightmap.glsl"

layout(location = 0) in vec4 inPosAo;
layout(location = 1) in vec4 inNormalWind;
layout(location = 2) in vec4 inUvLayer;

// One draw per tile: the instance is the cell in the tile
layout(push_constant) uniform PushConstants
{
    ivec2 tileOrigin;
    int tileCells;
    int wire;
}
pc;

layout(location = 0) out vec3 outPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outUvLayer;
layout(location = 3) out float outAo;
layout(location = 4) flat out float outLod;

uint hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint hash2(ivec2 c)
{
    return hash(uint(c.x) * 0x8da6b343u ^ hash(uint(c.y) * 0xd8163841u));
}

float rnd(inout uint s)
{
    s = hash(s);
    return float(s >> 8) * (1.0 / 16777216.0);
}

float valueNoise(vec2 p)
{
    vec2 i  = floor(p);
    vec2 f  = fract(p);
    f       = f * f * (3.0 - 2.0 * f);
    ivec2 c = ivec2(i);
    float a = float(hash2(c) >> 8) / 16777216.0;
    float b = float(hash2(c + ivec2(1, 0)) >> 8) / 16777216.0;
    float d = float(hash2(c + ivec2(0, 1)) >> 8) / 16777216.0;
    float e = float(hash2(c + ivec2(1, 1)) >> 8) / 16777216.0;
    return mix(mix(a, b, f.x), mix(d, e, f.x), f.y);
}

void main()
{
    int i      = gl_InstanceIndex;
    ivec2 cell = pc.tileOrigin + ivec2(i % pc.tileCells, i / pc.tileCells);
    uint seed  = hash2(cell);

    float cellSize = world.grass.y;
    vec2 p         = (vec2(cell) + vec2(rnd(seed), rnd(seed))) * cellSize;

    // Patches of dense and sparse grass, fading out towards the radius
    float radius  = world.grass.x;
    float d       = distance(p, world.lodCameraPos.xz);
    float density = world.grass.z * mix(0.25, 1.0, smoothstep(0.2, 0.7, valueNoise(p / 9.0)));
    float size    = rnd(seed) < density ? 1.0 : 0.0;
    size *= 1.0 - smoothstep(radius * 0.6, radius, d);

    float h = heightAt(p);
    vec3 tn = normalAt(p, 1.0);
    if (tn.y < 0.82 || h < world.fog.w + 0.4) size = 0.0;

    outLod     = 0.0;
    outUvLayer = inUvLayer.xyz;
    if (size <= 0.0)
    {
        // Degenerate: no fragments
        outPos      = vec3(0);
        outNormal   = vec3(0, 1, 0);
        outAo       = 1.0;
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    float yaw    = rnd(seed) * 6.2831853;
    float width  = mix(0.7, 1.3, rnd(seed)) * 0.9 * size;
    float height = mix(0.35, 0.8, rnd(seed)) * size;
    vec3 lp      = vec3(inPosAo.x * width, inPosAo.y * height, inPosAo.z * width);
    float c = cos(yaw), s = sin(yaw);
    lp = vec3(c * lp.x + s * lp.z, lp.y, -s * lp.x + c * lp.z);

    // Wind bends the tops (normalWind.w is 1 at the top)
    float time  = world.cameraPos.w;
    float phase = p.x * 0.35 + p.y * 0.27;
    lp.xz += vec2(sin(time * 1.7 + phase), cos(time * 1.3 + phase * 0.7)) * 0.08 * height * inNormalWind.w *
             world.terrain2.z;

    vec3 pos    = vec3(p.x, h - 0.04, p.y) + lp;
    outPos      = pos;
    outNormal   = normalize(mix(tn, vec3(0, 1, 0), 0.4));
    outAo       = inPosAo.w * 0.8;
    gl_Position = world.viewProj * vec4(pos, 1.0);
}
