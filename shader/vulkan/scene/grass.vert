#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The grass of S20 (GrassPass): one instanced draw per 16 m block of a chunk, the instance is the cell in the block.
// Everything comes from the chunk-local cell index and position (no absolute world coordinates): jitter, size and yaw
// hash the cell, the patches are value noise on an 8 m lattice wrapped per chunk, the wind phase uses integer wave
// numbers per 256 m. The height is the surface rule over the cell's base cell of the chunk's level-0 tile, the density
// the chunk's grass mask.

#include "terrain_common.glsl"
#include "terrain_heights.glsl"

layout(location = 0) in vec4 inPosAo;
layout(location = 1) in vec4 inNormalWind;
layout(location = 2) in vec4 inUvLayer;

layout(push_constant) uniform PushConstants
{
    ivec2 origin;  // the block origin relative to the anchor (m)
    ivec2 block;   // the chunk-local block index
    ivec2 chunk;   // the chunk (its level-0 tile)
    int maskLayer;
    int cells;  // cells per chunk side
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

// Value noise on the 8 m lattice of a chunk (32 cells), the lattice indices wrapped: continuous across chunk borders,
// periodic in 256 m. p: chunk-local position in lattice units
float latticeNoise(vec2 p)
{
    vec2 i  = floor(p);
    vec2 f  = fract(p);
    f       = f * f * (3.0 - 2.0 * f);
    ivec2 c = ivec2(i);
    float a = float(hash2(c & 31) >> 8) / 16777216.0;
    float b = float(hash2((c + ivec2(1, 0)) & 31) >> 8) / 16777216.0;
    float d = float(hash2((c + ivec2(0, 1)) & 31) >> 8) / 16777216.0;
    float e = float(hash2((c + ivec2(1, 1)) & 31) >> 8) / 16777216.0;
    return mix(mix(a, b, f.x), mix(d, e, f.x), f.y);
}

void degenerate()
{
    outPos      = vec3(0);
    outNormal   = vec3(0, 1, 0);
    outAo       = 1.0;
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}

void main()
{
    int perBlock = pc.cells / 16;
    int i        = gl_InstanceIndex;
    ivec2 local  = ivec2(i % perBlock, i / perBlock);
    // The chunk-local cell index
    ivec2 cell = pc.block * perBlock + local;
    uint seed  = hash2(cell);

    float cellSize = 256.0 / float(pc.cells);
    vec2 jitter    = vec2(rnd(seed), rnd(seed)) * cellSize;
    // Chunk-local position (m) and the offset from the block origin
    vec2 lp     = vec2(cell) * cellSize + jitter;
    vec2 offset = vec2(local) * cellSize + jitter;

    outLod     = 0.0;
    outUvLayer = inUvLayer.xyz;

    // The base cell of the level-0 tile (spacing: the base spacing) and the mask
    int bs     = frame.grid.w;
    vec2 g     = lp / float(bs);
    int m      = 256 / bs;
    ivec2 base = clamp(ivec2(floor(g)), ivec2(0), ivec2(m - 1));
    vec2 f     = clamp(g - vec2(base), vec2(0.0), vec2(1.0));
    float mask = maskDensity(pc.maskLayer, base);

    // Patches of dense and sparse grass, fading out towards the radius
    float density = frame.grass.z * mask * mix(0.25, 1.0, smoothstep(0.2, 0.7, latticeNoise(lp / 8.0)));
    float size    = rnd(seed) < density ? 1.0 : 0.0;
    // Relative to the camera: the integer block origin, then the small local offset
    vec2 rel = vec2(pc.origin) - frame.cameraRel.xz + offset;
    float d  = length(rel);
    size *= 1.0 - smoothstep(frame.grass.x * 0.6, frame.grass.x, d);
    if (size <= 0.0)
    {
        degenerate();
        return;
    }

    // The surface rule of the base cell (heights relative to refHeight, height units)
    int layer = pageLayer(0, pc.chunk);
    if (layer < 0)
    {
        degenerate();
        return;
    }
    float h00 = float(layerSample(layer, base) - frame.anchor.z);
    float h10 = float(layerSample(layer, base + ivec2(1, 0)) - frame.anchor.z);
    float h01 = float(layerSample(layer, base + ivec2(0, 1)) - frame.anchor.z);
    float h11 = float(layerSample(layer, base + ivec2(1, 1)) - frame.anchor.z);
    float dA, dB;
    if (f.x >= f.y)
    {
        dA = h10 - h00;
        dB = h11 - h10;
    }
    else
    {
        dA = h11 - h01;
        dB = h01 - h00;
    }
    float h = (h00 + dA * f.x + dB * f.y) * kHeightUnit;
    vec3 tn = normalize(vec3(-dA * kHeightUnit / float(bs), 1.0, -dB * kHeightUnit / float(bs)));

    float yaw    = rnd(seed) * 6.2831853;
    float width  = mix(0.7, 1.3, rnd(seed)) * 0.9 * size;
    float height = mix(0.35, 0.8, rnd(seed)) * size;
    vec3 lpos    = vec3(inPosAo.x * width, inPosAo.y * height, inPosAo.z * width);
    float c = cos(yaw), s = sin(yaw);
    lpos = vec3(c * lpos.x + s * lpos.z, lpos.y, -s * lpos.x + c * lpos.z);

    // Wind bends the tops (normalWind.w is 1 at the top): the CPU phases fmod(time * f, 2 pi) plus the chunk-local
    // phases 2 pi (22 lx + 17 lz) / 256 and 2 pi (15 lx + 12 lz) / 256 (integer wave numbers for both components:
    // continuous across chunk borders)
    float phase  = 6.2831853 * (22.0 * lp.x + 17.0 * lp.y) / 256.0;
    float phase2 = 6.2831853 * (15.0 * lp.x + 12.0 * lp.y) / 256.0;
    lpos.xz += vec2(sin(frame.grassPhases.x + phase), cos(frame.grassPhases.y + phase2)) * 0.08 * height *
               inNormalWind.w * frame.grassPhases.z;

    vec3 pos    = vec3(rel.x, h - 0.04 - frame.cameraRel.y, rel.y) + lpos;
    outPos      = pos;
    outNormal   = normalize(mix(tn, vec3(0, 1, 0), 0.4));
    outAo       = inPosAo.w * 0.8;
    gl_Position = frame.viewProjRel * vec4(pos, 1.0);
}
