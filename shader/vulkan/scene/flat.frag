#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The flat-color shading of RenderScene's instances (S18, docs/engine_design.md §10.8): the vertex color word (sRGB8
// RGB + linear alpha) or the palette entry it indexes, times the material's base color; SmoothTextured without textures
// renders its base color (textures in S43). A fixed sun and ambient (lighting in S36) plus the emissive color. The
// alpha test of an alphaMask material: alpha * baseColor.a < 0.5 is discarded. Output (S17's contract): linear values
// times the pre-exposure, or in clay mode display values (clamped, sRGB encoded here).

#include "color.glsl"

struct Material
{
    vec4 baseColor;
    vec4 emissiveRoughness;
    uint model;
    uint flags;
    float metalness;
    uint pad;
};
const uint materialUsePalette = 1u;
const uint materialAlphaMask  = 2u;
const uint materialTwoSided   = 4u;

layout(std430, set = 0, binding = 11) readonly buffer Materials
{
    Material materials[];
};
// Per entry: x: sRGB8 RGB + linear alpha, y: roughness | metalness << 8 (unorm8)
layout(std430, set = 0, binding = 12) readonly buffer Palette
{
    uvec2 palette[];
};

layout(push_constant) uniform Push
{
    mat4 viewProj;
    float alpha;
    float preExposure;
    uint linear;
    uint pad;
}
pc;

layout(location = 0) flat in uint inMaterial;
layout(location = 1) flat in uint inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inUv;

layout(location = 0) out vec4 outColor;

// The fixed light of S18 (S36 replaces it)
const vec3 sunDirection = vec3(0.3113996, 0.7784989, 0.5449493);  // normalize(0.4, 1, 0.7), towards the sun
const vec3 sunColor     = vec3(0.8);
const vec3 ambientColor = vec3(0.35);

void main()
{
    Material m = materials[inMaterial];
    uint word = inColor;
    if ((m.flags & materialUsePalette) != 0u) word = palette[min(inColor, uint(palette.length()) - 1u)].x;
    vec4 v = unpackUnorm4x8(word);
    vec4 c;
    if (m.model == 1u)
        c = vec4(m.baseColor.rgb, v.a * m.baseColor.a);  // the vertex alpha for the alpha test
    else
        c = vec4(srgbToLinear(v.rgb), v.a) * m.baseColor;
    if ((m.flags & materialAlphaMask) != 0u && c.a < 0.5) discard;

    vec3 n = normalize(inNormal);
    if (!gl_FrontFacing) n = -n;
    vec3 lit = c.rgb * (ambientColor + sunColor * max(dot(n, sunDirection), 0.0)) + m.emissiveRoughness.xyz;
    if (pc.linear != 0u)
        outColor = vec4(max(lit, vec3(0.0)) * pc.preExposure, 1.0);
    else
        outColor = vec4(linearToSrgb(clamp(lit, vec3(0.0), vec3(1.0))), 1.0);
}
