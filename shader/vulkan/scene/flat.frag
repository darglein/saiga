#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The flat-color shading of RenderScene's instances (S18, docs/engine_design.md §10.8): the vertex color word (sRGB8
// RGB + linear alpha) or the palette entry it indexes, times the material's base color; SmoothTextured without textures
// renders its base color (textures in S43). Lit by shadeSurface (S36, lighting.glsl): the material's roughness and
// metalness (or the palette entry's) and its emissive colour, the studio environment in clay mode, the physical light
// with an environment. The alpha test of an alphaMask material: alpha * baseColor.a < 0.5 is discarded. Output (S17's
// contract): linear values times the pre-exposure, or in clay mode display values (clamped, sRGB encoded here); a
// debug frame writes the view's value as it is.

#include "color.glsl"

#include "lighting/lighting.glsl"

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
layout(location = 4) in vec3 inRelPos;

layout(location = 0) out vec4 outColor;

void main()
{
    Material m = materials[inMaterial];
    uint word = inColor;
    uvec2 entry = uvec2(0u, 0u);
    if ((m.flags & materialUsePalette) != 0u)
    {
        entry = palette[min(inColor, uint(palette.length()) - 1u)];
        word  = entry.x;
    }
    vec4 v = unpackUnorm4x8(word);
    vec4 c;
    if (m.model == 1u)
        c = vec4(m.baseColor.rgb, v.a * m.baseColor.a);  // the vertex alpha for the alpha test
    else
        c = vec4(srgbToLinear(v.rgb), v.a) * m.baseColor;
    if ((m.flags & materialAlphaMask) != 0u && c.a < 0.5) discard;

    vec3 n = normalize(inNormal);
    if (!gl_FrontFacing) n = -n;
    SurfaceInput si = surfaceInput(c.rgb, n, inRelPos);
    si.roughness    = m.emissiveRoughness.w;
    si.metalness    = m.metalness;
    if ((m.flags & materialUsePalette) != 0u)
    {
        si.roughness = float(entry.y & 0xFFu) / 255.0;
        si.metalness = float((entry.y >> 8) & 0xFFu) / 255.0;
    }
    si.emissive = m.emissiveRoughness.xyz;
    vec3 lit    = shadeSurface(si);
    if (debugViewActive())
        outColor = vec4(lit, 1.0);
    else if (pc.linear != 0u)
        outColor = vec4(max(lit, vec3(0.0)) * pc.preExposure, 1.0);
    else
        outColor = vec4(linearToSrgb(clamp(lit, vec3(0.0), vec3(1.0))), 1.0);
}
