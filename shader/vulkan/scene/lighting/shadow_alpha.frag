#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The alpha test of the alpha-masked classes in the cascades (S36a2, A5): flat.frag's rule (the vertex colour word's
// or the palette entry's alpha times the material's base alpha below 0.5 is discarded), depth only.

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

layout(std430, set = 0, binding = 11) readonly buffer Materials
{
    Material materials[];
};
layout(std430, set = 0, binding = 12) readonly buffer Palette
{
    uvec2 palette[];
};

layout(location = 0) flat in uint inMaterial;
layout(location = 1) flat in uint inColor;
layout(location = 2) in vec2 inUv;

#ifdef ATLAS_DEPTH
// S36b: the shadow atlas: the linear depth distance(light, p) / range (D16: range / 65536 per step at every distance)
layout(location = 3) in vec3 inRelPos;
layout(push_constant) uniform AtlasPush
{
    mat4 viewProj;
    vec4 pad;
    vec4 light;  // xyz the light (camera-relative), w 1 / range
}
atlasPc;
#endif

void main()
{
    Material m = materials[inMaterial];
    uint word  = inColor;
    if ((m.flags & materialUsePalette) != 0u) word = palette[min(inColor, uint(palette.length()) - 1u)].x;
    float a = unpackUnorm4x8(word).a * m.baseColor.a;
    if ((m.flags & materialAlphaMask) != 0u && a < 0.5) discard;
#ifdef ATLAS_DEPTH
    gl_FragDepth = clamp(length(inRelPos - atlasPc.light.xyz) * atlasPc.light.w, 0.0, 1.0);
#endif
}
