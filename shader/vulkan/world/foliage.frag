#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "shading.glsl"

// Trees and grass: opaque parts (layer < 0) and alpha tested cards (alpha to coverage with MSAA).

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inUvLayer;
layout(location = 3) in float inAo;
layout(location = 4) flat in float inLod;

layout(set = 0, binding = 3) uniform sampler2DArray foliage;

layout(push_constant) uniform PushConstants
{
    layout(offset = 12) int wire;
}
pc;

layout(location = 0) out vec4 outColor;

void main()
{
    float alpha = 1.0;
    if (inUvLayer.z >= 0.0)
    {
        alpha = texture(foliage, inUvLayer).r;
        // Keep the coverage in the smaller mip levels
        alpha *= 1.0 + max(textureQueryLod(foliage, inUvLayer.xy).x, 0.0) * 0.25;
        // Sharpen the edge to about one pixel (alpha to coverage)
        alpha = (alpha - world.terrain2.w) / max(fwidth(alpha), 1e-4) + 0.5;
        if (alpha <= 0.0 || (world.grass.w < 1.5 && alpha < 0.5)) discard;
    }
    if (pc.wire != 0)
    {
        outColor = finalColor(vec3(0.08), 1.0);
        return;
    }

    vec3 N = normalize(inNormal);
    vec3 c = clayLighting(N, inAo);
    if (world.terrain2.y > 0.5) c *= lodTint(inLod);
    c        = applyFog(c, inPos);
    outColor = finalColor(c, clamp(alpha, 0.0, 1.0));
}
