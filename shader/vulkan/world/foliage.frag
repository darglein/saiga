#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "shading.glsl"

// The grass: alpha tested cards (alpha to coverage with MSAA). The trees are RenderScene's PrototypeFoliage class
// (vulkan/scene/prototype_foliage.frag, the same shading from prototype_shading.glsl).

// Position relative to the camera
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inUvLayer;
layout(location = 3) in float inAo;
layout(location = 4) flat in float inLod;

layout(set = 0, binding = 3) uniform sampler2DArray foliage;

layout(location = 0) out vec4 outColor;

void main()
{
    float alpha = 1.0;
    if (inUvLayer.z >= 0.0)
    {
        alpha = foliageCoverage(foliage, inUvLayer, world.terrain2.w);
        if (alpha <= 0.0 || (world.grass.w < 1.5 && alpha < 0.5)) discard;
    }

    vec3 N = normalize(inNormal);
    vec3 c = clayLighting(N, inAo);
    if (world.terrain2.y > 0.5) c *= lodTint(inLod);
    c        = applyFog(c, inPos);
    outColor = finalColor(c, clamp(alpha, 0.0, 1.0));
}
