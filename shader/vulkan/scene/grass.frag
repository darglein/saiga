#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The grass of S20 (a copy of world/foliage.frag on TerrainFrame): alpha tested cards (alpha to coverage with MSAA).
// The foliage texture comes from RenderScene::setPrototypeEnvironment

#include "terrain_common.glsl"
#include "terrain_shading.glsl"

// Position relative to the camera
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inUvLayer;
layout(location = 3) in float inAo;
layout(location = 4) flat in float inLod;

layout(set = 0, binding = 7) uniform sampler2DArray foliage;

layout(location = 0) out vec4 outColor;

void main()
{
    float alpha = 1.0;
    if (inUvLayer.z >= 0.0)
    {
        alpha = foliageCoverage(foliage, inUvLayer, frame.grassPhases.w);
        if (alpha <= 0.0 || (frame.grass.w < 1.5 && alpha < 0.5)) discard;
    }

    vec3 N = normalize(inNormal);
    vec3 c = clayLighting(N, inAo);
    if (frame.flags.z == 1) c *= lodTint(inLod);
    c        = applyFog(c, inPos);
    outColor = finalColor(c, clamp(alpha, 0.0, 1.0));
}
