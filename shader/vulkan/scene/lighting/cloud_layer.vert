#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The 2D cloud layer's mesh (S36a1c, A4; CloudLayer): a camera-centred polar grid generated from gl_VertexIndex (rings
// with quadratic spacing out to the radius, segments around), at the altitude above the sea level, dropping with the
// earth's curvature (r^2 / (2 R)). Camera-relative positions (the view matrix holds only the rotation).

#include "atmosphere_common.glsl"
#include "scene_block.glsl"

layout(push_constant) uniform Push
{
    mat4 viewProj;
    vec4 params;  // x: the pre-exposure, y: rings, z: segments
}
pc;

layout(location = 0) out vec3 outRelPos;
layout(location = 1) out float outR01;

void main()
{
    const int rings    = int(pc.params.y);
    const int segments = int(pc.params.z);
    const ivec2 corners[6] = ivec2[](ivec2(0, 0), ivec2(1, 0), ivec2(1, 1), ivec2(0, 0), ivec2(1, 1), ivec2(0, 1));
    int quad   = gl_VertexIndex / 6;
    ivec2 c    = corners[gl_VertexIndex % 6];
    int ring   = quad / segments + c.y;
    int seg    = quad % segments + c.x;
    float t    = float(ring) / float(rings);
    float r    = lighting.clouds.y * t * t;
    float a    = 2.0 * kAtmPi * float(seg) / float(segments);
    vec3 p     = vec3(r * cos(a), lighting.clouds.x - lighting.atmosphere.x - r * r / (2.0 * kEarthRadiusM), r * sin(a));
    outRelPos  = p;
    outR01     = t * t;
    gl_Position = pc.viewProj * vec4(p, 1.0);
}
