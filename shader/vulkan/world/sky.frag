#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "shading.glsl"

layout(location = 0) in vec2 inNdc;
layout(location = 0) out vec4 outColor;

void main()
{
    // A point on the view ray (depth 0.5 is at 2 * zNear)
    vec4 p   = world.invViewProj * vec4(inNdc, 0.5, 1.0);
    vec3 dir = normalize(p.xyz / p.w - world.cameraPos.xyz);

    float t = clamp(dir.y, 0.0, 1.0);
    vec3 c  = mix(world.fogColor.rgb, world.skyColor.rgb, pow(t, 0.45));
    // A soft glow around the sun
    float sun = max(dot(dir, normalize(world.sunDir.xyz)), 0.0);
    c += vec3(0.25) * pow(sun, 48.0);
    outColor = finalColor(c, 1.0);
}
