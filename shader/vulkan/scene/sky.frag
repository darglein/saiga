#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The sky of S20 (a copy of world/sky.frag on TerrainFrame): the gradient from the fog color at the horizon to the
// zenith color and a glow around the sun, only where nothing was drawn (EQUAL at depth 0). LIGHTING_PBR (S36): the
// same gradient in clay mode; in the physical path the ambient's sky radiance (the SH, until the atmosphere of a1c)

#include "terrain_common.glsl"
#include "terrain_shading.glsl"
#ifdef LIGHTING_PBR
#    include "lighting/lighting.glsl"
#endif

layout(location = 0) in vec2 inNdc;
layout(location = 0) out vec4 outColor;

void main()
{
    // A point on the view ray (depth 0.5 is at 2 * zNear)
    vec4 p   = frame.invViewProjRel * vec4(inNdc, 0.5, 1.0);
    vec3 dir = normalize(p.xyz / p.w);

#ifdef LIGHTING_PBR
    // A debug frame: no surface (0)
    if (debugViewActive())
    {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    if (physicalLighting())
    {
        outColor = finalColor(ambientDiffuse(dir, dir) / kPi, 1.0);
        return;
    }
#endif
    float t = clamp(dir.y, 0.0, 1.0);
    vec3 c  = mix(frame.fogColor.rgb, frame.skyColor.rgb, pow(t, 0.45));
    // A soft glow around the sun
    float sun = max(dot(dir, normalize(frame.sunDir.xyz)), 0.0);
    c += vec3(0.25) * pow(sun, 48.0);
    outColor = finalColor(c, 1.0);
}
