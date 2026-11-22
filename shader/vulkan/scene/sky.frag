#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The sky of S20, only where nothing was drawn (EQUAL at depth 0). In clay mode (S36: the studio environment) S20's
// gradient from the studio fog colour at the horizon to the zenith colour 0.2 and a glow around the studio sun; in the
// physical path the atmosphere (A4): the sky-view LUT at the camera height, the sun
// disc (its radiance clamped below the fp16 limit after the pre-exposure, limb darkened), the moon disc with its phase
// and the full moon's illuminance, and the stars (stars.glsl; faded with the sun's elevation), each through the
// transmittance to the top of the atmosphere; below the limb the LUT's ground with the height fog of its distance

#include "terrain_common.glsl"
#include "terrain_shading.glsl"
#include "lighting/lighting.glsl"
#include "lighting/stars.glsl"

layout(location = 0) in vec2 inNdc;
layout(location = 0) out vec4 outColor;

void main()
{
    // A point on the view ray (depth 0.5 is at 2 * zNear)
    vec4 p   = frame.invViewProjRel * vec4(inNdc, 0.5, 1.0);
    vec3 dir = normalize(p.xyz / p.w);

    // A debug frame: no surface (0)
    if (debugViewActive())
    {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    if (physicalLighting())
    {
        float r     = atmosphereRadius(lighting.atmosphere.x);
        vec3 c      = skyRadiance(skyViewLut, dir, lighting.atmosphere.w);
        float pixel = max(length(fwidth(dir)), 1e-5);
        vec3 Tview  = bodyTransmittanceAtRadius(transmittanceLut, r, dir);
        // The sun disc (angular radius 0.00465 rad), limb darkened
        const float sunRadius = 0.00465;
        float cs              = dot(dir, normalize(lighting.sun.xyz));
        if (lighting.sun.w > 0.0 && cs > cos(sunRadius + pixel))
        {
            float a    = acos(clamp(cs, -1.0, 1.0));
            float disc = 1.0 - smoothstep(sunRadius - 0.5 * pixel, sunRadius + 0.5 * pixel, a);
            float mu   = sqrt(max(1.0 - (a * a) / (sunRadius * sunRadius), 0.0));
            float limb = 1.0 - 0.6 * (1.0 - mu);
            c += Tview * (lighting.sun.w / (kPi * sunRadius * sunRadius) * disc * limb / 0.8);
        }
        // The moon disc and the stars (faded below the horizon)
        c += Tview * moonDiscRadiance(dir, normalize(lighting.moon.xyz), normalize(lighting.sun.xyz), pixel,
                                      lighting.starRotation[1].w);
        // The stars: faded out by the sun's elevation on the CPU (no star loop in a day sky)
        if (lighting.starRotation[0].w > 0.0)
        {
            vec3 celestial = vec3(dot(lighting.starRotation[0].xyz, dir), dot(lighting.starRotation[1].xyz, dir),
                                  dot(lighting.starRotation[2].xyz, dir));
            c += Tview * starRadiance(celestial, pixel, lighting.starRotation[0].w) * smoothstep(-0.02, 0.02, dir.y);
        }
        // Below the limb the LUT shows the planet's ground: it gets the height fog of its distance like the terrain
        // does (no seam where a terrain ends before the horizon)
        if (rayHitsGround(r, dir.y))
        {
            vec4 fog = physicalFog(dir * (distanceToGround(r, dir.y) * 1000.0));
            c        = c * fog.a + fog.rgb;
        }
        outColor = vec4(min(max(c, vec3(0.0)) * lighting.exposure.y, vec3(60000.0)), 1.0);
        return;
    }
    float t = clamp(dir.y, 0.0, 1.0);
    vec3 c  = mix(lighting.fogColorLin.rgb, vec3(0.2), pow(t, 0.45));
    // A soft glow around the sun
    float sun = max(dot(dir, normalize(lighting.studioSun.xyz)), 0.0);
    c += vec3(0.25) * pow(sun, 48.0);
    outColor = finalColor(c, 1.0);
}
