#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The 2D cloud layer (S36a1c, A4; CloudLayer): drawn after the sky in the scene pass (depth tested against the scene,
// no depth write, premultiplied alpha). The coverage of the tiling texture at the cloudiness, lit by the shadow light
// through the transmittance at the layer's height and by the sky ambient (cloudLayerRadiance), seen through the air
// (it fades into the sky's colour with the distance) and the fog; the alpha fades over the outer 30% of the radius.
// Not in a debug frame (the data views show the surfaces).

#include "lighting.glsl"

layout(push_constant) uniform Push
{
    mat4 viewProj;
    vec4 params;  // x: the pre-exposure
}
pc;

layout(location = 0) in vec3 inRelPos;
layout(location = 1) in float inR01;

layout(location = 0) out vec4 outColor;

void main()
{
    if (debugViewActive()) discard;
    vec2 uv        = (lighting.cloudMotion.xy + inRelPos.xz) / kCloudTile;
    float coverage = cloudCoverageFromNoise(texture(cloudCoverage, uv).r, lighting.clouds.z);
    float alpha    = coverage * cloudEdgeAlpha(inR01);
    if (alpha <= 0.002) discard;
    vec3 v     = normalize(inRelPos);
    bool below = lighting.atmosphere.x < lighting.clouds.x;
    float rL   = atmosphereRadius(lighting.clouds.x);
    // The light at the layer: the sun or the moon (the shadow light), through the atmosphere at its height
    vec3 E = vec3(0.0);
    vec3 l = vec3(0.0, 1.0, 0.0);
    if (lighting.shadowLight.w > 0.0)
    {
        l          = normalize(lighting.shadowLight.xyz);
        float illu = lighting.shadowLight.w == 1.0 ? lighting.sun.w : lighting.moon.w;
        E          = illu * bodyTransmittanceAtRadius(transmittanceLut, rL, l);
    }
    vec3 skyE    = ambientDiffuse(vec3(0.0, 1.0, 0.0), inRelPos);
    vec3 groundE = ambientDiffuse(vec3(0.0, -1.0, 0.0), inRelPos);
    vec3 L       = cloudLayerRadiance(E, l, v, skyE, groundE, coverage, below);
    // Through the air: towards the sky's colour with the distance
    float air = exp(-length(inRelPos) / 30000.0);
    L         = mix(skyRadiance(skyViewLut, v, atmosphereRadius(lighting.atmosphere.x)), L, air);
    L         = lightingFog(L, inRelPos);
    outColor  = vec4(min(max(L, vec3(0.0)) * pc.params.x, vec3(60000.0)) * alpha, alpha);
}
