#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// RenderScene's post pass (S17, docs/engine_design.md §10.5 pass 14): the resolved HDR image -> the swap chain or
// offscreen image. Tone mapped: exposure scale, ACES, the output transfer (the piecewise sRGB curve for UNORM targets,
// linear for sRGB targets, which encode in hardware). Identity (clay mode): the scene wrote display values, which are
// copied clamped to [0, 1] (decoded to linear first for an sRGB target, so the hardware encode restores them).

#include "color.glsl"

layout(set = 0, binding = 0) uniform sampler2D hdr;

layout(push_constant) uniform Push
{
    // exposure / preExposure
    float exposureScale;
    // 0: Identity, 1: ACES
    int toneMap;
    // 1: the target is sRGB (the hardware encodes)
    int srgbTarget;
}
pc;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 c = sanitize(texelFetch(hdr, ivec2(gl_FragCoord.xy), 0).rgb);
    if (pc.toneMap == 0)
    {
        c = clamp(c, 0.0, 1.0);
        if (pc.srgbTarget != 0) c = srgbToLinear(c);
    }
    else
    {
        c = acesNarkowicz(c * pc.exposureScale);
        if (pc.srgbTarget == 0) c = linearToSrgb(c);
    }
    outColor = vec4(c, 1.0);
}
