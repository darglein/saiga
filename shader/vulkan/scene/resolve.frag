#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The tone-map weighted MSAA resolve of RenderScene (S17, docs/engine_design.md §10.5 pass 7): every sample is
// sanitized and weighted with 1 / (1 + luma(c * exposureScale)) (Karis), so bright edges do not alias after the tone
// map. The output stays pre-exposed (post applies the exposure scale).

#include "color.glsl"

layout(set = 0, binding = 0) uniform sampler2DMS hdr;

layout(push_constant) uniform Push
{
    int samples;
    // exposure / preExposure of this frame (as in post)
    float exposureScale;
}
pc;

layout(location = 0) out vec4 outColor;

void main()
{
    ivec2 p    = ivec2(gl_FragCoord.xy);
    vec3 sum   = vec3(0.0);
    float wsum = 0.0;
    for (int i = 0; i < pc.samples; ++i)
    {
        vec3 c  = sanitize(texelFetch(hdr, p, i).rgb);
        float w = 1.0 / (1.0 + luma(c * pc.exposureScale));
        sum += w * c;
        wsum += w;
    }
    outColor = vec4(sum / wsum, 1.0);
}
