/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#ifndef NORMAL_PACKING_H
#define NORMAL_PACKING_H

#include "shaderConfig.h"


// https://aras-p.info/texts/CompactNormalStorage.html#method04spheremap
FUNC_DECL vec2 PackNormalSpheremap(const vec3& n)
{
    float p  = sqrt(n.z() * 8 + 8);
    vec2 enc = vec2(n[0], n[1]) * (1.f / p);
    // enc += vec2(0.5f, 0.5f);
    return enc;
}

FUNC_DECL vec3 UnpackNormalSpheremap(const vec2& enc)
{
    // vec2 fenc = enc * 4 - vec2(2, 2);
    vec2 fenc = enc * 4;
    float f   = fenc.dot(fenc);
    if (f != f)
    {
        return vec3(0, 0, -1);
    }
    float g = sqrt(max(1 - f * 0.25f, 0.f));
    vec3 n(fenc[0] * g, fenc[1] * g, 1 - f * 0.5f);
    return n;
}

// https://aras-p.info/texts/CompactNormalStorage.html#method07stereo
FUNC_DECL vec2 PackNormalStereographic(vec3 n)
{
    float scale = 1.7777;
    vec2 enc    = vec2(n[0], n[1]) / (n[2] + 1);
    enc /= scale;
    // enc = enc * 0.5 + vec2(0.5, 0.5);
    return enc;
}

vec3 UnpackNormalStereographic(vec2 enc)
{
    // enc = enc * 2 - vec2(1, 1);

    float scale = 1.7777;
    enc         = enc * scale;
    vec3 nn(enc[0], enc[1], 1);
    float g = 2.0 / dot(nn, nn);
    if (g != g)
    {
        return vec3(0, 0, -1);
    }
    vec3 n(nn[0] * g, nn[1] * g, g - 1);
    return n;
}



vec2 PackNormalSpherical(vec3 n)
{
    float kPI = 3.1415926536f;
    vec2 enc  = vec2(atan2(n[1], n[0]) / kPI, n[2]);
    return enc;
}
vec3 UnpackNormalSpherical(vec2 enc)
{
    float kPI = 3.1415926536f;
    vec2 scth;

    // sincos(ang.x * kPI, scth.x, scth.y);
    scth[0] = sin(enc[0] * kPI);
    scth[1] = cos(enc[0] * kPI);

    vec2 scphi = vec2(sqrt(1.0 - enc[1] * enc[1]), enc[1]);
    return vec3(scth[1] * scphi[0], scth[0] * scphi[0], scphi[1]);
}



#endif
