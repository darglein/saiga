/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#ifndef NORMAL_SF_H
#define NORMAL_SF_H

#include "shaderConfig.h"
//
#include "hlslDefines.h"

#undef INFINITY
#define INFINITY 1024.0f

// All credits to Benjamin Keinert
// @benjamin.keinert@fau.de

// inverseSF / SF / inverseHSF / HSF - final version from the SFM paper.
FUNC_DECL float inverseSF(float3 p, float n)
{
    float phi = std::min(atan2(p[1], p[0]), PI), cosTheta = p[2];

    float k = std::max(2.f, floor(log(n * PI * sqrt(5.f) * (1.f - cosTheta * cosTheta)) / log(PHI * PHI)));

    float Fk = pow(PHI, k) / sqrt(5);
    float F0 = round(Fk), F1 = round(Fk * PHI);

    float2x2 B    = float2x2(2 * PI * madfrac(F0 + 1, PHI - 1) - 2 * PI * (PHI - 1),
                          2 * PI * madfrac(F1 + 1, PHI - 1) - 2 * PI * (PHI - 1), -2 * F0 / n, -2 * F1 / n);
    float2x2 invB = inverse(B);
    float2 c      = floor(mul(invB, float2(phi, cosTheta - (1 - 1 / n))));

    float d = INFINITY, j = 0;
    for (uint s = 0; s < 4; ++s)
    {
        float cosTheta = dot(B[1], float2(s % 2, s / 2) + c) + (1 - 1 / n);
        cosTheta       = clamp(cosTheta, -1.f, +1.f) * 2 - cosTheta;

        float i        = floor(n * 0.5 - cosTheta * n * 0.5);
        float phi      = 2 * PI * madfrac(i, PHI - 1);
        cosTheta       = 1 - (2 * i + 1) * rcp(n);
        float sinTheta = sqrt(1 - cosTheta * cosTheta);

        float3 q = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

        float squaredDistance = dot(q - p, q - p);
        if (squaredDistance < d)
        {
            d = squaredDistance;
            j = i;
        }
    }

    return j;
}

FUNC_DECL float3 SF(float i, float n)
{
    float phi      = 2 * PI * madfrac(i, PHI - 1);
    float cosTheta = 1 - (2 * i + 1) * rcp(n);
    float sinTheta = sqrt(saturate(1 - cosTheta * cosTheta));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

FUNC_DECL float inverseHSF(float3 p, float n)
{
    float phi = std::min(atan2(p.y, p.x), PI), cosTheta = std::max(p.z, 0.f);

    float k =
        std::max(2.f, floor(log(2.f * n * PI * sqrt(5.f) * saturate(1.f - cosTheta * cosTheta)) / log(PHI * PHI)));

    float Fk = pow(PHI, k) / sqrt(5);
    float F0 = round(Fk), F1 = round(Fk * PHI);

    float2x2 B    = float2x2(2 * PI * madfrac(F0 + 1, PHI - 1) - 2 * PI * (PHI - 1),
                          2 * PI * madfrac(F1 + 1, PHI - 1) - 2 * PI * (PHI - 1), -F0 / n, -F1 / n);
    float2x2 invB = inverse(B);
    float2 c      = floor(mul(invB, float2(phi, cosTheta - (1 - 0.5 / n))));

    float d = INFINITY, j = 0;
    for (uint s = 0; s < 4; ++s)
    {
        float cosTheta = dot(B[1], float2(s % 2, s / 2) + c) + (1 - 0.5 / n);
        if (cosTheta < 0) cosTheta = dot(B[1], c + (float2(s % 2, s / 2) * 2.f - 1.f)) + (1 - 0.5 / n);
        if (cosTheta < 0) cosTheta = -1;

        float i        = floor(n - cosTheta * n);
        float phi      = 2 * PI * madfrac(i, PHI - 1);
        cosTheta       = 1 - (i + 0.5) * rcp(n);
        float sinTheta = sqrt(1 - cosTheta * cosTheta);

        float3 q = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

        float squaredDistance = dot(q - p, q - p);
        if (squaredDistance < d)
        {
            d = squaredDistance;
            j = i;
        }
    }

    return j;
}

FUNC_DECL float3 HSF(float i, float n)
{
    float phi      = 2 * PI * madfrac(i, PHI - 1);
    float cosTheta = 1 - (i + 0.5) * rcp(n);
    float sinTheta = sqrt(saturate(1 - cosTheta * cosTheta));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

#endif
