/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#ifndef HLSL_DEFINES_H
#define HLSL_DEFINES_H

#include "shaderConfig.h"

// HLSL to GLSL defines
#define float4 vec4
#define float3 vec3
#define float2 vec2
#define float4x4 mat4
#define float3x3 mat3
#define float2x2 mat2

//#define saturate(X) clamp(X, 0.f, 1.f)
#define frac(X) fract(X)
#define atan2(Y, X) atan(Y, X)
#define mul(A, B) ((B) * (A))

#define sat(X) saturate(X)
#define pfrac(X) ((X)-floor(X))
#define rcp(X) (1.0f / X)

#define PI uintBitsToFloat(0x40490fdc)
#define PHI (sqrt(5.f) * 0.5f + 0.5f)

#define mad(A, B, C) (A * B + C)
#define madfrac(A, B) mad((A), (B), -floor((A) * (B)))


#endif
