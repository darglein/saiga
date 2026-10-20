/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/cuda/cudaHelper.h"

#include <curand_kernel.h>

namespace Saiga
{
namespace CUDA
{
__device__ inline float linearRand(float min, float max, curandState& state)
{
    float n = curand_uniform(&state);
    return (max - min) * n + min;
}

__device__ inline vec3 sphericalRand(float radius, curandState& state)
{
    float z = linearRand(-1.0, 1.0, state);
    float a = linearRand(0.0, 6.283185307179586476925286766559, state);

    float r = sqrt(1.0 - z * z);

    float x = r * cos(a);
    float y = r * sin(a);

    return vec3(x, y, z) * radius;
}


__device__ inline vec3 sampleUnitCone(float angle, curandState& state)
{
    float z = linearRand(cos(angle), float(1.0), state);
    float a = linearRand(float(0), float(6.283185307179586476925286766559f), state);

    float r = sqrtf(float(1) - z * z);

    float x = r * cos(a);
    float y = r * sin(a);

    return vec3(x, z, y);
}

__device__ inline vec3 sampleCone(const vec3& dir, float angle, curandState& state)
{
    // sampleUnitCone samples around the y axis. Map y to 'dir' with an orthonormal basis (tangent, dir, bitangent).
    // Saiga::rotation is a host function and cannot be used here.
    vec3 v       = sampleUnitCone(angle, state);
    vec3 d       = dir.normalized();
    vec3 helper  = fabsf(d.x()) < 0.9f ? vec3(1, 0, 0) : vec3(0, 1, 0);
    vec3 tangent = helper.cross(d).normalized();
    vec3 bitan   = d.cross(tangent);
    return tangent * v.x() + d * v.y() + bitan * v.z();
}

SAIGA_CUDA_API extern void initRandom(ArrayView<curandState> states, unsigned long long seed);
SAIGA_CUDA_API extern void randomTest();

}  // namespace CUDA
}  // namespace Saiga
