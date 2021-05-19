/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
namespace Saiga
{
inline uint64_t Morton3DBitInterleave64(uint64_t x)
{
    x &= 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

inline uint64_t Morton3D(const ivec3& v)
{
    uint64_t x = Morton3DBitInterleave64(v.x());
    uint64_t y = Morton3DBitInterleave64(v.y());
    uint64_t z = Morton3DBitInterleave64(v.z());
    return x | (y << 1) | (z << 2);
}

}  // namespace Saiga
