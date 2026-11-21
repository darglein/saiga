/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/ContentConvert.h"

#include "saiga/engine/content/gen/TrigTable.h"

#include <cmath>

namespace Saiga
{
namespace Engine
{
Vec3 contentPosition(int64_t x256, int32_t yHeight, int64_t z256)
{
    return Vec3(double(x256) / 256.0, double(yHeight) / 1024.0, double(z256) / 256.0);
}

Vec3 contentPosition(const ChunkId& chunk, int32_t x256, int32_t yHeight, int32_t z256)
{
    return contentPosition(int64_t(chunk.x) * 65536 + x256, yHeight, int64_t(chunk.z) * 65536 + z256);
}

Quat contentRotation(uint16_t yaw)
{
    // The half angle of the yaw is index `yaw` of the 2^17-step circle
    const double w = std::ldexp(double(cosQ30(yaw)), -30);
    const double y = std::ldexp(double(sinQ30(yaw)), -30);
    const double n = std::sqrt(w * w + y * y);
    return Quat(w / n, 0, y / n, 0);
}

double contentScale(uint16_t scale)
{
    return double(scale) / 1024.0;
}

}  // namespace Engine
}  // namespace Saiga
