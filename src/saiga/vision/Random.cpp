/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Random.h"

#include "saiga/core/util/assert.h"

namespace Saiga
{
namespace Random
{
Vec3 linearRand(Vec3 low, Vec3 high)
{
    return {sampleDouble(low(0), high(0)), sampleDouble(low(1), high(1)), sampleDouble(low(2), high(2))};
}

Vec3 ballRand(double radius)
{
    SAIGA_ASSERT(radius >= 0);
    // Credits to random.inl from the glm library
    auto r2 = radius * radius;
    Vec3 low(-radius, -radius, -radius);
    Vec3 high(radius, radius, radius);
    double lenRes;
    Vec3 result;
    do
    {
        result = linearRand(low, high);
        lenRes = result.squaredNorm();
    } while (lenRes > r2);
    return result;
}



}  // namespace Random
}  // namespace Saiga
