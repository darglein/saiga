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

SE3 randomSE3()
{
    Vec3 t  = Vec3::Random();
    Vec4 qc = Vec4::Random();
    Quat q;
    q.coeffs() = qc;
    q.normalize();
    if (q.w() < 0) q.coeffs() *= -1;
    return SE3(q, t);
}

Sim3 randomSim3()
{
    return sim3(randomSE3(), sampleDouble(0.1, 2));
}

DSim3 randomDSim3()
{
    return DSim3(randomSE3(), sampleDouble(0.1, 2));
}

Quat randomQuat()
{
    Vec4 qc = Vec4::Random();
    Quat q;
    q.coeffs() = qc;
    q.normalize();
    if (q.w() < 0) q.coeffs() *= -1;
    return q;
}



}  // namespace Random
}  // namespace Saiga
