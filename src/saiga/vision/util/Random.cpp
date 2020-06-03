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

Vec3 sphericalRand(double radius)
{
    float z = sampleDouble(-1.0, 1.0);
    float a = sampleDouble(0.0, pi<double>() * 2.0);

    float r = sqrt(1.0 - z * z);

    float x = r * cos(a);
    float y = r * sin(a);

    return Vec3(x, y, z) * radius;
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

Sophus::Sim3d randomSim3()
{
    auto a = randomSE3();
    Sophus::Sim3d result(a.unit_quaternion(), a.translation());
    result.setScale(sampleDouble(0.1, 2));
    return result;
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
