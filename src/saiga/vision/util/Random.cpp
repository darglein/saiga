/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Random.h"

#include "saiga/core/util/assert.h"

namespace Saiga
{
namespace Random
{
SE3 randomSE3()
{
    Vec3 t  = MatrixUniform<Vec3>();
    Vec4 qc = MatrixUniform<Vec4>();
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




}  // namespace Random
}  // namespace Saiga
