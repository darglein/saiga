/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/Align.h"
#include "saiga/core/util/Range.h"
#include "saiga/vision/sophus/Sophus.h"

namespace Saiga
{
using SE3  = Sophus::SE3d;
using SO3  = Sophus::SO3d;
using Sim3 = Sophus::Sim3d;


// No idea why this method doesn't exist in sophus
inline Sim3 sim3(const SE3& se3, double scale)
{
    Sim3 s(se3.unit_quaternion(), se3.translation());
    s.setScale(scale);
    return s;
}

// extract se3 + scale from sim3
inline std::pair<SE3, double> se3Scale(const Sim3& sim3)
{
    double scale = sim3.scale();
    SE3 se3(sim3.rxso3().quaternion().normalized(), sim3.translation());
    return {se3, scale};
}


// Returns the SE3 which is inversely matching the given sim3:
//
// A [sim3] -> B [SE3]
// B.inverse() == se3Scale(sim3.inverse()).first
inline SE3 inverseMatchingSE3(const Sim3& sim3)
{
    // Alternative implementation using 2x inverse
    // return se3Scale(sim3.inverse()).first.inverse();

    Quat q   = sim3.rxso3().quaternion().normalized();
    Vec3 t   = sim3.translation();
    double s = sim3.scale();
    t *= (1. / s);
    return SE3(q, t);
}

// inline SE3 operator*(const Sim3& l, const SE3& r)
//{
//    return se3Scale(l * sim3(r, 1)).first;
//}

}  // namespace Saiga
