/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/Align.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/core/util/Range.h"
#include "saiga/core/sophus/Sophus.h"

namespace Saiga
{
using SE3 = Sophus::SE3d;
using SO3 = Sophus::SO3d;
// using Sim3  = Sophus::Sim3d;
using DSim3 = Sophus::DSim3<double>;


// No idea why this method doesn't exist in sophus
// template <typename T>
// inline Sophus::Sim3<T> sim3(const Sophus::SE3<T>& se3, T scale)
//{
//    Sophus::Sim3<T> s(se3.unit_quaternion(), se3.translation());
//    s.setScale(scale);
//    return s;
//}

//// extract se3 + scale from sim3
// template <typename T>
// inline std::pair<Sophus::SE3<T>, T> se3Scale(const Sophus::Sim3<T>& sim3)
//{
//    double scale = sim3.scale();
//    Sophus::SE3<T> se3(sim3.rxso3().quaternion().normalized(), sim3.translation());
//    return {se3, scale};
//}


// Returns the SE3 which is inversely matching the given sim3:
//
// A [sim3] -> B [SE3]
// B.inverse() == se3Scale(sim3.inverse()).first
template <typename T>
inline Sophus::SE3<T> inverseMatchingSE3(const Sophus::DSim3<T>& sim3)
{
    // Alternative implementation using 2x inverse
    // return se3Scale(sim3.inverse()).first.inverse();

    Quat q   = sim3.se3().unit_quaternion();
    Vec3 t   = sim3.se3().translation();
    double s = sim3.scale();
    t *= (1. / s);
    return Sophus::SE3<T>(q, t);
}


}  // namespace Saiga
