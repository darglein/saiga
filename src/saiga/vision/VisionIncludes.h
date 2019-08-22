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

#include "sophus/se3.hpp"
#include "sophus/sim3.hpp"


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

}  // namespace Saiga
