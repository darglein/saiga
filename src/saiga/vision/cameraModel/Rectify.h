/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "Distortion.h"
#include "Intrinsics4.h"

namespace Saiga
{
struct Rectification
{
    Intrinsics4 K_src;
    Saiga::Distortion D_src;

    Quat R;
    Intrinsics4 K_dst;

    double bf = 0;


    void Identity(const Intrinsics4& K, double bf);
    // unrectified -> rectified
    Vec2 Forward(const Vec2& x);

    // rectified -> unrectified
    Vec2 Backward(const Vec2& x);
};


}  // namespace Saiga
