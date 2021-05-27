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
    IntrinsicsPinholed K_src;
    Saiga::Distortion D_src;

    Quat R;
    IntrinsicsPinholed K_dst;

    double bf = 0;


    void Identity(const IntrinsicsPinholed& K, double bf);
    // unrectified -> rectified
    Vec2 Forward(const Vec2& x);

    // rectified -> unrectified
    Vec2 Backward(const Vec2& x);
};


}  // namespace Saiga
