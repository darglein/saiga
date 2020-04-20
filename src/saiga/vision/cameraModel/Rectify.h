/**
 * Copyright (c) 2017 Darius Rückert
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

    double bf =  0;


    void Identity(const Intrinsics4& K, double bf)
    {
        K_src = K;
        K_dst = K;
        R = Quat::Identity();
        D_src = Distortion::Zero();
        this->bf =   bf;
    }
    // unrectified -> rectified
    Vec2 Forward(const Vec2& x)
    {
        Vec2 p   = K_src.unproject2(x);
        p        = undistortNormalizedPoint(p, D_src);
        Vec3 p_r = R * Vec3(p(0), p(1), 1);
        p        = Vec2(p_r(0) / p_r(2), p_r(1) / p_r(2));
        return K_dst.normalizedToImage(p);
    }

    // rectified -> unrectified
    Vec2 Backward(const Vec2& x)
    {
        Vec2 p   = K_dst.unproject2(x);
        Vec3 p_r = R.inverse() * Vec3(p(0), p(1), 1);
        p        = Vec2(p_r(0) / p_r(2), p_r(1) / p_r(2));
        p        = distortNormalizedPoint(p, D_src);
        return K_src.normalizedToImage(p);
    }
};


}  // namespace Saiga
