/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "Rectify.h"

namespace Saiga
{
void Rectification::Identity(const IntrinsicsPinholed& K, double bf)
{
    K_src    = K;
    K_dst    = K;
    R        = Quat::Identity();
    D_src    = Distortion();
    this->bf = bf;
}
Vec2 Rectification::Forward(const Vec2& x)
{
    Vec2 p   = K_src.unproject2(x);
    p        = undistortPointGN(p, p, D_src);
    Vec3 p_r = R * Vec3(p(0), p(1), 1);
    p        = Vec2(p_r(0) / p_r(2), p_r(1) / p_r(2));
    return K_dst.normalizedToImage(p);
}
Vec2 Rectification::Backward(const Vec2& x)
{
    Vec2 p   = K_dst.unproject2(x);
    Vec3 p_r = R.inverse() * Vec3(p(0), p(1), 1);
    p        = Vec2(p_r(0) / p_r(2), p_r(1) / p_r(2));
    p        = distortNormalizedPoint(p, D_src);
    return K_src.normalizedToImage(p);
}
}  // namespace Saiga