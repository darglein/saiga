/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include "sophus/se3.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>


namespace Saiga
{
using SE3 = Sophus::SE3d;
using SO3 = Sophus::SO3d;

using Quat = Eigen::Quaterniond;

using Vec4 = Eigen::Vector4d;
using Vec3 = Eigen::Vector3d;
using Vec2 = Eigen::Vector2d;

using Mat4 = Eigen::Matrix4d;
using Mat3 = Eigen::Matrix3d;

// An Aligned std::vector
template <typename T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;


}  // namespace Saiga


namespace std
{
/**
 *  Basically a copy paste of the gcc make_shared implementation, but with the eigen aligned allocator.
 *
 * Note: For unique_ptr we do not need a custom make_aligned_unique, because it does not need an extra controlblock
 * and is therefore allocated with the default 'new' operator. (Assuming that the new operator was correctly
 * overloaded).
 */
template <typename _Tp, typename... _Args>
inline std::shared_ptr<_Tp> make_aligned_shared(_Args&&... __args)
{
    typedef typename std::remove_cv<_Tp>::type _Tp_nc;
    return std::allocate_shared<_Tp>(Eigen::aligned_allocator<_Tp_nc>(), std::forward<_Args>(__args)...);
}

}  // namespace std
