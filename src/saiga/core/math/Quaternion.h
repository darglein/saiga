/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/saiga_modules.h"

#ifdef SAIGA_USE_EIGEN
#include <Eigen/Geometry>
#else
#include "tiny-eigen/quaternion.h"
#endif

#include <iostream>

/**
 * Just a reminder how Eigen quaternions work:
 *
 * The Eigen::Quaternion is stored internally as (x,y,z,w).
 * The first three element (x,y,z) are the imaginary part.
 * The last element is the real part.
 * The identity quaternion is therefore (0,0,0,1).
 *
 * The Eigen Quaternion constructor is (w,x,y,z)
 * From the Eigen doc:
 * Note the order of the arguments: the real w coefficient first, while internally the coefficients are stored in the
 * following order: [x, y, z, w]
 */
namespace Saiga
{
using quat = Eigen::Quaternionf;
using Quat = Eigen::Quaterniond;
}  // namespace Saiga

namespace Eigen
{
template <typename Derived>
inline std::ostream& operator<<(std::ostream& os, const Eigen::QuaternionBase<Derived>& q)
{
    os << "quat(" << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ")";
    return os;
}

}  // namespace Eigen
