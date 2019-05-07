/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/Align.h"

#include "sophus/se3.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace Saiga
{
using SE3 = Sophus::SE3d;
using SO3 = Sophus::SO3d;

using Quat = Eigen::Quaterniond;

using Vec7 = Eigen::Matrix<double, 7, 1>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec4 = Eigen::Matrix<double, 4, 1>;
using Vec3 = Eigen::Matrix<double, 3, 1>;
using Vec2 = Eigen::Matrix<double, 2, 1>;

using Vec4f = Eigen::Matrix<float, 4, 1>;
using Vec3f = Eigen::Matrix<float, 3, 1>;
using Vec2f = Eigen::Matrix<float, 2, 1>;

using Mat4 = Eigen::Matrix4d;
using Mat3 = Eigen::Matrix3d;



}  // namespace Saiga

