/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <Eigen/Geometry>

/**
 * Just a reminder how Eigen quaternions work:
 *
 * The Eigen::Quaternion is stored internally as (x,y,z,w).
 * The first three element (x,y,z) are the imaginary part.
 * The last element is the real part.
 * The indentity quaternion is therefore (0,0,0,1).
 */
namespace Saiga
{
using quat  = Eigen::Quaternionf;
using quatd = Eigen::Quaterniond;

using Quat = Eigen::Quaterniond;

}  // namespace Saiga
