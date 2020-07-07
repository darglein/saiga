/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/imu/Imu.h"


namespace Saiga::Imu
{
// Computes a global gyro bias which minimizes the relative rotational error.
// The input is a array of IMU Sequences and the global start and end rotation for that sequence.
//
// Notes:
//   - In a perfect world, the problem is linear, but usually 2 iterations are recommended.
//   - If you're computing the bias for a VI system, make sure to transform the camera frame to the IMU frame.
SAIGA_VISION_API Vec3 SolveGlobalGyroBias(ArrayView<std::tuple<const Imu::ImuSequence*, Quat, Quat>> data, int max_its);

}  // namespace Saiga::Imu
