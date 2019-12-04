/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

#include <vector>


namespace Saiga
{
struct SAIGA_VISION_API IMUData
{
    // Angular velocity in [rad/s] given by the gyroscope.
    Vec3 omega;

    // Linear acceleration in [m/s^2] given my the accelerometer.
    Vec3 acceleration;

    // Timestamp of the sensor reading in [s]
    double timestamp;

    IMUData() = default;
    IMUData(const Vec3& omega, const Vec3& acceleration, double timestamp)
        : omega(omega), acceleration(acceleration), timestamp(timestamp)
    {
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const IMUData& data);

struct IMUSensor
{
    // Number of meassueremnt per second in [hz]
    double frequency;

    // Standard deviation of the sensor noise
    double acceleration_sigma;
    double acceleration_random_walk;

    double omega_sigma;
    double omega_random_walk;

    // Transformation from the sensor coordinate system to the devices' coordinate system.
    SE3 sensor_to_body;
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const IMUSensor& sensor);


}  // namespace Saiga
