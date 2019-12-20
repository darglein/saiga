/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

#include <vector>


namespace Saiga::Imu
{
using AngularVelocity = Vec3;
using Velocity        = Vec3;

using GyroBias = Vec3;
using AccBias  = Vec3;
using ImuBias  = Vec6;

// (Linear) velocity and ImuBias stacked as
// [Velocity | GyroBias | AccBias]
using VelocityAndBias = Vec9;


struct SAIGA_VISION_API Data
{
    // Angular velocity in [rad/s] given by the gyroscope.
    Vec3 omega;

    // Linear acceleration in [m/s^2] given my the accelerometer.
    Vec3 acceleration;

    // Timestamp of the sensor reading in [s]
    double timestamp;

    Data() = default;
    Data(const Vec3& omega, const Vec3& acceleration, double timestamp)
        : omega(omega), acceleration(acceleration), timestamp(timestamp)
    {
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    static Data Interpolate(const Data& before, const Data& after, double new_timestamp)
    {
        SAIGA_ASSERT(before.timestamp < new_timestamp);
        SAIGA_ASSERT(after.timestamp >  new_timestamp);
        double alpha = ( new_timestamp - before.timestamp) / (after.timestamp - before.timestamp);


        Data result;
        result.omega = (1.0 - alpha) * before.omega + alpha * after.omega;
        result.acceleration= (1.0 - alpha) * before.acceleration+ alpha * after.acceleration;
        result.timestamp = new_timestamp;
        return result;
    }
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const Imu::Data& data);


struct Sensor
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

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const Imu::Sensor& sensor);


}  // namespace Saiga::Imu
