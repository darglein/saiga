/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Imu.h"
namespace Saiga
{
std::ostream& operator<<(std::ostream& strm, const Imu::Data& data)
{
    Vec6 v;
    v.segment<3>(0) = data.omega;
    v.segment<3>(3) = data.acceleration;
    strm << data.timestamp << " " << v.transpose();
    return strm;
}

std::ostream& Imu::operator<<(std::ostream& strm, const Imu::Sensor& sensor)
{
    strm << "[IMU::Sensor]" << std::endl;
    strm << "Frequency:          " << sensor.frequency << " hz" << std::endl;
    strm << "Acceleration Noise: " << sensor.acceleration_sigma << "" << sensor.acceleration_random_walk << std::endl;
    strm << "Omega Noise:        " << sensor.omega_sigma << "" << sensor.omega_random_walk << std::endl;
    strm << "Extrinsics          " << sensor.sensor_to_body << std::endl;
    return strm;
}

void Imu::Frame::computeInterpolatedImuValue()
{
    SAIGA_ASSERT(!imu_data_since_last_frame.empty());
    Imu::Data before = imu_data_since_last_frame.back();
    Imu::Data after  = imu_directly_after_this_frame;
    SAIGA_ASSERT(std::isfinite(before.timestamp));
    SAIGA_ASSERT(std::isfinite(after.timestamp));
    interpolated_imu = Imu::Data::Interpolate(before, after, timestamp);
}

void Imu::Frame::sanityCheck(const Sensor& sensor)
{
    SAIGA_ASSERT(!imu_data_since_last_frame.empty());
    SAIGA_ASSERT(std::isfinite(timestamp));
    SAIGA_ASSERT(std::isfinite(imu_directly_after_this_frame.timestamp));
    SAIGA_ASSERT(std::isfinite(interpolated_imu.timestamp));

    double dt = imu_directly_after_this_frame.timestamp - timestamp;
    SAIGA_ASSERT(dt >= 0);
    SAIGA_ASSERT(dt <= sensor.frequency);

    dt = timestamp - imu_data_since_last_frame.back().timestamp;
    SAIGA_ASSERT(dt >= 0);
    SAIGA_ASSERT(dt <= sensor.frequency);

    for (auto d : imu_data_since_last_frame)
    {
        SAIGA_ASSERT(d.timestamp < timestamp);
    }
}


}  // namespace Saiga
