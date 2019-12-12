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


}  // namespace Saiga
