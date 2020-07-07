/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Imu.h"
namespace Saiga
{
namespace Imu
{
std::ostream& operator<<(std::ostream& strm, const Imu::Data& data)
{
    Vec6 v;
    v.segment<3>(0) = data.omega;
    v.segment<3>(3) = data.acceleration;
    strm << data.timestamp << " " << v.transpose();
    return strm;
}

std::ostream& operator<<(std::ostream& strm, const Imu::Sensor& sensor)
{
    strm << "[IMU::Sensor]" << std::endl;
    strm << "Frequency:          " << sensor.frequency << " hz" << std::endl;
    strm << "Acceleration Noise: " << sensor.acceleration_sigma << "" << sensor.acceleration_random_walk << std::endl;
    strm << "Omega Noise:        " << sensor.omega_sigma << "" << sensor.omega_random_walk << std::endl;
    strm << "Extrinsics          " << sensor.sensor_to_body << std::endl;
    return strm;
}


std::ostream& operator<<(std::ostream& strm, const Imu::ImuSequence& sequence)
{
    strm << "[Imu::Sequence] " << sequence.time_begin << " -> " << sequence.time_end << " ( " << sequence.data.size()
         << " Values)" << std::endl;
    for (auto i : sequence.data)
    {
        strm << i.timestamp << ": " << i.omega.transpose() << std::endl;
    }
    return strm;
}

void ImuSequence::Add(const ImuSequence& other)
{
    if (!Valid())
    {
        *this = other;
        return;
    }


    if (!other.Valid())
    {
        return;
    }


    SAIGA_ASSERT(time_end == other.time_begin);
    time_end = other.time_end;

    if (data.empty())
    {
        data = other.data;
        return;
    }

    if (other.data.empty())
    {
        return;
    }

    if (data.back().timestamp < other.data.front().timestamp)
    {
        data.push_back(other.data.front());
    }

    for (int i = 1; i < other.data.size(); ++i)
    {
        data.push_back(other.data[i]);
    }
}

void Preintegration::Add(const Vec3& omega_with_bias, const Vec3& acc_with_bias, double dt)
{
    //    std::cout << "Add " << dt << " " << omega_with_bias.transpose() << std::endl;


    if (dt == 0)
    {
        return;
    }

    Vec3 omega = omega_with_bias - bias_gyro_lin;
    //        Vec3 acc   = acc_with_bias - bias_accel_lin;

    Quat dR = Sophus::SO3d::exp(omega * dt).unit_quaternion();
    Mat3 Jr;
    Sophus::rightJacobianSO3(omega * dt, Jr);
    J_R_Biasg = dR.inverse().matrix() * J_R_Biasg - Jr * dt;
    delta_R   = (delta_R * dR).normalized();
    delta_t += dt;
}

void Preintegration::IntegrateForward(const Imu::ImuSequence& sequence)
{
    SAIGA_ASSERT(sequence.Valid());
    if (sequence.data.empty()) return;

    //
    auto first_value = sequence.data.front();
    SAIGA_ASSERT(first_value.timestamp >= sequence.time_begin);

    auto last_value = sequence.data.back();
    SAIGA_ASSERT(last_value.timestamp <= sequence.time_end);

    Add(first_value, first_value.timestamp - sequence.time_begin);

    for (int i = 0; i < sequence.data.size() - 1; ++i)
    {
        Add(sequence.data[i], sequence.data[i + 1].timestamp - sequence.data[i].timestamp);
    }

    Add(last_value, sequence.time_end - last_value.timestamp);


    // SAIGA_ASSERT(std::abs(delta_t - (sequence.time_end - sequence.time_begin)) < 1e-10);
}

void Preintegration::IntegrateMidPoint(const Imu::ImuSequence& sequence)
{
    SAIGA_ASSERT(sequence.Valid());
    if (sequence.data.empty()) return;

    //
    auto first_value = sequence.data.front();
    SAIGA_ASSERT(first_value.timestamp >= sequence.time_begin);

    auto last_value = sequence.data.back();
    SAIGA_ASSERT(last_value.timestamp <= sequence.time_end);

    Add(first_value, first_value.timestamp - sequence.time_begin);

    for (int i = 0; i < sequence.data.size() - 1; ++i)
    {
        auto mid_point = Data::InterpolateAlpha(sequence.data[i], sequence.data[i + 1], 0.5);
        Add(mid_point, sequence.data[i + 1].timestamp - sequence.data[i].timestamp);
    }

    Add(last_value, sequence.time_end - last_value.timestamp);


    SAIGA_ASSERT(std::abs(delta_t - (sequence.time_end - sequence.time_begin)) < 1e-10);
}

void InterpolateMissingValues(ArrayView<Imu::ImuSequence> sequences)
{
    // Find correct end of sequence i by interpolating between i and i+1
    for (int i = 0; i < sequences.size() - 1; ++i)
    {
        Imu::ImuSequence& current = sequences[i];
        Imu::ImuSequence& next    = sequences[i + 1];

        if (current.data.empty() || next.data.empty()) continue;

        SAIGA_ASSERT(current.time_end == next.time_begin);

        double t = current.time_end;
        auto in  = Imu::Data::Interpolate(current.data.back(), next.data.front(), t);

        if (current.data.back().timestamp < t)
        {
            current.data.push_back(in);
        }

        if (next.data.front().timestamp > t)
        {
            next.data.insert(next.data.begin(), in);
        }
    }
}


}  // namespace Imu
}  // namespace Saiga
