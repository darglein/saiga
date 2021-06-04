/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "Imu.h"

#include "saiga/core/util/BinaryFile.h"
#include "saiga/vision/imu/all.h"
#include "saiga/vision/util/Random.h"

#include <fstream>
#include <iomanip>
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
    strm << "Acc Noise:          " << sensor.acceleration_sigma << " / " << sensor.acceleration_random_walk
         << std::endl;
    strm << "Acc Noise Scaled:   " << sensor.acceleration_sigma * sqrt(sensor.frequency) << " / "
         << sensor.acceleration_random_walk * sqrt(sensor.frequency) << std::endl;
    strm << "Gyro Noise:         " << sensor.omega_sigma << " / " << sensor.omega_random_walk << std::endl;
    strm << "Gyro Noise Scaled:  " << sensor.omega_sigma * sqrt(sensor.frequency) << " / "
         << sensor.omega_random_walk * sqrt(sensor.frequency) << std::endl;
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

void ImuSequence::FixBorder()
{
    SAIGA_ASSERT(Valid());
    SAIGA_ASSERT(data.size() >= 2);

    if (data.front().timestamp < time_begin)
    {
        SAIGA_ASSERT(data[1].timestamp >= time_begin);
        data[0] = Imu::Data::Interpolate(data[0], data[1], time_begin);
    }

    if (data.back().timestamp > time_end)
    {
        SAIGA_ASSERT(data[data.size() - 2].timestamp <= time_end);
        data[data.size() - 1] = Imu::Data::Interpolate(data[data.size() - 2], data[data.size() - 1], time_end);
    }
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

void ImuSequence::AddNoise(double sigma_omega, double sigma_acc)
{
    for (auto& i : data)
    {
        i.omega += Random::MatrixGauss<Vec3>(0, sigma_omega);
        i.acceleration += Random::MatrixGauss<Vec3>(0, sigma_acc);
    }
}

void ImuSequence::AddBias(const Vec3& bias_gyro, const Vec3& bias_acc)
{
    for (auto& i : data)
    {
        i.omega += bias_gyro;
        i.acceleration += bias_acc;
    }
}

void ImuSequence::AddGravity(const Vec3& bias_gyro, const SO3& initial_orientation, const Vec3& global_gravity)
{
    Preintegration preint(bias_gyro);
    auto g = global_gravity;

    SO3 R = initial_orientation;
    data[0].acceleration += R.inverse() * g;
    //    data[0].acceleration += R * g;

    for (int i = 1; i < data.size(); ++i)
    {
        double dt = data[i].timestamp - data[i - 1].timestamp;
        preint.Add(data[i - 1].omega, Vec3::Zero(), dt, false);
        data[i].acceleration += (R * preint.delta_R).inverse() * g;
    }
}

void ImuSequence::Save(const std::string& dir) const
{
    BinaryFile ostream(dir, std::ios_base::out);
    ostream << time_begin << time_end;
    ostream << data;
}

void ImuSequence::Load(const std::string& dir)
{
    BinaryFile istream(dir, std::ios_base::in);
    istream >> time_begin >> time_end;
    istream >> data;
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

std::vector<ImuSequence> GenerateRandomSequence(int N, int K, double dt)
{
    SAIGA_ASSERT(dt > 0);
    SAIGA_ASSERT(N >= 2);
    SAIGA_ASSERT(K > 0);
    std::vector<Imu::ImuSequence> data;
    double t = 0;

    data.push_back(Imu::ImuSequence());
    data.front().time_end = t;


    Vec3 o   = Vec3::Random() * 0.02;
    Vec3 acc = Vec3::Random() * 0.2;

    for (int i = 0; i < N - 1; ++i)
    {
        Imu::ImuSequence seq;
        seq.time_begin = t;


        int this_K = K;  // Random::uniformInt(K / 2, K * 10);
        for (int k = 0; k < this_K; ++k)
        {
            Imu::Data id;
            o += Vec3::Random() * 0.01 * dt;
            acc += Vec3::Random() * 0.1 * dt;
            id.omega        = o;
            id.acceleration = acc;  // Vec3::Random() * 0.1;
            id.timestamp    = t;
            seq.data.push_back(id);
            t += dt;
        }
        seq.time_end = t;

        data.push_back(seq);
    }
    Imu::InterpolateMissingValues(data);

    return data;
}


}  // namespace Imu
}  // namespace Saiga
