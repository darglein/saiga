/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "Imu.h"

#include "saiga/core/util/BinaryFile.h"
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
        i.omega += Random::gaussRandMatrix<Vec3>(0, sigma_omega);
        i.acceleration += Random::gaussRandMatrix<Vec3>(0, sigma_acc);
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
        preint.Add(data[i - 1].omega, Vec3::Zero(), dt);
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

void Preintegration::Add(const Vec3& omega_with_bias, const Vec3& acc_with_bias, double dt)
{
    //    std::cout << "Add " << dt << " " << omega_with_bias.transpose() << std::endl;


    if (dt == 0)
    {
        return;
    }

    Vec3 omega = omega_with_bias - bias_gyro_lin;
    Vec3 acc   = acc_with_bias - bias_accel_lin;
    double dt2 = dt * dt;

    SO3 dR = Sophus::SO3d::exp(omega * dt);
    Mat3 Jr;
    Sophus::rightJacobianSO3(omega * dt, Jr);



    // noise covariance propagation of delta measurements
    // err_k+1 = A*err_k + B*err_gyro + C*err_acc
    Mat3 I3x3               = Mat3::Identity();
    Matrix<double, 9, 9> A  = Matrix<double, 9, 9>::Identity();
    A.block<3, 3>(6, 6)     = dR.inverse().matrix();
    A.block<3, 3>(3, 6)     = -delta_R.matrix() * skew(acc) * dt;
    A.block<3, 3>(0, 6)     = -0.5 * delta_R.matrix() * skew(acc) * dt2;
    A.block<3, 3>(0, 3)     = I3x3 * dt;
    Matrix<double, 9, 3> Bg = Matrix<double, 9, 3>::Zero();
    Bg.block<3, 3>(6, 0)    = Jr * dt;
    Matrix<double, 9, 3> Ca = Matrix<double, 9, 3>::Zero();
    Ca.block<3, 3>(3, 0)    = delta_R.matrix() * dt;
    Ca.block<3, 3>(0, 0)    = 0.5 * delta_R.matrix() * dt2;


    cov_P_V_Phi = A * cov_P_V_Phi * A.transpose() + Bg * cov_gyro * Bg.transpose() + Ca * cov_acc * Ca.transpose();


    // jacobian of delta measurements w.r.t bias of gyro/acc
    // update P first, then V, then R
    J_P_Biasa += J_V_Biasa * dt - 0.5 * delta_R.matrix() * dt2;
    J_P_Biasg += J_V_Biasg * dt - 0.5 * delta_R.matrix() * skew(acc) * J_R_Biasg * dt2;
    J_V_Biasa += -delta_R.matrix() * dt;
    J_V_Biasg += -delta_R.matrix() * skew(acc) * J_R_Biasg * dt;

    J_R_Biasg = dR.inverse().matrix() * J_R_Biasg - Jr * dt;


    delta_t += dt;
    delta_x += delta_v * dt + 0.5 * dt2 * (delta_R * acc);  // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
    delta_v += dt * (delta_R * acc);
    delta_R = (delta_R * dR);
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

std::pair<SE3, Vec3> Preintegration::Predict(const SE3& initial_pose, const Vec3& initial_velocity,
                                             const Vec3& g_) const
{
    auto g              = g_;
    SO3 new_orientation = (initial_pose.so3() * delta_R);
    Vec3 new_velocity   = initial_velocity + g * delta_t + initial_pose.so3() * delta_v;
    Vec3 new_position   = initial_pose.translation() + initial_velocity * delta_t + 0.5 * g * delta_t * delta_t +
                        initial_pose.so3() * delta_x;
    return {SE3(new_orientation, new_position), new_velocity};
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
