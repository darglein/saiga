/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/imu/Preintegration.h"


namespace Saiga::Imu
{
template <typename T>
struct CeresPreintegration
{
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Quat = Eigen::Quaternion<T>;
    using SO3  = Sophus::SO3<T>;
    using SE3  = Sophus::SE3<T>;

    CeresPreintegration(const Vec3& bias_gyro = Vec3::Zero(), const Vec3& bias_accel = Vec3::Zero())
        : bias_gyro_lin(bias_gyro), bias_accel_lin(bias_accel)
    {
    }



    Eigen::Matrix<T, 9, 1> Residual(const VelocityAndBiasBase<T>& vb_i, const SE3& _pose_i,
                                    const VelocityAndBiasBase<T>& vb_j, const SE3& _pose_j, const Vec3& g, T scale)
    {
        SE3 pose_i = _pose_i;
        SE3 pose_j = _pose_j;

        pose_i.translation() *= scale;
        pose_j.translation() *= scale;


        auto Pi = pose_i.translation();
        auto Pj = pose_j.translation();

        auto Vi = vb_i.velocity;
        auto Vj = vb_j.velocity;

        auto Ri = pose_i.so3();
        auto Rj = pose_j.so3();

        auto RiT = Ri.inverse();
        //        auto RjT = Rj.inverse();

        T dTij = delta_t;
        T dT2  = dTij * dTij;

        Eigen::Matrix<T, 3, 1> dPij = delta_x;
        Eigen::Matrix<T, 3, 1> dVij = delta_v;


        Eigen::Matrix<T, 3, 1> rPij = RiT * (Pi + Vi * dTij + g * 0.5 * dT2) + dPij - RiT * Pj;


        Eigen::Matrix<T, 3, 1> rVij = RiT * (Vi + g * dTij) + dVij - RiT * Vj;

        Sophus::SO3<T> rRij           = (delta_R).inverse() * RiT * Rj;
        Eigen::Matrix<T, 3, 1> rPhiij = rRij.log();

        Eigen::Matrix<T, 9, 1> residual;
        residual.template segment<3>(0) = rPij;    // position error
        residual.template segment<3>(3) = rVij;    // velocity error
        residual.template segment<3>(6) = rPhiij;  // rotation phi error
        return residual;
    }

    void Add(Imu::Data data, double dt)
    {
        if (dt == 0)
        {
            return;
        }
        Vec3 omega = data.omega.cast<T>() - bias_gyro_lin;
        Vec3 acc   = data.acceleration.cast<T>() - bias_accel_lin;
        T dt2      = T(dt * dt);

        SO3 dR = Sophus::SO3<T>::exp(omega * dt);

        delta_t += T(dt);
        delta_x += delta_v * dt + T(0.5) * dt2 * (delta_R * acc);  // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
        delta_v += dt * (delta_R * acc);
        delta_R = delta_R * dR;
    }

    void IntegrateMidPoint(const ImuSequence& sequence)
    {
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
    }

    T delta_t = T(0);
    Sophus::SO3<T> delta_R;
    Vec3 delta_x = Vec3::Zero();
    Vec3 delta_v = Vec3::Zero();

    Vec3 bias_gyro_lin, bias_accel_lin;
};



}  // namespace Saiga::Imu
