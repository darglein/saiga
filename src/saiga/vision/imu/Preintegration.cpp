/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "Preintegration.h"

#include "saiga/core/util/BinaryFile.h"
#include "saiga/vision/util/Random.h"

#include <fstream>
#include <iomanip>
namespace Saiga
{
namespace Imu
{
void Preintegration::Add(const Vec3& omega_with_bias, const Vec3& acc_with_bias, double dt, bool derive)
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


#if 0
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
#endif

    if (derive)
    {
        // jacobian of delta measurements w.r.t bias of gyro/acc
        // update P first, then V, then R
        J_P_Biasa += J_V_Biasa * dt - 0.5 * delta_R.matrix() * dt2;
        J_P_Biasg += J_V_Biasg * dt - 0.5 * delta_R.matrix() * skew(acc) * J_R_Biasg * dt2;
        J_V_Biasa += -delta_R.matrix() * dt;
        J_V_Biasg += -delta_R.matrix() * skew(acc) * J_R_Biasg * dt;

        J_R_Biasg = dR.inverse().matrix() * J_R_Biasg - Jr * dt;
    }


    delta_t += dt;
    delta_x += delta_v * dt + 0.5 * dt2 * (delta_R * acc);  // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
    delta_v += dt * (delta_R * acc);
    delta_R = (delta_R * dR);
}

void Preintegration::IntegrateForward(const Imu::ImuSequence& sequence, bool derive)
{
    SAIGA_ASSERT(sequence.Valid());
    if (sequence.data.empty()) return;

    //
    auto first_value = sequence.data.front();
    SAIGA_ASSERT(first_value.timestamp >= sequence.time_begin);

    auto last_value = sequence.data.back();
    SAIGA_ASSERT(last_value.timestamp <= sequence.time_end);

    Add(first_value, first_value.timestamp - sequence.time_begin, derive);

    for (int i = 0; i < sequence.data.size() - 1; ++i)
    {
        Add(sequence.data[i], sequence.data[i + 1].timestamp - sequence.data[i].timestamp, derive);
    }

    Add(last_value, sequence.time_end - last_value.timestamp, derive);


    // SAIGA_ASSERT(std::abs(delta_t - (sequence.time_end - sequence.time_begin)) < 1e-10);
}

void Preintegration::IntegrateMidPoint(const Imu::ImuSequence& sequence, bool derive)
{
    SAIGA_ASSERT(sequence.Valid());
    if (sequence.data.empty()) return;

    //
    auto first_value = sequence.data.front();
    SAIGA_ASSERT(first_value.timestamp >= sequence.time_begin);

    auto last_value = sequence.data.back();
    SAIGA_ASSERT(last_value.timestamp <= sequence.time_end);

    Add(first_value, first_value.timestamp - sequence.time_begin, derive);

    for (int i = 0; i < sequence.data.size() - 1; ++i)
    {
        auto mid_point = Data::InterpolateAlpha(sequence.data[i], sequence.data[i + 1], 0.5);
        Add(mid_point, sequence.data[i + 1].timestamp - sequence.data[i].timestamp, derive);
    }

    Add(last_value, sequence.time_end - last_value.timestamp, derive);


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

Vec9 Preintegration::ImuError(const VelocityAndBias& delta_bias_i, const Vec3& v_i, const SE3& _pose_i, const Vec3& v_j,
                              const SE3& _pose_j, const Gravity& g, double scale, const Vec3& _weight_pvr,
                              Matrix<double, 9, 3>* J_biasa, Matrix<double, 9, 3>* J_biasg, Matrix<double, 9, 3>* J_v1,
                              Matrix<double, 9, 3>* J_v2, Matrix<double, 9, 6>* J_p1, Matrix<double, 9, 6>* J_p2,
                              Matrix<double, 9, 1>* J_scale, Matrix<double, 9, 3>* J_g) const
{
    double time_weight = 1.0 / sqrt(delta_t);
    SAIGA_ASSERT(std::isfinite(time_weight));

    Vec3 weight_pvr = _weight_pvr * time_weight;



    Vec9 residual = Vec9::Zero();

    SE3 pose_i = _pose_i;
    SE3 pose_j = _pose_j;

    pose_i.translation() *= scale;
    pose_j.translation() *= scale;


    auto Pi = pose_i.translation();
    auto Pj = pose_j.translation();


    auto Ri = pose_i.so3();
    auto Rj = pose_j.so3();

    auto RiT = Ri.inverse();
    auto RjT = Rj.inverse();


    double dTij = delta_t;
    double dT2  = dTij * dTij;

    Vec3 dPij = delta_x;
    Vec3 dVij = delta_v;

    // residual error of Delta Position measurement
    Vec3 rPij = RiT * (Pi + v_i * dTij + g.Get() * 0.5 * dT2) + dPij - RiT * Pj;


    rPij += J_P_Biasg * delta_bias_i.gyro_bias + J_P_Biasa * delta_bias_i.acc_bias;


    // residual error of Delta Velocity measurement
    Vec3 rVij = RiT * (v_i + g.Get() * dTij) + dVij - RiT * v_j;


    rVij += J_V_Biasg * delta_bias_i.gyro_bias + J_V_Biasa * delta_bias_i.acc_bias;


    Sophus::SO3 dR_dbg = Sophus::SO3d::exp(J_R_Biasg * delta_bias_i.gyro_bias);


    SO3 rRij    = Rj.inverse() * Ri * delta_R;
    rRij        = rRij * dR_dbg;
    Vec3 rPhiij = rRij.log();


    residual.segment<3>(0) = rPij * weight_pvr(0);    // position error
    residual.segment<3>(3) = rVij * weight_pvr(1);    // velocity error
    residual.segment<3>(6) = rPhiij * weight_pvr(2);  // rotation phi error

    if (J_biasg)
    {
        Mat3 JrInv_rPhi;
        Sophus::leftJacobianInvSO3(rPhiij, JrInv_rPhi);

        //        Mat3 ExprPhiijTrans = Sophus::SO3d::exp(rPhiij).inverse().matrix();
        Mat3 ExprPhiijTrans = Sophus::SO3d::exp(rPhiij).matrix();


        Mat3 corr;
        Sophus::rightJacobianSO3(J_R_Biasg * delta_bias_i.gyro_bias, corr);

        J_biasg->block<3, 3>(0, 0) = J_P_Biasg * weight_pvr(0);
        J_biasg->block<3, 3>(3, 0) = J_V_Biasg * weight_pvr(1);
        J_biasg->block<3, 3>(6, 0) = JrInv_rPhi * ExprPhiijTrans * corr * J_R_Biasg * weight_pvr(2);
    }

    if (J_biasa)
    {
        J_biasa->block<3, 3>(0, 0) = J_P_Biasa * weight_pvr(0);
        J_biasa->block<3, 3>(3, 0) = J_V_Biasa * weight_pvr(1);
        J_biasa->block<3, 3>(6, 0).setZero();
    }

    if (J_v1)
    {
        J_v1->block<3, 3>(0, 0) = RiT.matrix() * delta_t * weight_pvr(0);
        J_v1->block<3, 3>(3, 0) = RiT.matrix() * weight_pvr(1);
        J_v1->block<3, 3>(6, 0).setZero();
    }

    if (J_v2)
    {
        J_v2->block<3, 3>(0, 0).setZero();
        J_v2->block<3, 3>(3, 0) = -RiT.matrix() * weight_pvr(1);
        J_v2->block<3, 3>(6, 0).setZero();
    }

    if (J_p1)
    {
        Mat3 JrInv_rPhi;
        Sophus::leftJacobianInvSO3(rPhiij, JrInv_rPhi);

        J_p1->block<3, 3>(0, 0) = Mat3::Identity() * scale * weight_pvr(0);
        J_p1->block<3, 3>(3, 0).setZero();
        J_p1->block<3, 3>(6, 0).setZero();

        J_p1->block<3, 3>(0, 3) =
            skew(RiT * (Pi + v_i * delta_t + 0.5 * g.Get() * delta_t * delta_t - Pj)) * weight_pvr(0);
        J_p1->block<3, 3>(3, 3) = skew(RiT * (v_i + g.Get() * delta_t - v_j)) * weight_pvr(1);
        J_p1->block<3, 3>(6, 3) = JrInv_rPhi * RjT.matrix() * Ri.matrix() * weight_pvr(2);
    }

    if (J_p2)
    {
        Mat3 JrInv_rPhi;
        Sophus::leftJacobianInvSO3(rPhiij, JrInv_rPhi);

        J_p2->block<3, 3>(0, 0) = -(RiT * Rj).matrix() * scale * weight_pvr(0);
        J_p2->block<3, 3>(3, 0).setZero();
        J_p2->block<3, 3>(6, 0).setZero();

        J_p2->block<3, 3>(0, 3).setZero();
        J_p2->block<3, 3>(3, 3).setZero();
        J_p2->block<3, 3>(6, 3) = -JrInv_rPhi * weight_pvr(2);
    }

    if (J_scale)
    {
        J_scale->segment<3>(0) = (RiT * Pi - RiT * Pj) / scale * weight_pvr(0);
        J_scale->segment<3>(3).setZero();
        J_scale->segment<3>(6).setZero();
    }

    if (J_g)
    {
        J_g->block<3, 3>(0, 0) = RiT.matrix() * skew(g.R * (g.unit_gravity * -0.5 * dT2)) * weight_pvr(0);
        J_g->block<3, 3>(3, 0) = RiT.matrix() * skew(g.R * (g.unit_gravity * -dTij)) * weight_pvr(1);
        J_g->block<3, 3>(6, 0).setZero();
    }

    return residual;
}

Vec6 Preintegration::BiasChangeError(const VelocityAndBias& bias_i, const VelocityAndBias& delta_bias_i,
                                     const VelocityAndBias& bias_j, const VelocityAndBias& delta_bias_j,
                                     double _weight_acc, double _weight_gyro, Matrix<double, 6, 6>* J_a_g_i,
                                     Matrix<double, 6, 6>* J_a_g_j) const
{
    double time_weight = 1.0 / sqrt(delta_t);


    SAIGA_ASSERT(std::isfinite(time_weight));

    double weight_acc  = _weight_acc * time_weight;
    double weight_gyro = _weight_gyro * time_weight;

    Vec3 bias_diff_acc  = (bias_i.acc_bias + delta_bias_i.acc_bias) - (bias_j.acc_bias + delta_bias_j.acc_bias);
    Vec3 bias_diff_gyro = (bias_i.gyro_bias + delta_bias_i.gyro_bias) - (bias_j.gyro_bias + delta_bias_j.gyro_bias);
    Vec6 result;
    result.segment<3>(0) = bias_diff_acc * weight_acc;
    result.segment<3>(3) = bias_diff_gyro * weight_gyro;

    if (J_a_g_i)
    {
        J_a_g_i->setZero();
        J_a_g_i->diagonal()(0) = weight_acc;
        J_a_g_i->diagonal()(1) = weight_acc;
        J_a_g_i->diagonal()(2) = weight_acc;

        J_a_g_i->diagonal()(3) = weight_gyro;
        J_a_g_i->diagonal()(4) = weight_gyro;
        J_a_g_i->diagonal()(5) = weight_gyro;
    }

    if (J_a_g_j)
    {
        SAIGA_ASSERT(J_a_g_i);
        (*J_a_g_j) = -(*J_a_g_i);
    }

    return result;
}

Vec3 Preintegration::RotationalError(const SO3& pose_i, const SO3& pose_j, Matrix<double, 3, 3>* J_g) const
{
    double time_weight = 1.0 / sqrt(delta_t);

    SO3 rRij      = pose_j.inverse() * pose_i * delta_R;
    Vec3 rPhiij   = rRij.log();
    Vec3 residual = rPhiij * time_weight;

    if (J_g)
    {
        Mat3 Jlinv;
        Sophus::leftJacobianInvSO3(rPhiij, Jlinv);
        *J_g = Jlinv * J_R_Biasg * time_weight;
    }

    return residual;
}



}  // namespace Imu
}  // namespace Saiga
