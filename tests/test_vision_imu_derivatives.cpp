/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vision/imu/Solver.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
#include "numeric_derivative.h"
namespace Saiga
{
// Generates smooth imu measurements
std::vector<Imu::ImuSequence> GenerateRandomSequence(int N, int K, double dt)
{
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


        int this_K = Random::uniformInt(K / 2, K * 10);
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


namespace Imu
{
Vec9 ImuError(const ImuSequence& s, const VelocityAndBias& vb_i, const SE3& pose_i, const VelocityAndBias& vb_j,
              const SE3& pose_j, const Vec3& g, Matrix<double, 9, 3>* J_P_biasa = nullptr,
              Matrix<double, 9, 3>* J_R_biasg = nullptr)
{
    Vec9 residual = Vec9::Zero();


    auto Pi = pose_i.translation();
    auto Pj = pose_j.translation();

    auto Vi = vb_i.velocity;
    auto Vj = vb_j.velocity;

    auto Ri = pose_i.so3();
    auto Rj = pose_j.so3();

    auto RiT = Ri.inverse();
    auto RjT = Rj.inverse();



    Preintegration preint(vb_i);
    preint.IntegrateMidPoint(s);

    double dTij = preint.delta_t;
    double dT2  = dTij * dTij;

    Vec3 dPij = preint.delta_x;
    Vec3 dVij = preint.delta_v;

    //    auto prediction = preint.Predict(pose_i, vb_i.velocity, g);


    // residual error of Delta Position measurement
    Vec3 rPij = RiT * (Pj - Pi - Vi * dTij - 0.5 * g * dT2) - dPij;
    // - (dPij + M.getJPBiasg() * dBgi + M.getJPBiasa() * dBai);  // this line includes correction term of bias change.
    // residual error of Delta Velocity measurement
    Vec3 rVij = RiT * (Vj - Vi - g * dTij) - dVij;
    // - (dVij + M.getJVBiasg() * dBgi + M.getJVBiasa() * dBai);  // this line includes correction term of bias change

    // Rotation
    Sophus::SO3 dRij = Sophus::SO3(preint.delta_R);

    Sophus::SO3 rRij = (dRij).inverse() * RiT * Rj;
    Vec3 rPhiij      = rRij.log();

    residual.segment<3>(0) = rPij;    // position error
    residual.segment<3>(3) = rVij;    // velocity error
    residual.segment<3>(6) = rPhiij;  // rotation phi error

    if (J_R_biasg)
    {
        J_R_biasg->setZero();

        Mat3 JrInv_rPhi;
        Sophus::rightJacobianInvSO3(rPhiij, JrInv_rPhi);

        Mat3 ExprPhiijTrans = Sophus::SO3d::exp(rPhiij).inverse().matrix();

        //        Mat3 JrBiasGCorr = Sophus::rightJacobianSO3(J_rPhi_dbg * dBgi);


        //          Matrix3d J_rPhi_dbg = M.getJRBiasg();

        J_R_biasg->block<3, 3>(0, 0) = -preint.J_P_Biasg;
        J_R_biasg->block<3, 3>(3, 0) = -preint.J_V_Biasg;

        J_R_biasg->block<3, 3>(6, 0) = -JrInv_rPhi * ExprPhiijTrans /** JrBiasGCorr*/ * preint.J_R_Biasg;
        //        J_R_biasg -
        //        std::cout << preint.J_P_Biasg << std::endl;
        //        std::cout << preint.J_V_Biasg << std::endl;
    }

    return residual;
}
}  // namespace Imu


TEST(Imu, SaveLoad)
{
    Random::setSeed(95665);
    auto s = GenerateRandomSequence(10, 5, 0.1)[1];


    Imu::VelocityAndBias vb_i, vb_j;
    vb_i.velocity  = Vec3::Random();
    vb_i.gyro_bias = Vec3::Random();
    vb_i.acc_bias  = Vec3::Random();

    vb_j = vb_i;



    Vec3 g = Vec3::Random().normalized() * 9.81;


    SE3 pose_i = SE3();  // Random::randomSE3();

    SE3 pose_j;



    {
        Imu::Preintegration preint(vb_i.gyro_bias, vb_i.acc_bias);
        preint.IntegrateMidPoint(s);
        auto p_v      = preint.Predict(pose_i, vb_i.velocity, g);
        pose_j        = Sophus::se3_expd(Sophus::Vector6d::Random() * 0.2) * p_v.first;
        vb_j.velocity = p_v.second + Vec3::Random() * 0.01;
    }



    Vec9 res1, res2;
    Matrix<double, 9, 3> J_P_biasa, J_P_biasa_2;
    Matrix<double, 9, 3> J_R_biasg, J_R_biasg_2;


    res1 = Imu::ImuError(s, vb_i, pose_i, vb_j, pose_j, g, &J_P_biasa, &J_R_biasg);

    {
        res2 = EvaluateNumeric(
            [=](auto p) {
                Imu::VelocityAndBias vb = vb_i;
                vb.acc_bias             = p;
                return Imu::ImuError(s, vb, pose_i, vb_j, pose_j, g);
            },
            vb_i.acc_bias, &J_P_biasa_2);
    }

    {
        res2 = EvaluateNumeric(
            [=](auto p) {
                Imu::VelocityAndBias vb = vb_i;
                vb.gyro_bias            = p;
                return Imu::ImuError(s, vb, pose_i, vb_j, pose_j, g);
            },
            vb_i.gyro_bias, &J_R_biasg_2);
    }


    //    J_R_biasg_2.block<3, 3>(3, 0).setZero();
    //    J_R_biasg_2.block<3, 3>(6, 0).setZero();

    ExpectCloseRelative(res1, res2, 1e-5);
    //        ExpectCloseRelative(J_P_biasa, J_P_biasa_2, 1e-5);
    ExpectCloseRelative(J_R_biasg, J_R_biasg_2, 1e-5);
}

}  // namespace Saiga
