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
              const SE3& pose_j, const Vec3& g, Matrix<double, 9, 3>* J_biasa = nullptr,
              Matrix<double, 9, 3>* J_biasg = nullptr, Matrix<double, 9, 3>* J_v1 = nullptr,
              Matrix<double, 9, 3>* J_v2 = nullptr, Matrix<double, 9, 6>* J_p1 = nullptr,
              Matrix<double, 9, 6>* J_p2 = nullptr)
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


    SE3 rel_ji = pose_i.inverse() * pose_j;

    Vec3 vel_Vij = pose_i.inverse().so3() * Vj;


    Preintegration preint(vb_i);
    preint.IntegrateMidPoint(s);

    SE3 preint_Pj;
    Vec3 preint_Vj;
    std::tie(preint_Pj, preint_Vj) = preint.Predict(pose_i, Vi, g);


    SE3 preint_ji   = pose_i.inverse() * preint_Pj;
    Vec3 preint_Vji = pose_i.inverse().so3() * preint_Vj;

    //    double dTij = preint.delta_t;
    //    double dT2  = dTij * dTij;

    //    Vec3 dPij = preint.delta_x;
    //    Vec3 dVij = preint.delta_v;

    //    auto prediction = preint.Predict(pose_i, vb_i.velocity, g);


    // residual error of Delta Position measurement
    //    Vec3 rPij = RiT * (-Pj + Pi + Vi * dTij + 0.5 * g * dT2) + dPij;
    //    Vec3 rPij = RiT * (Pi + Vi * dTij + 0.5 * g * dT2 - Pj) + dPij;
    //    Vec3 rPij = RiT * (Pi + Vi * dTij + 0.5 * g * dT2) + dPij - RiT * Pj;
    Vec3 rPij = preint_ji.translation() - rel_ji.translation();  // RiT * Pj;
    // - (dPij + M.getJPBiasg() * dBgi + M.getJPBiasa() * dBai);  // this line includes correction term of bias change.
    // residual error of Delta Velocity measurement
    //    Vec3 rVij = RiT * (Vi + g * dTij) + dVij - RiT * Vj;
    Vec3 rVij = preint_Vji - vel_Vij;

    //= (dRij).inverse() * RiT * Rj;
    Sophus::SO3 rRij = preint_ji.so3().inverse() * rel_ji.so3();
    Vec3 rPhiij      = rRij.log();


    residual.segment<3>(0) = rPij;    // position error
    residual.segment<3>(3) = rVij;    // velocity error
    residual.segment<3>(6) = rPhiij;  // rotation phi error

    if (J_biasg)
    {
        Mat3 JrInv_rPhi;
        Sophus::rightJacobianInvSO3(rPhiij, JrInv_rPhi);

        Mat3 ExprPhiijTrans = Sophus::SO3d::exp(rPhiij).inverse().matrix();

        //        Mat3 JrBiasGCorr = Sophus::rightJacobianSO3(J_rPhi_dbg * dBgi);


        //          Matrix3d J_rPhi_dbg = M.getJRBiasg();

        J_biasg->block<3, 3>(0, 0) = preint.J_P_Biasg;
        J_biasg->block<3, 3>(3, 0) = preint.J_V_Biasg;

        J_biasg->block<3, 3>(6, 0) = -JrInv_rPhi * ExprPhiijTrans /** JrBiasGCorr*/ * preint.J_R_Biasg;
        //        J_R_biasg -
        //        std::cout << preint.J_P_Biasg << std::endl;
        //        std::cout << preint.J_V_Biasg << std::endl;
    }

    if (J_biasa)
    {
        J_biasa->block<3, 3>(0, 0) = preint.J_P_Biasa;
        J_biasa->block<3, 3>(3, 0) = preint.J_V_Biasa;
        J_biasa->block<3, 3>(6, 0).setZero();
    }

    if (J_v1)
    {
        J_v1->block<3, 3>(0, 0) = RiT.matrix() * preint.delta_t;
        J_v1->block<3, 3>(3, 0) = RiT.matrix();
        J_v1->block<3, 3>(6, 0).setZero();
    }

    if (J_v2)
    {
        J_v2->block<3, 3>(0, 0).setZero();
        J_v2->block<3, 3>(3, 0) = -RiT.matrix();
        J_v2->block<3, 3>(6, 0).setZero();
    }

    if (J_p1)
    {
        Mat3 JrInv_rPhi;
        Sophus::rightJacobianInvSO3(rPhiij, JrInv_rPhi);

        J_p1->block<3, 3>(0, 0).setIdentity();
        J_p1->block<3, 3>(3, 0).setZero();
        J_p1->block<3, 3>(6, 0).setZero();

        J_p1->block<3, 3>(0, 3) =
            skew(RiT * (Pi + Vi * preint.delta_t + 0.5 * g * preint.delta_t * preint.delta_t - Pj));
        J_p1->block<3, 3>(3, 3) = skew(RiT * (Vi + g * preint.delta_t - Vj));
        J_p1->block<3, 3>(6, 3) = -JrInv_rPhi * RjT.matrix() * Ri.matrix();
    }

    if (J_p2)
    {
        Mat3 JrInv_rPhi;
        Sophus::rightJacobianInvSO3(rPhiij, JrInv_rPhi);

        J_p2->block<3, 3>(0, 0) = -(RiT * Rj).matrix();
        J_p2->block<3, 3>(3, 0).setZero();
        J_p2->block<3, 3>(6, 0).setZero();

        J_p2->block<3, 3>(0, 3).setZero();
        J_p2->block<3, 3>(3, 3).setZero();
        J_p2->block<3, 3>(6, 3) = JrInv_rPhi;
    }

    return residual;
}
}  // namespace Imu


struct ImuDerivTest
{
    ImuDerivTest()
    {
        // Random::setSeed(95665);
        s              = GenerateRandomSequence(10, 5, 0.1)[1];
        vb_i.velocity  = Vec3::Random();
        vb_i.gyro_bias = Vec3::Random();
        vb_i.acc_bias  = Vec3::Random();
        vb_j           = vb_i;
        g              = Vec3::Random().normalized() * 9.81;
        pose_i         = Random::randomSE3();
        {
            Imu::Preintegration preint(vb_i.gyro_bias, vb_i.acc_bias);
            preint.IntegrateMidPoint(s);
            auto p_v      = preint.Predict(pose_i, vb_i.velocity, g);
            pose_j        = Sophus::se3_expd(Sophus::Vector6d::Random() * 0.2) * p_v.first;
            vb_j.velocity = p_v.second + Vec3::Random() * 0.01;
        }
    }
    Imu::ImuSequence s;
    Vec3 g;
    SE3 pose_i, pose_j;
    Imu::VelocityAndBias vb_i, vb_j;
};

TEST(ImuDerivatives, BiasGyro)
{
    ImuDerivTest data;

    Vec9 res1, res2;
    Matrix<double, 9, 3> J_biasg, J_biasg_numeric;

    res1 = Imu::ImuError(data.s, data.vb_i, data.pose_i, data.vb_j, data.pose_j, data.g, nullptr, &J_biasg);
    {
        res2 = EvaluateNumeric(
            [=](auto p) {
                Imu::VelocityAndBias vb = data.vb_i;
                vb.gyro_bias            = p;
                return Imu::ImuError(data.s, vb, data.pose_i, data.vb_j, data.pose_j, data.g);
            },
            data.vb_i.gyro_bias, &J_biasg_numeric);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_biasg, J_biasg_numeric, 1e-5);
}


TEST(ImuDerivatives, BiasAcc)
{
    ImuDerivTest data;

    Vec9 res1, res2;
    Matrix<double, 9, 3> J_biasa, J_biasa_numeric;


    res1 = Imu::ImuError(data.s, data.vb_i, data.pose_i, data.vb_j, data.pose_j, data.g, &J_biasa);

    {
        res2 = EvaluateNumeric(
            [=](auto p) {
                Imu::VelocityAndBias vb = data.vb_i;
                vb.acc_bias             = p;
                return Imu::ImuError(data.s, vb, data.pose_i, data.vb_j, data.pose_j, data.g);
            },
            data.vb_i.acc_bias, &J_biasa_numeric);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_biasa, J_biasa_numeric, 1e-5);
}


TEST(ImuDerivatives, Velocity)
{
    ImuDerivTest data;

    Vec9 res1, res2;
    Matrix<double, 9, 3> J_v1, J_v2;
    Matrix<double, 9, 3> J_v1_numeric, J_v2_numeric;


    res1 =
        Imu::ImuError(data.s, data.vb_i, data.pose_i, data.vb_j, data.pose_j, data.g, nullptr, nullptr, &J_v1, &J_v2);

    {
        res2 = EvaluateNumeric(
            [=](auto p) {
                Imu::VelocityAndBias vb = data.vb_i;
                vb.velocity             = p;
                return Imu::ImuError(data.s, vb, data.pose_i, data.vb_j, data.pose_j, data.g);
            },
            data.vb_i.velocity, &J_v1_numeric);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_v1, J_v1_numeric, 1e-5);
    {
        res2 = EvaluateNumeric(
            [=](auto p) {
                Imu::VelocityAndBias vb = data.vb_j;
                vb.velocity             = p;
                return Imu::ImuError(data.s, data.vb_i, data.pose_i, vb, data.pose_j, data.g);
            },
            data.vb_j.velocity, &J_v2_numeric);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_v2, J_v2_numeric, 1e-5);
}


TEST(ImuDerivatives, Pose)
{
    ImuDerivTest data;

    Vec9 res1, res2;
    Matrix<double, 9, 6> J_p1, J_p2;
    Matrix<double, 9, 6> J_p1_numeric, J_p2_numeric;


    res1 = Imu::ImuError(data.s, data.vb_i, data.pose_i, data.vb_j, data.pose_j, data.g, nullptr, nullptr, nullptr,
                         nullptr, &J_p1, &J_p2);

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto pose_w_i_new = data.pose_i * Sophus::se3_expd(p);
                //                auto pose_w_i_new = data.pose_i * Sophus::SE3d::exp(p);

                return Imu::ImuError(data.s, data.vb_i, pose_w_i_new, data.vb_j, data.pose_j, data.g);
            },
            eps, &J_p1_numeric);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_p1, J_p1_numeric, 1e-5);


    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto pose_w_j_new = data.pose_j * Sophus::se3_expd(p);
                //                auto pose_w_i_new = data.pose_i * Sophus::SE3d::exp(p);

                return Imu::ImuError(data.s, data.vb_i, data.pose_i, data.vb_j, pose_w_j_new, data.g);
            },
            eps, &J_p2_numeric);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_p2, J_p2_numeric, 1e-5);
}

}  // namespace Saiga
