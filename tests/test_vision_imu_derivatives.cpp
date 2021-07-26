/**
 * Copyright (c) 2021 Darius Rückert
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
#ifndef WIN32

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



struct ImuDerivTest
{
    ImuDerivTest()
    {
        // Random::setSeed(95665);
        s = GenerateRandomSequence(10, 5, 0.1)[1];

        vb_i.velocity  = Vec3::Random();
        vb_i.gyro_bias = Vec3::Random();
        vb_i.acc_bias  = Vec3::Random();

        vb_j.velocity  = Vec3::Random();
        vb_j.gyro_bias = Vec3::Random();
        vb_j.acc_bias  = Vec3::Random();



        g.R    = Random::randomSE3().so3();
        pose_i = Random::randomSE3();
        scale  = 10;  // Random::sampleDouble(0.5, 2);
        {
            Imu::Preintegration preint(vb_i.gyro_bias, vb_i.acc_bias);
            preint.IntegrateMidPoint(s, true);
            auto p_v      = preint.Predict(pose_i, vb_i.velocity, g.Get());
            pose_j        = Sophus::se3_expd(Sophus::Vector6d::Random() * 0.2) * p_v.first;
            vb_j.velocity = p_v.second + Vec3::Random() * 0.01;
        }

        preint = Imu::Preintegration(vb_i);
        preint.IntegrateMidPoint(s, true);

        //        delta_bias_i.acc_bias  = Vec3::Random() * 0.01;
        //        delta_bias_i.gyro_bias = Vec3::Random() * 0.01;

        //        delta_bias_i.acc_bias  = Vec3::Ones() * 1.1;
        //        delta_bias_i.gyro_bias = Vec3::Ones() * 1.1;

        weight_pvr = 2 * Vec3::Ones() + Vec3::Random();
    }
    double scale = 1;
    Imu::Preintegration preint;
    Imu::ImuSequence s;
    Imu::Gravity g;
    SE3 pose_i, pose_j;
    Imu::VelocityAndBias vb_i, vb_j;
    Vec3 weight_pvr;


    Imu::VelocityAndBias delta_bias_i, delta_bias_j;
};


TEST(ImuDerivatives, BiasChange)
{
    ImuDerivTest data;

    data.delta_bias_i.acc_bias  = Vec3::Random() * 0.5;
    data.delta_bias_i.gyro_bias = Vec3::Random() * 0.5;

    data.delta_bias_j.acc_bias  = Vec3::Random() * 0.5;
    data.delta_bias_j.gyro_bias = Vec3::Random() * 0.5;

    Vec6 res1, res2;
    Matrix<double, 6, 6> J_a_g_i, J_a_g_i_numeric;
    Matrix<double, 6, 6> J_a_g_j, J_a_g_j_numeric;

    double w_a = 5;
    double w_g = 3;
    res1 = data.preint.BiasChangeError(data.vb_i, data.delta_bias_i, data.vb_j, data.delta_bias_j, w_a, w_g, &J_a_g_i,
                                       &J_a_g_j);

    {
        Vec6 bias_ag;
        bias_ag.segment<3>(0) = data.vb_i.acc_bias;
        bias_ag.segment<3>(3) = data.vb_i.gyro_bias;
        res2                  = EvaluateNumeric(
            [=](auto p) {
                Imu::VelocityAndBias vb_i = data.vb_i;

                vb_i.acc_bias  = p.template segment<3>(0);
                vb_i.gyro_bias = p.template segment<3>(3);

                vb_i.acc_bias += data.delta_bias_i.acc_bias;
                vb_i.gyro_bias += data.delta_bias_i.gyro_bias;

                Imu::VelocityAndBias vb_j = data.vb_j;
                vb_j.acc_bias += data.delta_bias_j.acc_bias;
                vb_j.gyro_bias += data.delta_bias_j.gyro_bias;

                Imu::VelocityAndBias empty_delta;

                Imu::Preintegration preint(vb_i);
                preint.IntegrateMidPoint(data.s, true);
                return data.preint.BiasChangeError(vb_i, empty_delta, vb_j, empty_delta, w_a, w_g);
            },
            bias_ag, &J_a_g_i_numeric);
    }

    {
        Vec6 bias_ag;
        bias_ag.segment<3>(0) = data.vb_j.acc_bias;
        bias_ag.segment<3>(3) = data.vb_j.gyro_bias;
        res2                  = EvaluateNumeric(
            [=](auto p) {
                Imu::VelocityAndBias vb_i = data.vb_i;


                vb_i.acc_bias += data.delta_bias_i.acc_bias;
                vb_i.gyro_bias += data.delta_bias_i.gyro_bias;

                Imu::VelocityAndBias vb_j = data.vb_j;

                vb_j.acc_bias  = p.template segment<3>(0);
                vb_j.gyro_bias = p.template segment<3>(3);

                vb_j.acc_bias += data.delta_bias_j.acc_bias;
                vb_j.gyro_bias += data.delta_bias_j.gyro_bias;

                Imu::VelocityAndBias empty_delta;

                Imu::Preintegration preint(vb_i);
                preint.IntegrateMidPoint(data.s, true);
                return data.preint.BiasChangeError(vb_i, empty_delta, vb_j, empty_delta, w_a, w_g);
            },
            bias_ag, &J_a_g_j_numeric);
    }
    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_a_g_i, J_a_g_i_numeric, 1e-5);
    ExpectCloseRelative(J_a_g_j, J_a_g_j_numeric, 1e-5);
}

TEST(ImuDerivatives, BiasGyro)
{
    ImuDerivTest data;
    //    data.delta_bias_i.acc_bias  = Vec3::Ones() * 0.5;
    //    data.delta_bias_i.gyro_bias = Vec3::Ones() * 0.5;

    Vec9 res1, res2;
    Matrix<double, 9, 3> J_biasg, J_biasg_numeric;

    res1 = data.preint.ImuError(data.delta_bias_i, data.vb_i.velocity, data.pose_i, data.vb_j.velocity, data.pose_j,
                                data.g, data.scale, data.weight_pvr, nullptr, &J_biasg);
    {
        res2 = EvaluateNumeric(
            [=](auto p) {
                Imu::VelocityAndBias vb = data.vb_i;

                vb.gyro_bias = p;
                vb.gyro_bias += data.delta_bias_i.gyro_bias;
                vb.acc_bias += data.delta_bias_i.acc_bias;

                Imu::VelocityAndBias empty_delta;

                Imu::Preintegration preint(vb);
                preint.IntegrateMidPoint(data.s, true);
                return preint.ImuError(empty_delta, vb.velocity, data.pose_i, data.vb_j.velocity, data.pose_j, data.g,
                                       data.scale, data.weight_pvr);
            },
            data.vb_i.gyro_bias, &J_biasg_numeric);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_biasg, J_biasg_numeric, 1e-5);
}

TEST(ImuDerivatives, BiasGyroDelta)
{
    ImuDerivTest data;
    data.delta_bias_i.acc_bias  = Vec3::Random() * 0.2;
    data.delta_bias_i.gyro_bias = Vec3::Random() * 0.2;

    Imu::VelocityAndBias empty_delta;

    Vec9 res1, res2, res3;
    Matrix<double, 9, 3> J_biasg_reference, J_biasg_delta, J_biasg_zero_delta;

    {
        Imu::VelocityAndBias vb = data.vb_i;
        vb.gyro_bias += data.delta_bias_i.gyro_bias;
        vb.acc_bias += data.delta_bias_i.acc_bias;

        // Compute reference by preintegration
        Imu::Preintegration preint2(vb);
        preint2.IntegrateMidPoint(data.s, true);


        res1 = preint2.ImuError(empty_delta, data.vb_i.velocity, data.pose_i, data.vb_j.velocity, data.pose_j, data.g,
                                data.scale, data.weight_pvr, nullptr, &J_biasg_reference);
    }

    res2 = data.preint.ImuError(empty_delta, data.vb_i.velocity, data.pose_i, data.vb_j.velocity, data.pose_j, data.g,
                                data.scale, data.weight_pvr, nullptr, &J_biasg_zero_delta);


    res3 = data.preint.ImuError(data.delta_bias_i, data.vb_i.velocity, data.pose_i, data.vb_j.velocity, data.pose_j,
                                data.g, data.scale, data.weight_pvr, nullptr, &J_biasg_delta);
}

TEST(ImuDerivatives, BiasAcc)
{
    ImuDerivTest data;
    Vec9 res1, res2;
    Matrix<double, 9, 3> J_biasa, J_biasa_numeric;


    res1 = data.preint.ImuError(data.delta_bias_i, data.vb_i.velocity, data.pose_i, data.vb_j.velocity, data.pose_j,
                                data.g, data.scale, data.weight_pvr, &J_biasa);

    {
        res2 = EvaluateNumeric(
            [=](auto p) {
                Imu::VelocityAndBias vb = data.vb_i;
                vb.acc_bias             = p;
                vb.gyro_bias += data.delta_bias_i.gyro_bias;
                vb.acc_bias += data.delta_bias_i.acc_bias;

                Imu::VelocityAndBias empty_delta;
                Imu::Preintegration preint(vb);
                preint.IntegrateMidPoint(data.s, true);

                return preint.ImuError(empty_delta, vb.velocity, data.pose_i, data.vb_j.velocity, data.pose_j, data.g,
                                       data.scale, data.weight_pvr);
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


    res1 = data.preint.ImuError(data.delta_bias_i, data.vb_i.velocity, data.pose_i, data.vb_j.velocity, data.pose_j,
                                data.g, data.scale, data.weight_pvr, nullptr, nullptr, &J_v1, &J_v2);

    {
        res2 = EvaluateNumeric(
            [=](auto p) {
                Imu::VelocityAndBias vb = data.vb_i;
                vb.velocity             = p;
                return data.preint.ImuError(data.delta_bias_i, vb.velocity, data.pose_i, data.vb_j.velocity,
                                            data.pose_j, data.g, data.scale, data.weight_pvr);
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
                return data.preint.ImuError(data.delta_bias_i, data.vb_i.velocity, data.pose_i, vb.velocity,
                                            data.pose_j, data.g, data.scale, data.weight_pvr);
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


    res1 = data.preint.ImuError(data.delta_bias_i, data.vb_i.velocity, data.pose_i, data.vb_j.velocity, data.pose_j,
                                data.g, data.scale, data.weight_pvr, nullptr, nullptr, nullptr, nullptr, &J_p1, &J_p2);

    {
        Vec6 eps = Vec6::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                auto pose_w_i_new = data.pose_i * Sophus::se3_expd(p);
                //                auto pose_w_i_new = data.pose_i * Sophus::SE3d::exp(p);

                return data.preint.ImuError(data.delta_bias_i, data.vb_i.velocity, pose_w_i_new, data.vb_j.velocity,
                                            data.pose_j, data.g, data.scale, data.weight_pvr);
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

                return data.preint.ImuError(data.delta_bias_i, data.vb_i.velocity, data.pose_i, data.vb_j.velocity,
                                            pose_w_j_new, data.g, data.scale, data.weight_pvr);
            },
            eps, &J_p2_numeric);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_p2, J_p2_numeric, 1e-5);
}


TEST(ImuDerivatives, Scale)
{
    ImuDerivTest data;

    Vec9 res1, res2;
    Matrix<double, 9, 1> J_scale, J_scale_numeric;


    res1 = data.preint.ImuError(data.delta_bias_i, data.vb_i.velocity, data.pose_i, data.vb_j.velocity, data.pose_j,
                                data.g, data.scale, data.weight_pvr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                nullptr, &J_scale);

    {
        Matrix<double, 1, 1> scale_m;
        scale_m(0, 0) = data.scale;

        res2 = EvaluateNumeric(
            [=](auto p) {
                return data.preint.ImuError(data.delta_bias_i, data.vb_i.velocity, data.pose_i, data.vb_j.velocity,
                                            data.pose_j, data.g, p(0, 0), data.weight_pvr);
            },
            scale_m, &J_scale_numeric);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_scale, J_scale_numeric, 1e-5);
}



TEST(ImuDerivatives, Gravity)
{
    ImuDerivTest data;

    Vec9 res1, res2;
    Matrix<double, 9, 3> J_g;
    Matrix<double, 9, 3> J_g_numeric;


    res1 = data.preint.ImuError(data.delta_bias_i, data.vb_i.velocity, data.pose_i, data.vb_j.velocity, data.pose_j,
                                data.g, data.scale, data.weight_pvr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                nullptr, nullptr, &J_g);

    {
        Vec3 eps = Vec3::Zero();
        res2     = EvaluateNumeric(
            [=](auto p) {
                Imu::Gravity g = data.g;
                //                g.R            = g.R * Sophus::SO3d::exp(p);
                g.R = Sophus::SO3d::exp(p) * g.R;

                return data.preint.ImuError(data.delta_bias_i, data.vb_i.velocity, data.pose_i, data.vb_j.velocity,
                                            data.pose_j, g, data.scale, data.weight_pvr);
            },
            eps, &J_g_numeric);
    }

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_g, J_g_numeric, 1e-5);
}

#endif

}  // namespace Saiga
