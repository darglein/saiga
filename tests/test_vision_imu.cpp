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

static std::vector<std::pair<Imu::ImuSequence, SE3>> RandomTrajectory2(int N, int K, double sigma_gyro,
                                                                       double sigma_acc, const Vec3& bias_gyro,
                                                                       const Vec3& bias_acc, const Vec3& g, double dt)
{
    auto data = GenerateRandomSequence(N, K, dt);

    std::vector<std::pair<Imu::ImuSequence, SE3>> result;
    SE3 pose = Random::randomSE3();
    Vec3 v   = Vec3::Random();

    result.emplace_back(data.front(), pose);

    // Integrate to get actual poses
    for (int i = 1; i < N; ++i)
    {
        Imu::Preintegration preint;
        //        preint.IntegrateMidPoint(data[i]);
        preint.IntegrateForward(data[i], true);

        auto pose_v = preint.Predict(pose, v, Vec3::Zero());

        pose = pose_v.first;
        v    = pose_v.second;
        result.emplace_back(data[i], pose);
    }

    // Add bias+noise to measurements
    for (int i = 1; i < N; ++i)
    {
        result[i].first.AddNoise(sigma_gyro, sigma_acc);
        result[i].first.AddBias(bias_gyro, bias_acc);
        result[i].first.AddGravity(bias_gyro, result[i - 1].second.so3(), g);
    }
#if 0

    for (int i = 1; i < N; ++i)
    {
        Vec3 mean_gravity = Vec3::Zero();
        Imu::Preintegration preint;

        Quat R     = result[i - 1].second.unit_quaternion();
        auto& data = result[i].first.data;

        mean_gravity += R * data.front().acceleration;
        //    data[0].acceleration += R * g;

        for (int i = 1; i < data.size(); ++i)
        {
            double dt = data[i].timestamp - data[i - 1].timestamp;
            preint.Add(data[i - 1].omega, Vec3::Zero(), dt);
            mean_gravity += (R * preint.delta_R) * data.front().acceleration;
        }
        mean_gravity /= data.size();
        std::cout << mean_gravity.transpose() << std::endl;
    }
#endif


    return result;
}


TEST(Imu, SaveLoad)
{
    auto s = GenerateRandomSequence(10, 5, 0.1);


    for (auto seq : s)
    {
        if (!std::isfinite(seq.time_begin)) continue;

        seq.Save("seq.txt");

        Imu::ImuSequence seq2;
        seq2.Load("seq.txt");

        EXPECT_EQ(seq.time_begin, seq2.time_begin);
        EXPECT_EQ(seq.time_end, seq2.time_end);
    }
}

TEST(Imu, SolveGyroBias)
{
    //    return;
    auto test_trajectory = [](auto trajectory, auto bias) -> double {
        std::vector<Imu::ImuPosePair> solver_data;
        std::vector<Imu::Preintegration> preints(trajectory.size());



        for (int i = 1; i < trajectory.size(); ++i)
        {
            preints[i] = Imu::Preintegration(Vec3::Zero());
            preints[i].IntegrateForward(trajectory[i].first, true);
            //            preints[i].IntegrateMidPoint(trajectory[i].first);

            Imu::ImuPosePair ipp;
            ipp.pose1     = &trajectory[i - 1].second;
            ipp.pose2     = &trajectory[i].second;
            ipp.preint_12 = &preints[i];

            solver_data.push_back(ipp);
            //            solver_data.push_back({&trajectory[i].first, trajectory[i - 1].second.unit_quaternion(),
            //                                   trajectory[i].second.unit_quaternion()});
        }
        auto bias_error = Imu::SolveGlobalGyroBias(solver_data);
        std::cout << bias_error.second << std::endl;
        return (bias - bias_error.first).norm();
    };


    double sigma_acc = 0.0;
    Vec3 bias_acc    = Vec3::Zero();
    Vec3 g           = Vec3::Zero();

    double dt = 1.0 / 200;
    // With added noise the error is expected to be in the same range as sigma

    for (int i = 0; i < 5; ++i)
    {
        double gyro_sigma = 1e-6;
        Vec3 bias         = Vec3::Random() * 0.1;
        auto trajectory   = RandomTrajectory2(10, 50, gyro_sigma, sigma_acc, bias, bias_acc, g, dt);
        EXPECT_LE(test_trajectory(trajectory, bias), 1e-3);
    }

    for (int i = 0; i < 5; ++i)
    {
        double gyro_sigma = 1e-4;
        Vec3 bias         = Vec3::Random() * 0.1;
        auto trajectory   = RandomTrajectory2(10, 50, gyro_sigma, sigma_acc, bias, bias_acc, g, dt);
        EXPECT_LE(test_trajectory(trajectory, bias), 1e-3);
    }

    for (int i = 0; i < 5; ++i)
    {
        double gyro_sigma = 1e-2;
        Vec3 bias         = Vec3::Random() * 0.1;
        auto trajectory   = RandomTrajectory2(10, 50, gyro_sigma, sigma_acc, bias, bias_acc, g, dt);
        EXPECT_LE(test_trajectory(trajectory, bias), 1e-2);
    }
}


TEST(Imu, SolveGravityScale)
{
    auto test_trajectory = [](auto trajectory, auto acc_bias) -> double {
        int N = trajectory.size();
        std::vector<Imu::ImuPoseTriplet> solver_data;
        std::vector<Imu::Preintegration> preintegration(N);

        for (int i = 1; i < N; ++i)
        {
            preintegration[i] = Imu::Preintegration(Vec3::Zero(), Vec3::Zero());
            //            preintegration[i].IntegrateMidPoint(trajectory[i].first);
            preintegration[i].IntegrateForward(trajectory[i].first, true);
        }


        for (int i = 0; i < trajectory.size() - 2; ++i)
        {
            Imu::ImuPoseTriplet ipt;
            ipt.pose1     = &trajectory[i].second;
            ipt.pose2     = &trajectory[i + 1].second;
            ipt.pose3     = &trajectory[i + 2].second;
            ipt.preint_12 = &(preintegration[i + 1]);
            ipt.preint_23 = &(preintegration[i + 2]);

            solver_data.push_back(ipt);
        }
        //        Vec3 bias2 = Imu::SolveGlobalGyroBias(solver_data, 2);
        Vec3 gravity = Vec3(0, 0, 1);
        double predicted_scale;
        std::tie(predicted_scale, gravity) = Imu::SolveScaleGravityLinear(solver_data, SE3());

        std::cout << "s = " << predicted_scale << " g = " << gravity.transpose() << std::endl;

        double error;
        double scale_delta;
        Vec3 bias_delta;
        std::tie(scale_delta, gravity, bias_delta, error) =
            Imu::SolveScaleGravityBiasLinear(solver_data, gravity, SE3());

        double scale = scale_delta;

        std::cout << "s = " << scale << " g = " << gravity.transpose() << std::endl;
        std::cout << "bias = " << bias_delta.transpose() << std::endl;
        std::cout << std::endl;

        return scale;
    };


    double dt = 1.0 / 1000;

    double sigma_acc  = 0.001;
    double sigma_gyro = 0.0;
    //    Vec3 bias_acc     = Vec3::Zero();
    Vec3 bias_acc  = Vec3(0.02, 0.1, 0.1);
    Vec3 bias_gyro = Vec3::Zero();

    Vec3 g = Vec3(-0.195012, 9.09673, 3.6671);


    // With added noise the error is expected to be in the same range as sigma

    for (int i = 0; i < 5; ++i)
    {
        auto trajectory = RandomTrajectory2(10, 100, sigma_gyro, sigma_acc, bias_gyro, bias_acc, g, dt);

        double target_scale = Random::sampleDouble(0.5, 5.0);
        for (auto& p : trajectory)
        {
            p.second.translation() *= target_scale;
        }

        std::cout << "target scale " << (1.0 / target_scale) << std::endl;
        auto scale = test_trajectory(trajectory, bias_acc);

        EXPECT_NEAR(scale, (1.0 / target_scale), 0.1);
    }
}

}  // namespace Saiga
