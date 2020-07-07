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
static std::vector<std::pair<Imu::ImuSequence, SE3>> RandomTrajectory(int N, int K, double gyro_sigma, double dt,
                                                                      const Vec3& bias_gyro)
{
    std::vector<std::pair<Imu::ImuSequence, SE3>> trajectory;


    double t         = 0;
    SE3 initial_pose = Random::randomSE3();
    SE3 current_pose = initial_pose;

    trajectory.push_back({Imu::ImuSequence(), current_pose});
    trajectory.front().first.time_end = t;

    for (int i = 1; i < N; ++i)
    {
        Imu::ImuSequence seq;
        auto inner_its = Random::uniformInt(K / 2, K * 4);
        seq.time_begin = t;
        for (int k = 0; k < inner_its; ++k)
        {
            Imu::Data id;
            id.omega     = Sophus::Vector3d::Random() / 10.0;
            id.timestamp = t;
            current_pose.setQuaternion(current_pose.unit_quaternion() *
                                       Sophus::SO3<double>::exp(id.omega * dt).unit_quaternion());

            Vec3 noise = Vec3(Random::gaussRand(0, gyro_sigma), Random::gaussRand(0, gyro_sigma),
                              Random::gaussRand(0, gyro_sigma));
            id.omega += bias_gyro + noise;

            seq.data.push_back(id);
            t += dt;
        }
        seq.time_end = t;
        trajectory.push_back({seq, current_pose});
    }
    return trajectory;
}


TEST(Imu, SolveGyroBias)
{
    auto test_trajectory = [](auto trajectory, auto bias) -> double {
        std::vector<std::tuple<const Imu::ImuSequence*, Quat, Quat>> solver_data;
        for (int i = 1; i < trajectory.size(); ++i)
        {
            solver_data.push_back({&trajectory[i].first, trajectory[i - 1].second.unit_quaternion(),
                                   trajectory[i].second.unit_quaternion()});
        }
        Vec3 bias2 = Imu::SolveGlobalGyroBias(solver_data, 2);
        return (bias - bias2).norm();
    };


    // With added noise the error is expected to be in the same range as sigma

    for (int i = 0; i < 5; ++i)
    {
        double gyro_sigma = 1e-8;
        Vec3 bias         = Vec3::Random() * 0.1;
        auto trajectory   = RandomTrajectory(10, 50, gyro_sigma, 0.05, bias);
        EXPECT_LE(test_trajectory(trajectory, bias), gyro_sigma);
    }

    for (int i = 0; i < 5; ++i)
    {
        double gyro_sigma = 1e-4;
        Vec3 bias         = Vec3::Random() * 0.1;
        auto trajectory   = RandomTrajectory(10, 50, gyro_sigma, 0.05, bias);
        EXPECT_LE(test_trajectory(trajectory, bias), gyro_sigma);
    }

    for (int i = 0; i < 5; ++i)
    {
        double gyro_sigma = 1e-2;
        Vec3 bias         = Vec3::Random() * 0.1;
        auto trajectory   = RandomTrajectory(10, 50, gyro_sigma, 0.05, bias);
        EXPECT_LE(test_trajectory(trajectory, bias), gyro_sigma);
    }
}

}  // namespace Saiga
