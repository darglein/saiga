/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vision/imu/all.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
#include "numeric_derivative.h"
namespace Saiga
{
namespace Imu
{
TEST(ImuDecoupledSolver, All)
{
    DecoupledImuScene::SolverOptions options;
    options.solve_bias_acc  = true;
    options.solve_bias_gyro = true;
    options.solve_velocity  = true;
    options.solve_gravity   = true;
    options.solve_scale     = true;


    DecoupledImuScene scene;
    scene.MakeRandom(50, 130, 1.0 / 100.0);

    scene.weight_P = 1;
    scene.weight_V = 1;
    scene.weight_R = 1;

    auto ref = scene;
    for (auto& s : scene.states)
    {
        s.velocity_and_bias.acc_bias += Vec3::Random();
        s.velocity_and_bias.gyro_bias += Vec3::Random();
        s.velocity_and_bias.velocity += Vec3::Random();
    }
    scene.scale += Random::sampleDouble(-1, 1);

    {
        auto cpy1 = scene;
        cpy1.SolveCeres(options, true);

        auto cpy2 = scene;
        cpy2.SolveCeres(options, false);

        EXPECT_NEAR(cpy1.chi2(), cpy2.chi2(), 1e-3);
    }
}


TEST(ImuDecoupledSolver, Gravity)
{
    DecoupledImuScene::SolverOptions options;
    options.solve_bias_acc  = false;
    options.solve_bias_gyro = false;
    options.solve_velocity  = false;
    options.solve_gravity   = true;
    options.solve_scale     = false;


    DecoupledImuScene scene;
    scene.MakeRandom(50, 130, 1.0 / 100.0);

    scene.weight_P = 1;
    scene.weight_V = 1;
    scene.weight_R = 1;

    auto ref = scene;


    scene.gravity.R = SO3(Random::randomSE3().so3());


    {
        auto cpy1 = scene;
        cpy1.SolveCeres(options, true);

        auto cpy2 = scene;
        cpy2.SolveCeres(options, false);

        EXPECT_NEAR(cpy1.chi2(), cpy2.chi2(), 1e-3);
    }
}

TEST(ImuDecoupledSolver, Velocity)
{
    return;
    DecoupledImuScene::SolverOptions options;
    options.solve_bias_acc  = false;
    options.solve_bias_gyro = false;
    options.solve_velocity  = true;
    options.solve_gravity   = false;
    options.solve_scale     = false;


    DecoupledImuScene scene;
    scene.MakeRandom(50, 130, 1.0 / 100.0);

    scene.weight_P = 1;
    scene.weight_V = 1;
    scene.weight_R = 1;

    auto ref = scene;

    for (auto& s : scene.states)
    {
        s.velocity_and_bias.velocity += Vec3::Random();
    }


    {
        auto cpy1 = scene;
        cpy1.SolveCeres(options, true);

        auto cpy2 = scene;
        cpy2.SolveCeres(options, false);

        EXPECT_NEAR(cpy1.chi2(), cpy2.chi2(), 1e-3);
    }
}

TEST(ImuDecoupledSolver, Bias)
{
    DecoupledImuScene::SolverOptions options;
    options.solve_bias_acc  = true;
    options.solve_bias_gyro = true;
    options.solve_velocity  = false;
    options.solve_gravity   = false;
    options.solve_scale     = false;


    DecoupledImuScene scene;
    scene.MakeRandom(50, 130, 1.0 / 100.0);

    scene.weight_P = 1;
    scene.weight_V = 1;
    scene.weight_R = 1;

    auto ref = scene;

    for (auto& s : scene.states)
    {
        s.velocity_and_bias.acc_bias += Vec3::Random();
        s.velocity_and_bias.gyro_bias += Vec3::Random();
    }


    {
        auto cpy1 = scene;
        cpy1.SolveCeres(options, true);

        auto cpy2 = scene;
        cpy2.SolveCeres(options, false);

        EXPECT_NEAR(cpy1.chi2(), cpy2.chi2(), 1e-3);
    }
}



TEST(ImuDecoupledSolver, Scale)
{
    DecoupledImuScene::SolverOptions options;
    options.solve_bias_acc  = false;
    options.solve_bias_gyro = false;
    options.solve_velocity  = false;
    options.solve_gravity   = false;
    options.solve_scale     = true;


    DecoupledImuScene scene;
    scene.MakeRandom(50, 130, 1.0 / 100.0);

    scene.weight_P = 1;
    scene.weight_V = 1;
    scene.weight_R = 1;

    auto ref = scene;



    scene.scale += Random::sampleDouble(-1, 1);

    {
        auto cpy1 = scene;
        cpy1.SolveCeres(options, true);

        auto cpy2 = scene;
        cpy2.SolveCeres(options, false);

        EXPECT_NEAR(cpy1.chi2(), cpy2.chi2(), 1e-3);
    }
}

}  // namespace Imu
}  // namespace Saiga
