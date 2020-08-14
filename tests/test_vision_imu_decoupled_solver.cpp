/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/time/all.h"
#include "saiga/vision/imu/all.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
#include "numeric_derivative.h"
namespace Saiga
{
namespace Imu
{
static DecoupledImuScene::SolverOptions DefaultSolverOptions()
{
    DecoupledImuScene::SolverOptions options;
    options.solve_bias_acc               = false;
    options.solve_bias_gyro              = false;
    options.solve_velocity               = false;
    options.solve_gravity                = false;
    options.solve_scale                  = false;
    options.max_its                      = 10;
    options.bias_recompute_delta_squared = 0;  // 0.1;
                                               //    options.bias_recompute_delta_squared = 0.1;
    return options;
}
static OptimizationOptions DefaultOptOptions()
{
    OptimizationOptions oopts;
    oopts.maxIterations = DefaultSolverOptions().max_its;
    oopts.debugOutput   = false;
    oopts.solverType    = OptimizationOptions::SolverType::Direct;
    return oopts;
}

static DecoupledImuScene MakeScene(DecoupledImuScene::SolverOptions options)
{
    DecoupledImuScene scene;
    scene.MakeRandom(200, 50, 1.0 / 100.0);



    for (auto& s : scene.states)
    {
        if (options.solve_bias_acc) s.velocity_and_bias.acc_bias += Vec3::Random() * 0.1;
        if (options.solve_bias_gyro) s.velocity_and_bias.gyro_bias += Vec3::Random() * 0.1;
        if (options.solve_velocity) s.velocity_and_bias.velocity += Vec3::Random() * 0.1;
    }
    if (options.solve_scale) scene.scale += Random::sampleDouble(-0.2, 0.2);
    if (options.solve_gravity) scene.gravity.R = scene.gravity.R * SO3::exp(Vec3::Random() * 0.1);
    scene.PreintAll();
    return scene;
}


TEST(ImuDecoupledSolver, AllWithConstant)
{
    DecoupledImuScene::SolverOptions options = DefaultSolverOptions();
    options.solve_bias_acc                   = true;
    options.solve_bias_gyro                  = true;
    options.solve_velocity                   = true;
    options.solve_gravity                    = true;
    options.solve_scale                      = true;


    DecoupledImuScene scene = MakeScene(options);

    scene.states[5].constant = true;
    auto test                = scene.states[5].velocity_and_bias;


    {
        auto cpy1 = scene;
        cpy1.PreintAll();
        cpy1.SolveCeres(options, true);

        auto test2 = cpy1.states[5].velocity_and_bias;
        ExpectCloseRelative(test.acc_bias, test2.acc_bias, 1e-20);

        auto cpy2 = scene;
        cpy2.PreintAll();
        cpy2.SolveCeres(options, false);

        auto test3 = cpy2.states[5].velocity_and_bias;
        ExpectCloseRelative(test.acc_bias, test3.acc_bias, 1e-20);

        auto cpy3 = scene;
        cpy3.PreintAll();
        DecoupledImuSolver solver;
        solver.optimizationOptions             = DefaultOptOptions();
        solver.optimizationOptions.debugOutput = false;
        solver.Create(cpy3, options);

        auto r = solver.initAndSolve();

        auto test4 = cpy3.states[5].velocity_and_bias;
        ExpectCloseRelative(test.acc_bias, test4.acc_bias, 1e-20);

        EXPECT_NEAR(cpy1.chi2(), cpy2.chi2(), 1e-1);
        EXPECT_NEAR(cpy2.chi2(), cpy3.chi2(), 1e-1);
    }
}


TEST(ImuDecoupledSolver, All)
{
    DecoupledImuScene::SolverOptions options = DefaultSolverOptions();
    options.solve_bias_acc                   = true;
    options.solve_bias_gyro                  = true;
    options.solve_velocity                   = true;
    options.solve_gravity                    = true;
    options.solve_scale                      = true;


    DecoupledImuScene scene = MakeScene(options);

    {
        auto cpy1 = scene;
        cpy1.PreintAll();
        cpy1.SolveCeres(options, true);

        auto cpy2 = scene;
        cpy2.PreintAll();
        cpy2.SolveCeres(options, false);

        auto cpy3 = scene;
        cpy3.PreintAll();
        DecoupledImuSolver solver;
        solver.optimizationOptions             = DefaultOptOptions();
        solver.optimizationOptions.debugOutput = false;
        solver.Create(cpy3, options);

        auto r = solver.initAndSolve();


        EXPECT_NEAR(cpy1.chi2(), cpy2.chi2(), 1e-3);
        EXPECT_NEAR(cpy2.chi2(), cpy3.chi2(), 1e-3);
    }
}


TEST(ImuDecoupledSolver, Bias)
{
    DecoupledImuScene::SolverOptions options = DefaultSolverOptions();
    options.solve_bias_acc                   = true;
    options.solve_bias_gyro                  = true;
    options.solve_velocity                   = false;
    options.solve_gravity                    = false;
    options.solve_scale                      = false;


    DecoupledImuScene scene = MakeScene(options);

    {
        auto cpy1 = scene;
        cpy1.PreintAll();
        cpy1.SolveCeres(options, true);

        auto cpy2 = scene;
        cpy2.PreintAll();
        cpy2.SolveCeres(options, false);

        auto cpy3 = scene;
        cpy3.PreintAll();
        DecoupledImuSolver solver;
        solver.optimizationOptions             = DefaultOptOptions();
        solver.optimizationOptions.debugOutput = false;
        solver.Create(cpy3, options);

        auto r = solver.initAndSolve();


        EXPECT_NEAR(cpy1.chi2(), cpy2.chi2(), 1e-3);
        EXPECT_NEAR(cpy2.chi2(), cpy3.chi2(), 1e-3);
    }
}

TEST(ImuDecoupledSolver, Scale)
{
    DecoupledImuScene::SolverOptions options = DefaultSolverOptions();
    options.solve_bias_acc                   = false;
    options.solve_bias_gyro                  = false;
    options.solve_velocity                   = false;
    options.solve_gravity                    = false;
    options.solve_scale                      = true;


    DecoupledImuScene scene = MakeScene(options);

    {
        auto cpy1 = scene;
        cpy1.PreintAll();
        cpy1.SolveCeres(options, true);

        auto cpy2 = scene;
        cpy2.PreintAll();
        cpy2.SolveCeres(options, false);

        auto cpy3 = scene;
        cpy3.PreintAll();
        DecoupledImuSolver solver;
        solver.optimizationOptions             = DefaultOptOptions();
        solver.optimizationOptions.debugOutput = false;
        solver.Create(cpy3, options);

        auto r = solver.initAndSolve();


        EXPECT_NEAR(cpy1.chi2(), cpy2.chi2(), 1e-3);
        EXPECT_NEAR(cpy2.chi2(), cpy3.chi2(), 1e-3);
    }
}

TEST(ImuDecoupledSolver, Gravity)
{
    DecoupledImuScene::SolverOptions options = DefaultSolverOptions();
    options.solve_bias_acc                   = false;
    options.solve_bias_gyro                  = false;
    options.solve_velocity                   = false;
    options.solve_gravity                    = true;
    options.solve_scale                      = false;


    DecoupledImuScene scene = MakeScene(options);

    {
        auto cpy1 = scene;
        cpy1.PreintAll();
        cpy1.SolveCeres(options, true);

        auto cpy2 = scene;
        cpy2.PreintAll();
        cpy2.SolveCeres(options, false);

        auto cpy3 = scene;
        cpy3.PreintAll();
        DecoupledImuSolver solver;
        solver.optimizationOptions             = DefaultOptOptions();
        solver.optimizationOptions.debugOutput = false;
        solver.Create(cpy3, options);

        auto r = solver.initAndSolve();


        EXPECT_NEAR(cpy1.chi2(), cpy2.chi2(), 1e-3);
        EXPECT_NEAR(cpy2.chi2(), cpy3.chi2(), 1e-3);
    }
}



TEST(ImuDecoupledSolver, Velocity)
{
    DecoupledImuScene::SolverOptions options = DefaultSolverOptions();
    options.solve_bias_acc                   = false;
    options.solve_bias_gyro                  = false;
    options.solve_velocity                   = true;
    options.solve_gravity                    = false;
    options.solve_scale                      = false;


    DecoupledImuScene scene = MakeScene(options);

    {
        auto cpy1 = scene;
        cpy1.PreintAll();
        cpy1.SolveCeres(options, true);

        auto cpy2 = scene;
        cpy2.PreintAll();
        cpy2.SolveCeres(options, false);

        auto cpy3 = scene;
        cpy3.PreintAll();
        DecoupledImuSolver solver;
        solver.optimizationOptions             = DefaultOptOptions();
        solver.optimizationOptions.debugOutput = false;
        solver.Create(cpy3, options);

        auto r = solver.initAndSolve();


        EXPECT_NEAR(cpy1.chi2(), cpy2.chi2(), 1e-3);
        EXPECT_NEAR(cpy2.chi2(), cpy3.chi2(), 1e-3);
    }
}



}  // namespace Imu
}  // namespace Saiga
