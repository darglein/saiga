/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/time/all.h"
#include "saiga/core/util/table.h"
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
    options.solver_flags                 = 0;
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

static DecoupledImuScene MakeScene(DecoupledImuScene::SolverOptions options, int N = 100, int K = 50)
{
    DecoupledImuScene scene;
    scene.MakeRandom(N, K, 1.0 / 100.0);



    for (auto& s : scene.states)
    {
        if (options.solver_flags & IMU_SOLVE_BA) s.velocity_and_bias.acc_bias += Vec3::Random() * 0.1;
        if (options.solver_flags & IMU_SOLVE_BG) s.velocity_and_bias.gyro_bias += Vec3::Random() * 0.1;
        if (options.solver_flags & IMU_SOLVE_VELOCITY) s.velocity_and_bias.velocity += Vec3::Random() * 0.1;
    }
    if (options.solver_flags & IMU_SOLVE_SCALE) scene.scale += Random::sampleDouble(-0.2, 0.2);
    if (options.solver_flags & IMU_SOLVE_GRAVITY) scene.gravity.R = scene.gravity.R * SO3::exp(Vec3::Random() * 0.1);
    scene.PreintAll();
    scene.SanityCheck();
    return scene;
}


TEST(ImuDecoupledSolver, ReuseSolver)
{
    DecoupledImuScene::SolverOptions options = DefaultSolverOptions();
    options.solver_flags = IMU_SOLVE_BA | IMU_SOLVE_BG | IMU_SOLVE_VELOCITY | IMU_SOLVE_GRAVITY | IMU_SOLVE_SCALE;


    DecoupledImuSolver solver;
    solver.optimizationOptions               = DefaultOptOptions();
    solver.optimizationOptions.maxIterations = 5;
    solver.optimizationOptions.debugOutput   = false;

    {
        auto scene = MakeScene(options, 2, 10);
        scene.PreintAll();
        solver.Create(scene, options);
        solver.initAndSolve();
        EXPECT_LE(scene.chi2(), 0.1);


        scene = MakeScene(options, 3, 100);
        scene.PreintAll();
        solver.Create(scene, options);
        solver.initAndSolve();
        EXPECT_LE(scene.chi2(), 0.1);
    }
}



TEST(ImuDecoupledSolver, AllWithConstant)
{
    DecoupledImuScene::SolverOptions options = DefaultSolverOptions();
    options.solver_flags = IMU_SOLVE_BA | IMU_SOLVE_BG | IMU_SOLVE_VELOCITY | IMU_SOLVE_GRAVITY | IMU_SOLVE_SCALE;


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
    options.solver_flags = IMU_SOLVE_BA | IMU_SOLVE_BG | IMU_SOLVE_VELOCITY | IMU_SOLVE_GRAVITY | IMU_SOLVE_SCALE;


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
    options.solver_flags                     = IMU_SOLVE_BA | IMU_SOLVE_BG;


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
    options.solver_flags                     = IMU_SOLVE_SCALE;


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
    options.solver_flags                     = IMU_SOLVE_GRAVITY;


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
    options.solver_flags                     = IMU_SOLVE_VELOCITY;


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



TEST(ImuDecoupledSolver, All_Benchmark)
{
    DecoupledImuScene::SolverOptions options = DefaultSolverOptions();
    options.solver_flags = IMU_SOLVE_BA | IMU_SOLVE_BG | IMU_SOLVE_VELOCITY | IMU_SOLVE_GRAVITY | IMU_SOLVE_SCALE;
    options.max_its      = 5;
    options.bias_recompute_delta_squared = 0.01;


    Table tab({20, 5, 5, 10});
    tab << "Name"
        << "N"
        << "K"
        << "Time (ms)";

    std::vector<int> ns = {20, 50, 100, 500};
    std::vector<int> ks = {20, 50, 100};

    DecoupledImuSolver solver;
    solver.optimizationOptions             = DefaultOptOptions();
    solver.optimizationOptions.debugOutput = false;

    for (int N : ns)
    {
        for (int K : ks)
        {
            DecoupledImuScene scene = MakeScene(options, N, K);

            {
                float t;
                auto cpy1 = scene;
                cpy1.PreintAll();
                {
                    ScopedTimer tim(t);
                    cpy1.SolveCeres(options, true);
                }
                tab << "Ceres AD" << N << K << t;
            }
            {
                float t;
                auto cpy1 = scene;
                cpy1.PreintAll();
                {
                    ScopedTimer tim(t);
                    cpy1.SolveCeres(options, false);
                }
                tab << "Ceres" << N << K << t;
            }
            {
                float t;
                auto cpy1 = scene;
                cpy1.PreintAll();


                {
                    ScopedTimer tim(t);
                    solver.Create(cpy1, options);
                    auto r = solver.initAndSolve();
                }
                tab << "Saiga" << N << K << t;
            }
            tab << "" << N << K << "";
        }
    }
}



}  // namespace Imu
}  // namespace Saiga
