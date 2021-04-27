/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "g2oBA2.h"

#include "saiga/core/time/timer.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/g2o/g2oHelper.h"
#include "saiga/vision/scene/Scene.h"

#include "g2o_kernels/sophus_sba.h"

namespace Saiga
{
OptimizationResults g2oBA2::initAndSolve()
{
    Scene& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(optimizationOptions.debugOutput);
    using BlockSolver = g2o::BlockSolver_6_3;
    //    using OptimizationAlgorithm = g2o::OptimizationAlgorithmLevenberg;
    //    using LinearSolver          = std::unique_ptr<BlockSolver::LinearSolverType>;

    //    auto linearSolver = g2o_make_linearSolver<BlockSolver>(optimizationOptions);
    //    switch (optimizationOptions.solverType)
    //    {
    //        case OptimizationOptions::SolverType::Direct:
    //        {
    //#ifdef USE_CHOLMOD_SOLVER
    //            auto ls = std::make_unique<g2o::LinearSolverCholmod<BlockSolver::PoseMatrixType>>();
    //#else
    //            auto ls = std::make_unique<g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>>();
    //#endif
    //            linearSolver = std::move(ls);
    //            break;
    //        }
    //        case OptimizationOptions::SolverType::Iterative:
    //        {
    //            auto ls = g2o::make_unique<g2o::LinearSolverPCG<BlockSolver::PoseMatrixType>>();
    //            ls->setMaxIterations(optimizationOptions.maxIterativeIterations);
    //            ls->setTolerance(optimizationOptions.iterativeTolerance * optimizationOptions.iterativeTolerance);
    //            linearSolver = std::move(ls);
    //            break;
    //        }
    //    }

    auto solver = g2o_make_optimizationAlgorithm<BlockSolver>(optimizationOptions);
    //    OptimizationAlgorithm* solver = new
    //    OptimizationAlgorithm(std::make_unique<BlockSolver>(std::move(linearSolver)));
    solver->setUserLambdaInit(optimizationOptions.initialLambda * optimizationOptions.initialLambda);
    //    solver->setMaxTrialsAfterFailure(2);
    g2o::SparseOptimizer optimizer;
    //    optimizer.setVerbose(optimizationOptions.debugOutput);
    //    //    optimizer.setComputeBatchStatistics(options.debugOutput);
    optimizer.setComputeBatchStatistics(true);
    optimizer.setAlgorithm(solver);


    std::vector<int> validImages;
    validImages.reserve(scene.images.size());

    int extrStartId = 0;
    for (int i = 0; i < (int)scene.images.size(); ++i)
    {
        auto& img = scene.images[i];
        if (!img) continue;

        int validId = validImages.size();

        g2o::VertexSE3* v_se3 = new g2o::VertexSE3();
        v_se3->setId(validId);
        auto& e = img;
        v_se3->setEstimate((e.se3));
        v_se3->setFixed(e.constant);
        optimizer.addVertex(v_se3);
        validImages.push_back(i);
    }


    std::vector<int> validPoints;
    validPoints.reserve(scene.worldPoints.size());

    std::vector<int> pointToValidMap(scene.worldPoints.size());

    int wpStartId = validImages.size();

    int point_id = wpStartId;
    for (int i = 0; i < (int)scene.worldPoints.size(); ++i)
    {
        auto& wp = scene.worldPoints[i];
        if (!wp) continue;
        int validId           = validPoints.size();
        g2o::VertexPoint* v_p = new g2o::VertexPoint();
        v_p->setId(point_id);
        v_p->setMarginalized(true);
        v_p->setEstimate(wp.p);
        optimizer.addVertex(v_p);
        point_id++;
        pointToValidMap[i] = validId;
        validPoints.push_back(i);
    }


    int stereoEdges = 0;
    int monoEdges   = 0;

    int currentImage = 0;
    for (SceneImage& img : scene.images)
    {
        if (!img) continue;
        auto& camera     = scene.intrinsics[img.intr];
        int camvertexid  = currentImage + extrStartId;
        auto vertex_extr = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(camvertexid));
        SAIGA_ASSERT(vertex_extr);


        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;
            double w = ip.weight * scene.scale();

            int wpvertexid = pointToValidMap[ip.wp] + wpStartId;
            auto vertex_wp = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(wpvertexid));
            SAIGA_ASSERT(vertex_wp);

            if (ip.depth > 0)
            {
                auto stereoPoint = ip.point(0) - scene.bf / ip.depth;

                Saiga::Vec3 obs(ip.point(0), ip.point(1), stereoPoint);

                g2o::EdgeSE3PointProjectDepth* e = new g2o::EdgeSE3PointProjectDepth();
                e->setVertex(0, vertex_wp);
                e->setVertex(1, vertex_extr);
                e->setMeasurement(obs);
                e->information() = Eigen::Matrix3d::Identity();
                e->bf            = scene.bf;
                e->intr          = camera;
                e->weights       = Vec2(w, w);

                if (baOptions.huberStereo > 0)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    rk->setDelta(baOptions.huberStereo);
                    e->setRobustKernel(rk);
                }

                optimizer.addEdge(e);

                stereoEdges++;
            }
            else
            {
                g2o::EdgeSE3PointProject* e = new g2o::EdgeSE3PointProject();
                e->setVertex(0, vertex_wp);
                e->setVertex(1, vertex_extr);
                e->setMeasurement(ip.point);
                e->information() = Eigen::Matrix2d::Identity();
                e->intr          = camera;
                e->weight        = w;

                if (baOptions.huberMono > 0)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    rk->setDelta(baOptions.huberMono);
                    e->setRobustKernel(rk);
                }
                optimizer.addEdge(e);
                monoEdges++;
            }
        }
        currentImage++;
    }

    if (optimizationOptions.debugOutput)
        std::cout << "g2o problem created. Mono/Stereo " << monoEdges << "/" << stereoEdges << std::endl;


    OptimizationResults result;

    {
        Saiga::ScopedTimer<double> timer(result.total_time);
        SAIGA_OPTIONAL_BLOCK_TIMER(optimizationOptions.debugOutput);
        optimizer.initializeOptimization();

        optimizer.computeActiveErrors();
        result.cost_initial = optimizer.chi2();

        if (optimizationOptions.debugOutput)
        {
            optimizer.computeActiveErrors();
            double chi2b = optimizer.chi2();
            optimizer.optimize(optimizationOptions.maxIterations);
            double chi2a = optimizer.chi2();
            std::cout << "g2o::optimize " << chi2b << " -> " << chi2a << std::endl;
        }
        else
        {
            optimizer.optimize(optimizationOptions.maxIterations);
        }
    }

#if 0
    auto stats = optimizer.batchStatistics();
    for (auto s : stats)
    {
        std::cout << " levenbergIterations " << s.levenbergIterations << std::endl
             << " timeResiduals " << s.timeResiduals << std::endl
             << " timeLinearize " << s.timeLinearize << std::endl
             << " timeQuadraticForm " << s.timeQuadraticForm << std::endl
             << " timeSchurComplement " << s.timeSchurComplement << std::endl
             << " timeLinearSolution " << s.timeLinearSolution << std::endl
             << " timeLinearSolver " << s.timeLinearSolver << std::endl
             << " timeUpdate " << s.timeUpdate << std::endl
             << " timeIteration " << s.timeIteration << std::endl
             << " timeMarginals " << s.timeMarginals << std::endl;
        std::cout << std::endl;
    }

    //    costFinal = optimizer.activeRobustChi2();

    //    std::cout << "Optimize g2o stereo/mono/dense " << stereoEdges << "/" << monoEdges << "/" << totalDensePoints
    //         << " Error: " << costInit << "->" << costFinal << std::endl;

#endif


    for (size_t i = 0; i < validImages.size(); ++i)
    {
        int vertex            = i + extrStartId;
        g2o::VertexSE3* v_se3 = static_cast<g2o::VertexSE3*>(optimizer.vertex(vertex));
        auto& e               = scene.images[validImages[i]];
        auto se3              = v_se3->estimate();
        e.se3                 = (se3);
    }

    for (size_t i = 0; i < validPoints.size(); ++i)
    {
        auto& wp                = scene.worldPoints[validPoints[i]];
        int vertex              = i + wpStartId;
        g2o::VertexPoint* v_se3 = static_cast<g2o::VertexPoint*>(optimizer.vertex(vertex));
        wp.p                    = v_se3->estimate();
    }

    {
        int its      = 0;
        double ltime = 0;
        auto stats   = optimizer.batchStatistics();
        bool invalid = false;
        for (auto s : stats)
        {
            ltime += s.timeLinearSolution * 1000;
            its += s.iterationsLinearSolver;
            if (s.levenbergIterations != 1) invalid = true;
        }
        result.linear_solver_time = ltime;
        //    result.cost_initial       = stats.front().chi2;

        result.cost_final = stats.back().chi2;

        if (invalid) result.cost_final = -1;
    }

    return result;
}  // namespace Saiga
}  // namespace Saiga
