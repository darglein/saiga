/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "g2oBA2.h"

#include "saiga/time/timer.h"
#include "saiga/util/assert.h"

#include "g2o/core/solver.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o_kernels/sophus_sba.h"

namespace Saiga
{
void g2oBA2::solve(Scene& scene, const BAOptions& options)
{
    SAIGA_OPTIONAL_BLOCK_TIMER(options.debugOutput);
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(options.debugOutput);
    optimizer.setComputeBatchStatistics(options.debugOutput);

    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    //    linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();



    switch (options.solverType)
    {
        case BAOptions::SolverType::Direct:
            linearSolver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
            break;
        case BAOptions::SolverType::Iterative:
        {
            auto ls = g2o::make_unique<g2o::LinearSolverPCG<g2o::BlockSolver_6_3::PoseMatrixType>>();
            ls->setMaxIterations(options.maxIterativeIterations);
            ls->setTolerance(options.iterativeTolerance);
            linearSolver = std::move(ls);
        }

        break;
    }



    g2o::OptimizationAlgorithmLevenberg* solver =
        new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
    solver->setUserLambdaInit(1.0);

    //    g2o::OptimizationAlgorithmGaussNewton* solver =
    //        new
    //        g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));

    optimizer.setAlgorithm(solver);



    int extrStartId = 0;
    for (size_t i = 0; i < scene.extrinsics.size(); ++i)
    {
        g2o::VertexSE3* v_se3 = new g2o::VertexSE3();
        v_se3->setId(i);
        auto e = scene.extrinsics[i];
        v_se3->setEstimate((e.se3));
        v_se3->setFixed(e.constant);
        optimizer.addVertex(v_se3);
    }


    int wpStartId = scene.extrinsics.size();

    int point_id = wpStartId;
    for (auto wp : scene.worldPoints)
    {
        g2o::VertexPoint* v_p = new g2o::VertexPoint();
        v_p->setId(point_id);
        v_p->setMarginalized(true);
        v_p->setEstimate(wp.p);
        optimizer.addVertex(v_p);
        point_id++;
    }


    int stereoEdges = 0;
    int monoEdges   = 0;

    for (SceneImage& img : scene.images)
    {
        auto& camera     = scene.intrinsics[img.intr];
        auto vertex_extr = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(extrStartId + img.extr));
        SAIGA_ASSERT(vertex_extr);


        for (auto& ip : img.monoPoints)
        {
            if (!ip) continue;
            double w = ip.weight * scene.scale();

            auto vertex_wp = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(wpStartId + ip.wp));
            SAIGA_ASSERT(vertex_wp);

            //                continue;
            g2o::EdgeSE3PointProject* e = new g2o::EdgeSE3PointProject();
            e->setVertex(0, vertex_wp);
            e->setVertex(1, vertex_extr);
            e->setMeasurement(ip.point);
            e->information() = Eigen::Matrix2d::Identity();
            e->intr          = camera;
            e->weight        = w;

            if (options.huberMono > 0)
            {
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                rk->setDelta(options.huberMono);
                e->setRobustKernel(rk);
            }
            optimizer.addEdge(e);
            monoEdges++;
        }
        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;
            double w = ip.weight * scene.scale();

            auto vertex_wp = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(wpStartId + ip.wp));
            SAIGA_ASSERT(vertex_wp);
            SAIGA_ASSERT(ip.depth > 0);

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

                if (options.huberStereo > 0)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    rk->setDelta(options.huberStereo);
                    e->setRobustKernel(rk);
                }

                optimizer.addEdge(e);

                stereoEdges++;
            }
        }
    }
    //    double costInit = 0, costFinal = 0;
    //    int totalDensePoints = 0;

    if (options.debugOutput) cout << "g2o problem created. Mono/Stereo " << monoEdges << "/" << stereoEdges << endl;


    {
        SAIGA_OPTIONAL_BLOCK_TIMER(options.debugOutput);
        optimizer.initializeOptimization();
        //    optimizer.computeActiveErrors();
        //    costInit = optimizer.activeRobustChi2();

        //    cout << "starting optimization initial chi2: " << costInit << endl;
        optimizer.optimize(options.maxIterations);
    }

#if 0
    auto stats = optimizer.batchStatistics();
    for (auto s : stats)
    {
        cout << " levenbergIterations " << s.levenbergIterations << endl
             << " timeResiduals " << s.timeResiduals << endl
             << " timeLinearize " << s.timeLinearize << endl
             << " timeQuadraticForm " << s.timeQuadraticForm << endl
             << " timeSchurComplement " << s.timeSchurComplement << endl
             << " timeLinearSolution " << s.timeLinearSolution << endl
             << " timeLinearSolver " << s.timeLinearSolver << endl
             << " timeUpdate " << s.timeUpdate << endl
             << " timeIteration " << s.timeIteration << endl
             << " timeMarginals " << s.timeMarginals << endl;
    }
#endif


    //    costFinal = optimizer.activeRobustChi2();

    //    cout << "Optimize g2o stereo/mono/dense " << stereoEdges << "/" << monoEdges << "/" << totalDensePoints
    //         << " Error: " << costInit << "->" << costFinal << endl;



    for (size_t i = 0; i < scene.extrinsics.size(); ++i)
    {
        g2o::VertexSE3* v_se3 = static_cast<g2o::VertexSE3*>(optimizer.vertex(i));
        auto& e               = scene.extrinsics[i];
        auto se3              = v_se3->estimate();
        e.se3                 = (se3);
    }



    for (size_t i = 0; i < scene.worldPoints.size(); ++i)
    {
        auto& wp = scene.worldPoints[i];

        //        if (wp.references.size() >= 2)
        {
            g2o::VertexPoint* v_se3 = static_cast<g2o::VertexPoint*>(optimizer.vertex(i + wpStartId));

            wp.p = v_se3->estimate();
        }
    }
}  // namespace Saiga
}  // namespace Saiga
