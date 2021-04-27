/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "g2oPoseGraph.h"

#include "saiga/core/time/timer.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/g2o/g2oHelper.h"

#include "g2o_kernels/sophus_posegraph.h"


namespace Saiga
{
OptimizationResults g2oPGO::initAndSolve()
{
    auto& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(optimizationOptions.debugOutput);
    using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>>;
    //    using OptimizationAlgorithm = g2o::OptimizationAlgorithmLevenberg;
    //    using LinearSolver          = std::unique_ptr<BlockSolver::LinearSolverType>;

    auto solver = g2o_make_optimizationAlgorithm<BlockSolver>(optimizationOptions);
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(optimizationOptions.debugOutput);
    //    //    optimizer.setComputeBatchStatistics(options.debugOutput);
    optimizer.setComputeBatchStatistics(true);
    optimizer.setAlgorithm(solver);

    // Add all pose vertices
    for (int i = 0; i < (int)scene.vertices.size(); ++i)
    {
        auto& img         = scene.vertices[i];
        VertexSim3* v_se3 = new VertexSim3();
        v_se3->setId(i);
        v_se3->setEstimate(img.Pose());
        v_se3->fixScale = scene.fixScale;
        // fix the first camera
        v_se3->setFixed(img.constant);
        optimizer.addVertex(v_se3);
    }

    // Add all transformation edges
    double chi2 = 0;
    for (auto& e : scene.edges)
    {
        auto vertex_from = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(e.from));
        auto vertex_to   = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(e.to));
        SAIGA_ASSERT(vertex_from);
        SAIGA_ASSERT(vertex_to);

#ifdef LSD_REL
        const bool LSD = true;
#else
        const bool LSD = false;
#endif
        auto ge = new EdgeSim3<LSD>();
        ge->setVertex(0, vertex_from);
        ge->setVertex(1, vertex_to);
        ge->setMeasurement(e.GetSE3());
        //        ge->setMeasurementFromState();
        using PGOTransformation = SE3;
        ge->information()       = Eigen::Matrix<double, PGOTransformation::DoF, PGOTransformation::DoF>::Identity();
        optimizer.addEdge(ge);

        ge->computeError();
        chi2 += ge->chi2();
    }

    //    std::cout << "chi2: " << chi2 << std::endl;



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

    for (size_t i = 0; i < scene.vertices.size(); ++i)
    {
        //        VertexSim3* v_se3 = static_cast<VertexSim3*>(optimizer.vertex(i));
        //        auto& e           = scene.vertices[i].T_w_i;
        //        e                 = v_se3->estimate();
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

    result.name = name;
    return result;
}

}  // namespace Saiga
