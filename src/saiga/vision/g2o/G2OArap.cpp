/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "G2OArap.h"

#include "saiga/core/time/timer.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/g2o/g2oHelper.h"

#include "g2o_kernels/sophus_arap.h"

namespace Saiga
{
OptimizationResults G2OArap::initAndSolve()
{
    auto& scene = *_scene;
    auto& arap  = *_scene;

    using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>>;
    auto solver       = g2o_make_optimizationAlgorithm<BlockSolver>(optimizationOptions);
    g2o::SparseOptimizer optimizer;
    optimizer.setComputeBatchStatistics(true);
    optimizer.setVerbose(optimizationOptions.debugOutput);
    optimizer.setAlgorithm(solver);



    // Add all pose vertices
    for (int i = 0; i < (int)scene.vertices.size(); ++i)
    {
        auto& img         = scene.vertices[i];
        ArapVertex* v_se3 = new ArapVertex();
        v_se3->setId(i);
        v_se3->setEstimate(img);
        optimizer.addVertex(v_se3);
    }


    // Add targets
    for (int i = 0; i < (int)scene.target_indices.size(); ++i)
    {
        int id           = scene.target_indices[i];
        auto vertex_from = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id));


        auto* ge = new ArapEdgeTarget();
        ge->setVertex(0, vertex_from);
        ge->setMeasurement(scene.target_positions[i]);
        ge->information() = Eigen::Matrix<double, 3, 3>::Identity();

        optimizer.addEdge(ge);
    }

    for (auto& c2 : arap.constraints)
    {
        {
            auto c  = c2;
            int i   = c.ids.first;
            int j   = c.ids.second;
            auto& p = arap.vertices[i];
            auto& q = arap.vertices[j];

            auto vertex_from = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i));
            auto vertex_to   = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(j));

            auto ge = new ArapEdge();
            ge->setVertex(0, vertex_from);
            ge->setVertex(1, vertex_to);
            ge->w_Reg = sqrt(c2.weight);
            ge->setMeasurement(p.translation() - q.translation());

            ge->information() = Eigen::Matrix<double, 3, 3>::Identity();

            optimizer.addEdge(ge);
        }

        if (1)
        {
            auto c  = c2.flipped();
            int i   = c.ids.first;
            int j   = c.ids.second;
            auto& p = arap.vertices[i];
            auto& q = arap.vertices[j];

            auto vertex_from = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i));
            auto vertex_to   = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(j));

            auto ge = new ArapEdge();
            ge->setVertex(0, vertex_from);
            ge->setVertex(1, vertex_to);
            ge->w_Reg = sqrt(c2.weight);
            ge->setMeasurement(p.translation() - q.translation());

            ge->information() = Eigen::Matrix<double, 3, 3>::Identity();

            optimizer.addEdge(ge);
        }
    }

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
        auto* v_se3 = static_cast<ArapVertex*>(optimizer.vertex(i));
        auto& e     = scene.vertices[i];
        e           = v_se3->estimate();
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
            if (s.chi2 != 0) result.cost_final = s.chi2;
        }
        result.linear_solver_time = ltime;
        //    result.cost_initial       = stats.front().chi2;

        //        result.cost_final = stats.back().chi2;

        if (invalid) result.cost_final = -1;
    }


    result.name = name;
    return result;
}


}  // namespace Saiga
