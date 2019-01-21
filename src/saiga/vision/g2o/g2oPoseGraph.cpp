/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "g2oPoseGraph.h"

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
#include "g2o_kernels/sophus_posegraph.h"


namespace Saiga
{
void g2oPoseGraph::solve(PoseGraph& scene)
{
    cout << "g2oPoseGraph::solve" << endl;

    using BlockSolver           = g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1> >;
    using LinearSolver          = g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;
    using OptimizationAlgorithm = g2o::OptimizationAlgorithmLevenberg;


    auto linearSolver = std::make_unique<LinearSolver>();
    linearSolver->setBlockOrdering(false);

    OptimizationAlgorithm* solver = new OptimizationAlgorithm(std::make_unique<BlockSolver>(std::move(linearSolver)));
    solver->setUserLambdaInit(1.0);

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);
    optimizer.setAlgorithm(solver);

    // Add all pose vertices
    for (int i = 0; i < (int)scene.poses.size(); ++i)
    {
        auto& img         = scene.poses[i];
        VertexSim3* v_se3 = new VertexSim3();
        v_se3->setId(i);
        v_se3->setEstimate(img);
        // fix the first camera
        v_se3->setFixed(i == 0);
        optimizer.addVertex(v_se3);
    }

    // Add all transformation edges
    for (auto& e : scene.edges)
    {
        auto vertex_from = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(e.from));
        auto vertex_to   = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(e.to));
        SAIGA_ASSERT(vertex_from);
        SAIGA_ASSERT(vertex_to);

        EdgeSim3* ge = new EdgeSim3();
        ge->setVertex(0, vertex_from);
        ge->setVertex(1, vertex_to);
        ge->setMeasurement(e.meassurement);
        //        ge->setMeasurementFromState();
        ge->information() = Eigen::Matrix<double, 6, 6>::Identity();
        optimizer.addEdge(ge);
    }


    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    double chi2b = optimizer.chi2();
    optimizer.optimize(5);
    double chi2a = optimizer.chi2();
    cout << "g2o::optimize " << chi2b << " -> " << chi2a << endl;
}

}  // namespace Saiga
