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
#include "g2o_kernels/sophus_sba.h"

namespace Saiga
{
void g2oBA2::optimize(Scene& scene, int its, double huberMono, double huberStereo)
{
    //    Saiga::ScopedTimerPrint tim("optimize g2o");

    SAIGA_BLOCK_TIMER;
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);

    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    //    linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    linearSolver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();


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

    for (SceneImage& im : scene.images)
    {
        auto& camera     = scene.intrinsics[im.intr];
        auto vertex_extr = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(extrStartId + im.extr));
        SAIGA_ASSERT(vertex_extr);


        for (auto& o : im.monoPoints)
        {
            if (o.wp == -1) continue;
            auto vertex_wp = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(wpStartId + o.wp));
            SAIGA_ASSERT(vertex_wp);

#if 0
            if (o.depth > 0)
            {
                auto stereoPoint = o.point(0) - scene.bf / o.depth;

                Saiga::Vec3 obs(o.point(0), o.point(1), stereoPoint);

                g2o::EdgeSE3PointProjectDepth* e = new g2o::EdgeSE3PointProjectDepth();
                e->setVertex(0, vertex_wp);
                e->setVertex(1, vertex_extr);
                e->setMeasurement(obs);
                e->information() = Eigen::Matrix3d::Identity();


                e->bf   = bf;
                e->intr = camera;

                Vec2 weights(1, 1);
                weights *= o.weight;
                e->weights = weights;

                if (huberStereo > 0)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    rk->setDelta(huberStereo);
                    e->setRobustKernel(rk);
                }

                optimizer.addEdge(e);

                stereoEdges++;
            }
            else
#endif
            {
                //                continue;
                g2o::EdgeSE3PointProject* e = new g2o::EdgeSE3PointProject();
                e->setVertex(0, vertex_wp);
                e->setVertex(1, vertex_extr);
                e->setMeasurement(o.point);
                e->information() = Eigen::Matrix2d::Identity();

                e->intr   = camera;
                e->weight = o.weight;


                if (huberMono > 0)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    rk->setDelta(huberMono);
                    e->setRobustKernel(rk);
                }

                optimizer.addEdge(e);
                monoEdges++;
            }
        }
    }
    double costInit = 0, costFinal = 0;
    int totalDensePoints = 0;

    cout << "g2o problem created." << endl;


    optimizer.initializeOptimization();
    //    optimizer.computeActiveErrors();
    //    costInit = optimizer.activeRobustChi2();

    //    cout << "starting optimization initial chi2: " << costInit << endl;
    optimizer.optimize(its);
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
}
}  // namespace Saiga
