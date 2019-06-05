#include "BAPoseOnly.h"

#include "saiga/core/math/random.h"
#include "saiga/core/time/timer.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/kernels/BAPosePoint.h"

#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"
#include "EigenRecursive/All.h"

#include <fstream>

namespace Saiga
{
#if 0
void BAPoseOnly::poseOnlySparse(Scene& scene, int its)
{
    SAIGA_BLOCK_TIMER();

    using T          = double;
    using KernelType = Saiga::Kernel::BAPoseMono<T>;
    KernelType::JacobiType JrowPose;
    KernelType::ResidualType res;

    auto numCameras = scene.extrinsics.size();

    std::vector<KernelType::PoseDiaBlockType> diagBlocks(numCameras);
    std::vector<KernelType::ResidualBlockType> resBlocks(numCameras);

    for (int k = 0; k < its; ++k)
    {
        for (size_t i = 0; i < numCameras; ++i)
        {
            diagBlocks[i].setZero();
            resBlocks[i].setZero();
        }

        for (auto& img : scene.images)
        {
            auto extr   = scene.extrinsics[img.extr].se3;
            auto camera = scene.intrinsics[img.intr];

            for (auto& ip : img.stereoPoints)
            {
                if (!ip) continue;
                auto wp = scene.worldPoints[ip.wp].p;
                Saiga::Kernel::BAPoseMono<T>::evaluateResidualAndJacobian(camera, extr, wp, ip.point, res, JrowPose,
                                                                          ip.weight);
                diagBlocks[img.extr] += (JrowPose.transpose() * JrowPose).template triangularView<Eigen::Upper>();
                resBlocks[img.extr] += JrowPose.transpose() * res;
            }
        }

        for (size_t i = 0; i < numCameras; ++i)
        {
            Sophus::SE3d::Tangent t = diagBlocks[i].selfadjointView<Eigen::Upper>().ldlt().solve(resBlocks[i]);
            auto& se3               = scene.extrinsics[i].se3;
            se3                     = Sophus::SE3d::exp(t) * se3;
        }
    }
}
#endif

void BAPoseOnly::posePointDense(Scene& scene, int its)
{
    using T          = double;
    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
    KernelType::PoseJacobiType JrowPose;
    KernelType::PointJacobiType JrowPoint;
    KernelType::ResidualType res;


    int numObservations = 0;
    for (auto& img : scene.images)
    {
        for (auto& ip : img.stereoPoints)
        {
            if (!ip)
            {
                SAIGA_ASSERT(0);
                continue;
            }
            numObservations++;
        }
    }

    SAIGA_BLOCK_TIMER();
    int numCameras   = scene.extrinsics.size();
    int numPoints    = scene.worldPoints.size();
    int numUnknowns  = numCameras * 6 + numPoints * 3;
    using MatrixType = Eigen::MatrixXd;
    MatrixType JtJ(numUnknowns, numUnknowns);
    Eigen::VectorXd Jtb(numUnknowns);
    MatrixType J(numObservations * 2, numUnknowns);


    for (int k = 0; k < its; ++k)
    {
        JtJ.setZero();
        Jtb.setZero();
        J.setZero();

        int obs = 0;
        for (auto& img : scene.images)
        {
            auto extr   = scene.extrinsics[img.extr].se3;
            auto camera = scene.intrinsics[img.intr];

            for (auto& ip : img.stereoPoints)
            {
                if (!ip)
                {
                    SAIGA_ASSERT(0);
                    continue;
                }

                auto wp = scene.worldPoints[ip.wp].p;



                KernelType::evaluateResidualAndJacobian(camera, extr, wp, ip.point, ip.weight, res, JrowPose,
                                                        JrowPoint);

#if 0
                auto poseStart  = img.extr * 6;
                auto pointStart = numCameras * 6 + ip.wp * 3;
                JtJ.block(poseStart, poseStart, 6, 6) +=
                    (JrowPose.transpose() * JrowPose);  //.template triangularView<Eigen::Upper>();

                JtJ.block(pointStart, pointStart, 3, 3) +=
                    (JrowPoint.transpose() * JrowPoint);  //.template triangularView<Eigen::Upper>();

                JtJ.block(poseStart, pointStart, 6, 3) = JrowPose.transpose() * JrowPoint;
                JtJ.block(pointStart, poseStart, 3, 6) = JrowPoint.transpose() * JrowPose;

                Jtb.segment(poseStart, 6) += JrowPose.transpose() * res;
                Jtb.segment(pointStart, 3) += JrowPoint.transpose() * res;
#else
                auto poseStart  = img.extr * 6;
                auto pointStart = numCameras * 6 + ip.wp * 3;
                JtJ.block(pointStart, pointStart, 3, 3) += (JrowPoint.transpose() * JrowPoint);
                JtJ.block(poseStart, poseStart, 6, 6) += (JrowPose.transpose() * JrowPose);


                JtJ.block(poseStart, pointStart, 6, 3) = JrowPose.transpose() * JrowPoint;
                JtJ.block(pointStart, poseStart, 3, 6) = JrowPoint.transpose() * JrowPose;

                Jtb.segment(pointStart, 3) -= JrowPoint.transpose() * res;
                Jtb.segment(poseStart, 6) -= JrowPose.transpose() * res;

                J.block<2, 6>(obs * 2, poseStart)  = JrowPose;
                J.block<2, 3>(obs * 2, pointStart) = JrowPoint;
                obs++;
#endif
            }
        }

        //        cout << "J" << endl << J << endl << endl;

        //        cout << JtJ << endl << endl;


        //        std::ofstream strm("jtjdense.txt");
        //        strm << JtJ << endl;
        //        strm.close();


        if (1)
        {
            double lambda = 1;
            // lm diagonal
            for (int i = 0; i < numUnknowns; ++i)
            {
                // that's what g2o does
                JtJ(i, i) += lambda;  // * JtJ(i, i);
            }
        }


        Eigen::VectorXd x = JtJ.ldlt().solve(Jtb);


        Eigen::VectorXd x2 = x.segment(0, numCameras * 6);
        Eigen::VectorXd x1 = x.segment(numCameras * 6, numPoints * 3);



        for (int i = 0; i < numPoints; ++i)
        {
            Vec3 t  = x1.segment(i * 3, 3);
            auto& p = scene.worldPoints[i].p;
            p += t;
        }

        for (int i = 0; i < numCameras; ++i)
        {
            Sophus::SE3d::Tangent t = x2.segment(i * 6, 6);
            auto& se3               = scene.extrinsics[i].se3;
            se3                     = Sophus::SE3d::exp(t) * se3;
        }
    }
}

void BAPoseOnly::posePointSparseSchur(Scene& scene)
{
    using T          = double;
    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
    KernelType::PoseJacobiType JrowPose;
    KernelType::PointJacobiType JrowPoint;
    KernelType::ResidualType res;

    int numCameras = scene.extrinsics.size();
    int numPoints  = scene.worldPoints.size();

    int n = numCameras * 6;
    int m = numPoints * 3;

    using Vector = Eigen::Matrix<double, -1, 1>;
    using Matrix = Eigen::Matrix<double, -1, -1>;

    //    Eigen::DiagonalMatrix<double, -1> U(n * 6);
    //    Eigen::DiagonalMatrix<double, -1> V(m * 3);
    Eigen::SparseMatrix<double, Eigen::RowMajor> U(n, n);
    Eigen::SparseMatrix<double, Eigen::RowMajor> V(m, m);
    Eigen::SparseMatrix<double, Eigen::RowMajor> Vinv(m, m);


    // Right hand side of the linear system
    Vector ea(n);
    Vector eb(m);

    Eigen::SparseMatrix<double, Eigen::RowMajor> W(n, m);
    Eigen::SparseMatrix<double, Eigen::RowMajor> WT;

    std::vector<KernelType::PoseDiaBlockType> diagPoseBlocks(numCameras);
    std::vector<KernelType::PointDiaBlockType> diagPointBlocks(numPoints);

    for (int k = 0; k < optimizationOptions.maxIterations; ++k)
    {
        typedef Eigen::Triplet<T> Trip;
        std::vector<Trip> tripletListW;
        std::vector<Trip> tripletListU;
        std::vector<Trip> tripletListV;
        std::vector<Trip> tripletListVinv;

        W.setZero();
        WT.setZero();
        U.setZero();
        V.setZero();
        Vinv.setZero();

        ea.setZero();
        eb.setZero();
        for (auto& b : diagPoseBlocks) b.setZero();
        for (auto& b : diagPointBlocks) b.setZero();

        for (auto& img : scene.images)
        {
            auto& extr   = scene.extrinsics[img.extr].se3;
            auto& camera = scene.intrinsics[img.intr];
            StereoCamera4 scam(camera, scene.bf);

            for (auto& ip : img.stereoPoints)
            {
                if (!ip) continue;

                auto& wp = scene.worldPoints[ip.wp].p;

                auto poseStart  = img.extr * 6;
                auto pointStart = ip.wp * 3;

                auto& targetPosePose   = diagPoseBlocks[img.extr];
                auto& targetPointPoint = diagPointBlocks[ip.wp];

                KernelType::PosePointUpperBlockType cross;
                auto& targetPosePoint = cross;

                auto targetPoseRes  = ea.segment<6>(poseStart);
                auto targetPointRes = eb.segment<3>(pointStart);

                if (ip.depth > 0)
                {
                    using KernelType = Saiga::Kernel::BAPosePointStereo<T>;
                    KernelType::PoseJacobiType JrowPose;
                    KernelType::PointJacobiType JrowPoint;
                    KernelType::ResidualType res;

                    KernelType::evaluateResidualAndJacobian(scam, extr, wp, ip.point, ip.depth, ip.weight, res,
                                                            JrowPose, JrowPoint);
                    targetPosePose += JrowPose.transpose() * JrowPose;
                    targetPointPoint += JrowPoint.transpose() * JrowPoint;
                    targetPosePoint = JrowPose.transpose() * JrowPoint;
                    targetPoseRes -= JrowPose.transpose() * res;
                    targetPointRes -= JrowPoint.transpose() * res;
                }
                else
                {
                    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
                    KernelType::PoseJacobiType JrowPose;
                    KernelType::PointJacobiType JrowPoint;
                    KernelType::ResidualType res;

                    KernelType::evaluateResidualAndJacobian(camera, extr, wp, ip.point, ip.weight, res, JrowPose,
                                                            JrowPoint);
                    targetPosePose += JrowPose.transpose() * JrowPose;
                    targetPointPoint += JrowPoint.transpose() * JrowPoint;
                    targetPosePoint = JrowPose.transpose() * JrowPoint;
                    targetPoseRes -= JrowPose.transpose() * res;
                    targetPointRes -= JrowPoint.transpose() * res;
                }

                for (int r = 0; r < 6; ++r)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        tripletListW.emplace_back(poseStart + r, pointStart + c, cross(r, c));
                    }
                }
            }
        }



        for (int i = 0; i < numCameras; ++i)
        {
            auto starti = i * 6;
            auto startj = i * 6;
            auto b      = diagPoseBlocks[i];
            b += KernelType::PoseDiaBlockType::Identity();
            for (int j = 0; j < 6; ++j)
            {
                for (int k = 0; k < 6; ++k)
                {
                    tripletListU.emplace_back(starti + k, startj + j, b(k, j));
                }
            }
        }
        for (int i = 0; i < numPoints; ++i)
        {
            auto starti = i * 3;
            auto startj = i * 3;
            auto b      = diagPointBlocks[i];
            b += KernelType::PointDiaBlockType::Identity();
            auto binv = b.inverse().eval();
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 3; ++k)
                {
                    tripletListV.emplace_back(starti + k, startj + j, b(k, j));
                    tripletListVinv.emplace_back(starti + k, startj + j, binv(k, j));
                }
            }
        }
        W.setFromTriplets(tripletListW.begin(), tripletListW.end());
        U.setFromTriplets(tripletListU.begin(), tripletListU.end());
        V.setFromTriplets(tripletListV.begin(), tripletListV.end());
        Vinv.setFromTriplets(tripletListVinv.begin(), tripletListVinv.end());
        WT = W.transpose();



        Vector deltaA(n);
        Vector deltaB(m);
        Eigen::SparseMatrix<double, Eigen::RowMajor> S(n, n);
        Eigen::SparseMatrix<double, Eigen::RowMajor> Y(n, m);
        Vector ej(n);
        Vector tmp(n);
        //        SAIGA_BLOCK_TIMER();
        {
            // Step 2
            // Compute Y
            Y = W * Vinv;

            // Step 3
            // Compute the Schur complement S
            // Not sure how good the sparse matrix mult is of eigen
            // maybe own implementation because the structure is well known before hand
            S = -Y * WT;
            //        cout << "S" << endl << S.toDense() << endl;
            S = U + S;

            // Step 4
            // Compute the right hand side of the schur system ej
            // S * da = ej
            ej = ea - Y * eb;

            // Step 5
            // Solve the schur system for da
            deltaA.setZero();

            if (optimizationOptions.solverType == OptimizationOptions::SolverType::Direct)
            {
                Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
                solver.compute(S);
                deltaA = solver.solve(ej);
            }
            else
            {
                Eigen::Recursive::RecursiveDiagonalPreconditioner<double> P;
                Eigen::Index iters = optimizationOptions.maxIterativeIterations;
                double tol         = optimizationOptions.iterativeTolerance;

                if (true)
                {
                    P.compute(S);
                    P.compute(Matrix::Identity(n, n));


                    recursive_conjugate_gradient(
                        [&](const Vector& v) {
                            tmp = S * v;
                            //                            tmp = Y * (WT * v);
                            //                            tmp = U * v - tmp;
                            return tmp;
                        },
                        ej, deltaA, P, iters, tol);
                }
                if (optimizationOptions.debugOutput) cout << "error " << tol << " iterations " << iters << endl;
            }

            // Step 6
            // Substitute the solultion deltaA into the original system and
            // bring it to the right hand side
            Vector q = eb - WT * deltaA;

            // Step 7
            // Solve the remaining partial system with the precomputed inverse of V

            deltaB = Vinv * q;
        }

        for (int i = 0; i < numCameras; ++i)
        {
            Sophus::SE3d::Tangent t = deltaA.segment(i * 6, 6);
            auto& se3               = scene.extrinsics[i].se3;
            se3                     = Sophus::SE3d::exp(t) * se3;
        }
        for (int i = 0; i < numPoints; ++i)
        {
            Vec3 t  = deltaB.segment(i * 3, 3);
            auto& p = scene.worldPoints[i].p;
            p += t;
        }
    }
}
OptimizationResults BAPoseOnly::initAndSolve()
{
    Scene& scene = *_scene;

    posePointSparseSchur(scene);

    OptimizationResults result;
    return result;

    using T          = double;
    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
    KernelType::PoseJacobiType JrowPose;
    KernelType::PointJacobiType JrowPoint;
    KernelType::ResidualType res;


    int N = 0;
    for (auto& img : scene.images)
    {
        for (auto& ip : img.stereoPoints)
        {
            if (ip) N++;
        }
    }

    SAIGA_BLOCK_TIMER();
    int numCameras  = scene.extrinsics.size();
    int numPoints   = scene.worldPoints.size();
    int numUnknowns = numCameras * 6 + numPoints * 3;

    std::vector<KernelType::PoseDiaBlockType> diagPoseBlocks(numCameras);
    std::vector<KernelType::PointDiaBlockType> diagPointBlocks(numPoints);
    std::vector<KernelType::PosePointUpperBlockType> posePointBlocks(N);
    Eigen::VectorXd Jtb(numUnknowns);

    for (int k = 0; k < optimizationOptions.maxIterations; ++k)
    {
        Jtb.setZero();
        for (auto& b : diagPoseBlocks) b.setZero();
        for (auto& b : diagPointBlocks) b.setZero();
        for (auto& b : posePointBlocks) b.setZero();

        int n = 0;
        for (auto& img : scene.images)
        {
            auto& extr   = scene.extrinsics[img.extr].se3;
            auto& camera = scene.intrinsics[img.intr];
            StereoCamera4 scam(camera, scene.bf);

            for (auto& ip : img.stereoPoints)
            {
                if (!ip) continue;

                auto& wp = scene.worldPoints[ip.wp].p;

                auto poseStart  = img.extr * 6;
                auto pointStart = numCameras * 6 + ip.wp * 3;

                auto& targetPosePose   = diagPoseBlocks[img.extr];
                auto& targetPointPoint = diagPointBlocks[ip.wp];
                auto& targetPosePoint  = posePointBlocks[n];
                auto targetPoseRes     = Jtb.segment<6>(poseStart);
                auto targetPointRes    = Jtb.segment<3>(pointStart);

                if (ip.depth > 0)
                {
                    using KernelType = Saiga::Kernel::BAPosePointStereo<T>;
                    KernelType::PoseJacobiType JrowPose;
                    KernelType::PointJacobiType JrowPoint;
                    KernelType::ResidualType res;

                    KernelType::evaluateResidualAndJacobian(scam, extr, wp, ip.point, ip.depth, ip.weight, res,
                                                            JrowPose, JrowPoint);
                    targetPosePose += JrowPose.transpose() * JrowPose;
                    targetPointPoint += JrowPoint.transpose() * JrowPoint;
                    targetPosePoint = JrowPose.transpose() * JrowPoint;
                    targetPoseRes += JrowPose.transpose() * res;
                    targetPointRes += JrowPoint.transpose() * res;
                }
                else
                {
                    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
                    KernelType::PoseJacobiType JrowPose;
                    KernelType::PointJacobiType JrowPoint;
                    KernelType::ResidualType res;

                    KernelType::evaluateResidualAndJacobian(camera, extr, wp, ip.point, ip.weight, res, JrowPose,
                                                            JrowPoint);
                    targetPosePose += JrowPose.transpose() * JrowPose;
                    targetPointPoint += JrowPoint.transpose() * JrowPoint;
                    targetPosePoint = JrowPose.transpose() * JrowPoint;
                    targetPoseRes += JrowPose.transpose() * res;
                    targetPointRes += JrowPoint.transpose() * res;
                }
                n++;
            }
        }



        typedef Eigen::Triplet<T> Trip;
        std::vector<Trip> tripletList;

        Eigen::SparseMatrix<T> mat(numUnknowns, numUnknowns);  // default is column major
        //        mat.reserve(Eigen::VectorXi::Constant(numUnknowns, 6));

        for (int i = 0; i < numCameras; ++i)
        {
            auto starti = i * 6;
            auto startj = i * 6;
            for (int j = 0; j < 6; ++j)
            {
                for (int k = 0; k < 6; ++k)
                {
                    //                    mat.insert(starti + k, startj + j) = diagPoseBlocks[i](k, j);
                    tripletList.emplace_back(starti + k, startj + j, diagPoseBlocks[i](k, j));
                }
            }
        }
        for (int i = 0; i < numPoints; ++i)
        {
            auto starti = numCameras * 6 + i * 3;
            auto startj = numCameras * 6 + i * 3;
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 3; ++k)
                {
                    //                    mat.insert(starti + k, startj + j) = diagPointBlocks[i](k, j);
                    tripletList.emplace_back(starti + k, startj + j, diagPointBlocks[i](k, j));
                }
            }
        }
#if 1

        n = 0;
        for (auto& img : scene.images)
        {
            for (auto& ip : img.stereoPoints)
            {
                if (!ip) continue;

                auto poseStart  = img.extr * 6;
                auto pointStart = numCameras * 6 + ip.wp * 3;

                for (int c = 0; c < 3; ++c)
                {
                    for (int r = 0; r < 6; ++r)
                    {
                        //                        mat.insert(poseStart + r, pointStart + c) = posePointBlocks[n](r,
                        //                        c);
                        tripletList.emplace_back(poseStart + r, pointStart + c, posePointBlocks[n](r, c));
                        tripletList.emplace_back(pointStart + c, poseStart + r, posePointBlocks[n](r, c));
                    }
                }
                n++;
            }
        }
#endif

        mat.setFromTriplets(tripletList.begin(), tripletList.end());

        //        std::ofstream strm("jtjsparse.txt");
        //        strm << mat << endl;
        //        strm.close();
        {
            //            double lambda = 1;
            //            double lambda = 1.0 / scene.intrinsics.front().fx;
            double lambda = 1.0 / (scene.scale() * scene.scale());
            // lm diagonal
            for (int i = 0; i < numUnknowns; ++i)
            {
                // that's what g2o does
                mat.coeffRef(i, i) += lambda;  // * JtJ(i, i);
            }
        }



#if 0
        cout << mat.toDense() << endl;
#endif


        Eigen::VectorXd x;

        {
            //            SAIGA_BLOCK_TIMER();
            //        using SolverType = Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>, Eigen::Upper>;
            //            using SolverType = Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>, Eigen::Upper>;
            //            using SolverType =
            //            using SolverType = Eigen::ConjugateGradient<Eigen::SparseMatrix<T>, Eigen::Upper>;

            //            if (options.solverType == BAOptions::SolverType::Direct)
            {
                Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>, Eigen::Lower> solver;
                x = solver.compute(mat).solve(Jtb);
            }
            //            else if (options.solverType == BAOptions::SolverType::Iterative)
            //            {
            //                Eigen::ConjugateGradient<Eigen::SparseMatrix<T>, Eigen::Upper> solver;
            //                x = solver.compute(mat).solve(Jtb);
            //            }
            //            SolverType solver;
        }


#if 1
        for (int i = 0; i < numCameras; ++i)
        {
            Sophus::SE3d::Tangent t = x.segment(i * 6, 6);
            auto& se3               = scene.extrinsics[i].se3;
            se3                     = Sophus::SE3d::exp(t) * se3;
        }
        for (int i = 0; i < numPoints; ++i)
        {
            Vec3 t  = x.segment(numCameras * 6 + i * 3, 3);
            auto& p = scene.worldPoints[i].p;
            p += t;
        }
#endif
    }
    return {};
}

}  // namespace Saiga
