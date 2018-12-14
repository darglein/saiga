#include "BAPoseOnly.h"

#include "saiga/time/timer.h"
#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/kernels/BAPosePoint.h"

#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"

#include <fstream>

namespace Saiga
{
void BAPoseOnly::poseOnlySparse(Scene& scene, int its)
{
    SAIGA_BLOCK_TIMER();

    using T          = double;
    using KernelType = Saiga::Kernel::BAPoseMono<T>;
    KernelType::PoseJacobiType JrowPose;
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

            for (auto& ip : img.monoPoints)
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


void BAPoseOnly::posePointDense(Scene& scene, int its)
{
    using T          = double;
    using KernelType = Saiga::Kernel::BAPosePointMono<T>;
    KernelType::PoseJacobiType JrowPose;
    KernelType::PointJacobiType JrowPoint;
    KernelType::ResidualType res;

    SAIGA_BLOCK_TIMER();
    auto numCameras  = scene.extrinsics.size();
    auto numPoints   = scene.worldPoints.size();
    auto numUnknowns = numCameras * 6 + numPoints * 3;
    using MatrixType = Eigen::MatrixXd;
    MatrixType JtJ(numUnknowns, numUnknowns);
    Eigen::VectorXd Jtb(numUnknowns);

    for (int k = 0; k < its; ++k)
    {
        JtJ.setZero();
        Jtb.setZero();


        int n = 0;
        for (auto& img : scene.images)
        {
            auto extr   = scene.extrinsics[img.extr].se3;
            auto camera = scene.intrinsics[img.intr];

            for (auto& ip : img.monoPoints)
            {
                if (!ip)
                {
                    SAIGA_ASSERT(0);
                    continue;
                }

                auto wp = scene.worldPoints[ip.wp].p;

                auto poseStart  = img.extr * 6;
                auto pointStart = numCameras * 6 + ip.wp * 3;


                KernelType::evaluateResidualAndJacobian(camera, extr, wp, ip.point, ip.weight, res, JrowPose,
                                                        JrowPoint);

                JtJ.block(poseStart, poseStart, 6, 6) +=
                    (JrowPose.transpose() * JrowPose);  //.template triangularView<Eigen::Upper>();

                JtJ.block(pointStart, pointStart, 3, 3) +=
                    (JrowPoint.transpose() * JrowPoint);  //.template triangularView<Eigen::Upper>();

                JtJ.block(poseStart, pointStart, 6, 3) = JrowPose.transpose() * JrowPoint;
                JtJ.block(pointStart, poseStart, 3, 6) = JrowPoint.transpose() * JrowPose;

                Jtb.segment(poseStart, 6) += JrowPose.transpose() * res;
                Jtb.segment(pointStart, 3) += JrowPoint.transpose() * res;

                n++;
            }
        }

        //        std::ofstream strm("jtjdense.txt");
        //        strm << JtJ << endl;
        //        strm.close();

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
        for (size_t i = 0; i < numCameras; ++i)
        {
            Sophus::SE3d::Tangent t = x.segment(i * 6, 6);
            auto& se3               = scene.extrinsics[i].se3;
            se3                     = Sophus::SE3d::exp(t) * se3;
        }

        for (size_t i = 0; i < numPoints; ++i)
        {
            Vec3 t  = x.segment(numCameras * 6 + i * 3, 3);
            auto& p = scene.worldPoints[i].p;
            p += t;
        }
    }
}


void BAPoseOnly::posePointSparse(Scene& scene, int its)
{
    using T          = double;
    using KernelType = Saiga::Kernel::BAPosePointMono<T>;

    KernelType::PoseJacobiType JrowPose;
    KernelType::PointJacobiType JrowPoint;
    KernelType::ResidualType res;


    int N = 0;
    for (auto& img : scene.images)
    {
        for (auto& ip : img.monoPoints)
        {
            if (ip) N++;
        }
    }

    SAIGA_BLOCK_TIMER();
    auto numCameras  = scene.extrinsics.size();
    auto numPoints   = scene.worldPoints.size();
    auto numUnknowns = numCameras * 6 + numPoints * 3;

    std::vector<KernelType::PoseDiaBlockType> diagPoseBlocks(numCameras);
    std::vector<KernelType::PointDiaBlockType> diagPointBlocks(numPoints);
    std::vector<KernelType::PosePointUpperBlockType> posePointBlocks(N);
    Eigen::VectorXd Jtb(numUnknowns);

    for (int k = 0; k < its; ++k)
    {
        Jtb.setZero();
        for (auto& b : diagPoseBlocks) b.setZero();
        for (auto& b : diagPointBlocks) b.setZero();
        for (auto& b : posePointBlocks) b.setZero();

        int n = 0;
        for (auto& img : scene.images)
        {
            for (auto& ip : img.monoPoints)
            {
                if (!ip) continue;

                auto wp     = scene.worldPoints[ip.wp].p;
                auto extr   = scene.extrinsics[img.extr].se3;
                auto camera = scene.intrinsics[img.intr];

                auto poseStart  = img.extr * 6;
                auto pointStart = numCameras * 6 + ip.wp * 3;


                KernelType::evaluateResidualAndJacobian(camera, extr, wp, ip.point, ip.weight, res, JrowPose,
                                                        JrowPoint);

                diagPoseBlocks[img.extr] +=
                    (JrowPose.transpose() * JrowPose);  //.template triangularView<Eigen::Upper>();
                diagPointBlocks[ip.wp] +=
                    (JrowPoint.transpose() * JrowPoint);  //.template triangularView<Eigen::Upper>();

                posePointBlocks[n] = JrowPose.transpose() * JrowPoint;

                Jtb.segment(poseStart, 6) += JrowPose.transpose() * res;
                Jtb.segment(pointStart, 3) += JrowPoint.transpose() * res;

                n++;
            }
        }


        typedef Eigen::Triplet<T> Trip;
        std::vector<Trip> tripletList;

        Eigen::SparseMatrix<T> mat(numUnknowns, numUnknowns);  // default is column major
        //        mat.reserve(Eigen::VectorXi::Constant(numUnknowns, 6));

        for (size_t i = 0; i < numCameras; ++i)
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
        for (size_t i = 0; i < numPoints; ++i)
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
            for (auto& ip : img.monoPoints)
            {
                if (!ip) continue;

                auto poseStart  = img.extr * 6;
                auto pointStart = numCameras * 6 + ip.wp * 3;

                for (int c = 0; c < 3; ++c)
                {
                    for (int r = 0; r < 6; ++r)
                    {
                        //                        mat.insert(poseStart + r, pointStart + c) = posePointBlocks[n](r, c);
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
            double lambda = 1;
            // lm diagonal
            for (int i = 0; i < numUnknowns; ++i)
            {
                // that's what g2o does
                mat.coeffRef(i, i) += lambda;  // * JtJ(i, i);
            }
        }
        cout << "insert done" << endl;

        Eigen::VectorXd x;

        {
            SAIGA_BLOCK_TIMER();
            //        using SolverType = Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>, Eigen::Upper>;
            //            using SolverType = Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>, Eigen::Upper>;
            using SolverType = Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>, Eigen::Lower>;
            //            using SolverType = Eigen::ConjugateGradient<Eigen::SparseMatrix<T>, Eigen::Upper>;

            SolverType solver;
            x = solver.compute(mat).solve(Jtb);

            if (solver.info() != Eigen::Success)
            {
                cout << "decomposition failed" << endl;
                return;
            }
        }


        for (size_t i = 0; i < numCameras; ++i)
        {
            Sophus::SE3d::Tangent t = x.segment(i * 6, 6);
            auto& se3               = scene.extrinsics[i].se3;
            se3                     = Sophus::SE3d::exp(t) * se3;
        }
#if 1
        for (size_t i = 0; i < numPoints; ++i)
        {
            Vec3 t  = x.segment(numCameras * 6 + i * 3, 3);
            auto& p = scene.worldPoints[i].p;
            p += t;
        }
#endif
    }
}

}  // namespace Saiga
