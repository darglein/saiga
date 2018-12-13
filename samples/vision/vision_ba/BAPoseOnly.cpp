#include "BAPoseOnly.h"

#include "saiga/time/timer.h"
#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/kernels/BAPosePoint.h"

#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"

namespace Saiga
{
void BAPoseOnly::poseOnlyDense(Scene& scene, int its)
{
    using T = double;
    SAIGA_BLOCK_TIMER
    auto numCameras  = scene.extrinsics.size();
    using MatrixType = Eigen::MatrixXd;
    MatrixType JtJ(numCameras * 6, numCameras * 6);
    Eigen::VectorXd Jtb(numCameras * 6);

    for (int k = 0; k < its; ++k)
    {
        JtJ.setZero();
        Jtb.setZero();
        for (auto& img : scene.images)
        {
            for (auto& ip : img.monoPoints)
            {
                if (!ip) continue;

                auto wp     = scene.worldPoints[ip.wp].p;
                auto extr   = scene.extrinsics[img.extr].se3;
                auto camera = scene.intrinsics[img.intr];


                Eigen::Matrix<T, 2, 6> Jrow;
                Vec2 res;
                Saiga::Kernel::BAPoseMono<T>::evaluateResidualAndJacobian(camera, extr, wp, ip.point, res, Jrow,
                                                                          ip.weight);

                JtJ.block(img.extr * 6, img.extr * 6, 6, 6) +=
                    (Jrow.transpose() * Jrow).template triangularView<Eigen::Upper>();
                Jtb.block(img.extr * 6, 0, 6, 1) += Jrow.transpose() * res;
            }
        }

        Eigen::VectorXd x = JtJ.selfadjointView<Eigen::Upper>().ldlt().solve(Jtb);
        for (size_t i = 0; i < numCameras; ++i)
        {
            Sophus::SE3d::Tangent t = x.segment(i * 6, 6);
            auto& se3               = scene.extrinsics[i].se3;
            se3                     = Sophus::SE3d::exp(t) * se3;
        }
    }
}

void BAPoseOnly::posePointDense(Scene& scene, int its)
{
    using T = double;
    SAIGA_BLOCK_TIMER
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

                using KernelType = Saiga::Kernel::BAPosePointMono<T>;

                KernelType::PoseJacobiType JrowPose;
                KernelType::PointJacobiType JrowPoint;
                KernelType::ResidualType res;
                KernelType::evaluateResidualAndJacobian(camera, extr, wp, ip.point, ip.weight, res, JrowPose,
                                                        JrowPoint);

                JtJ.block(poseStart, poseStart, 6, 6) +=
                    (JrowPose.transpose() * JrowPose).template triangularView<Eigen::Upper>();
                Jtb.block(poseStart, 0, 6, 1) += JrowPose.transpose() * res;
            }
        }

        Eigen::VectorXd x = JtJ.selfadjointView<Eigen::Upper>().ldlt().solve(Jtb);
        for (size_t i = 0; i < numCameras; ++i)
        {
            Sophus::SE3d::Tangent t = x.segment(i * 6, 6);
            auto& se3               = scene.extrinsics[i].se3;
            se3                     = Sophus::SE3d::exp(t) * se3;
        }
    }
}

}  // namespace Saiga
