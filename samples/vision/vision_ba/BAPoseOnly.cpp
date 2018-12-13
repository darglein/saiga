#include "BAPoseOnly.h"

#include "saiga/time/timer.h"
#include "saiga/vision/kernels/BAPose.h"

#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"

namespace Saiga
{
void BAPoseOnly::optimize(Scene& scene, int its)
{
    SAIGA_BLOCK_TIMER

    int N = 0;

    // count residuals
    for (auto& img : scene.images)
    {
        for (auto& ip : img.monoPoints)
        {
            if (ip) N++;
        }
    }

    auto numCameras = scene.extrinsics.size();

    for (int k = 0; k < its; ++k)
    {
        // We have #cameras*6 unknowns and N*2 equations
        using MatrixType = Eigen::MatrixXd;
        MatrixType J(N * 2, numCameras * 6);
        J.setZero();



        Eigen::VectorXd r(N * 2);

        int i = 0;
        // Compute residuum
        for (auto& img : scene.images)
        {
            for (auto& ip : img.monoPoints)
            {
                if (!ip) continue;

                Vec2 re             = scene.residual(img, ip);
                r.segment(i * 2, 2) = re;


                auto wp = scene.worldPoints[ip.wp].p;
                auto e  = scene.extrinsics[img.extr].se3;
                auto in = scene.intrinsics[img.intr];



                Eigen::Vector3d cp = e * wp;


                // Transform to the actual image in pixels
                auto x  = cp(0);
                auto y  = cp(1);
                auto z  = cp(2);
                auto zz = z * z;


                Eigen::Matrix<double, 2, 6> Ji;

                // Translation
                Ji(0, 0) = 1.0 / z;
                Ji(0, 1) = 0;
                Ji(0, 2) = -x / zz;

                Ji(1, 0) = 0;
                Ji(1, 1) = 1.0 / z;
                Ji(1, 2) = -y / zz;


                // Rotation
                Ji(0, 3) = -y * x / zz;
                Ji(0, 4) = (zz + (x * x)) / zz;
                Ji(0, 5) = -y * z / zz;

                Ji(1, 3) = (-zz - (y * y)) / zz;
                Ji(1, 4) = x * y / zz;
                Ji(1, 5) = x * z / zz;


                // Mult rows with fx and fy
                Ji.row(0) *= in.fx;
                Ji.row(1) *= in.fy;

                J.block(i * 2, img.extr * 6, 2, 6) = Ji;

                i++;
            }
        }


        SAIGA_ASSERT(i == N);


        Eigen::VectorXd b = J.transpose() * r;
        MatrixType JtJ    = J.transpose() * J;

        Eigen::LDLT<Eigen::MatrixXd> ldlt(JtJ);


        Eigen::VectorXd x = ldlt.solve(b);

        for (int i = 0; i < numCameras; ++i)
        {
            Sophus::SE3d::Tangent t = x.segment(i * 6, 6);

            auto& se3 = scene.extrinsics[i].se3;
            se3       = Sophus::SE3d::exp(t) * se3;
        }
    }
}

}  // namespace Saiga
