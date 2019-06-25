#include "BAPoseOnly.h"

#include "saiga/core/math/random.h"
#include "saiga/core/time/timer.h"
#include "saiga/vision/LM.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/kernels/BAPoint.h"
#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/kernels/BAPosePoint.h"

#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"
#include "EigenRecursive/All.h"

#include <fstream>

namespace Saiga
{
double BAPoseOnly::computeQuadraticForm()
{
    Scene& scene = *_scene;

    for (int i = 0; i < n; ++i)
    {
        diagBlocks[i].setZero();
        resBlocks[i].setZero();
    }

    double newChi2 = 0;

    for (auto& img : scene.images)
    {
        auto extr   = scene.extrinsics[img.extr].se3;
        auto camera = scene.intrinsics[img.intr];
        StereoCamera4 scam(camera, scene.bf);

        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;
            auto i = ip.wp;

            //            auto wp = scene.worldPoints[ip.wp].p;
            auto wp = x_v[i];
            auto w  = ip.weight * img.imageWeight * scene.scale();

            auto& targetJ   = diagBlocks[i];
            auto& targetRes = resBlocks[i];
            if (ip.depth > 0)
            {
                using StereoKernel = Saiga::Kernel::BAPointStereo<T>;
                StereoKernel::PointJacobiType JrowPoint;
                StereoKernel::ResidualType res;
                StereoKernel::evaluateResidualAndJacobian(scam, extr, wp, ip.point, ip.depth, 1, res, JrowPoint);


                auto sqrtrw = sqrt(w);
                JrowPoint *= sqrtrw;
                res *= sqrtrw;

                auto c = res.squaredNorm();
                newChi2 += c;

                targetJ += JrowPoint.transpose() * JrowPoint;
                targetRes -= JrowPoint.transpose() * res;
            }
            else
            {
                using MonoKernel = Saiga::Kernel::BAPointMono<T>;
                MonoKernel::PointJacobiType JrowPoint;
                MonoKernel::ResidualType res;
                MonoKernel::evaluateResidualAndJacobian(camera, extr, wp, ip.point, 1, res, JrowPoint);


                auto sqrtrw = sqrt(w);
                JrowPoint *= sqrtrw;
                res *= sqrtrw;

                auto c = res.squaredNorm();
                newChi2 += c;

                targetJ += JrowPoint.transpose() * JrowPoint;
                targetRes -= JrowPoint.transpose() * res;
            }
        }
    }

    //    std::cout << "chi2 " << newChi2 << std::endl;


    return newChi2;
}

double BAPoseOnly::computeCost()
{
    Scene& scene = *_scene;

    for (int i = 0; i < n; ++i)
    {
        diagBlocks[i].setZero();
        resBlocks[i].setZero();
    }

    double newChi2 = 0;

    for (auto& img : scene.images)
    {
        auto extr   = scene.extrinsics[img.extr].se3;
        auto camera = scene.intrinsics[img.intr];
        StereoCamera4 scam(camera, scene.bf);

        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;
            //            auto wp = scene.worldPoints[ip.wp].p;
            auto i  = ip.wp;
            auto wp = x_v[i];
            auto w  = ip.weight * img.imageWeight * scene.scale();



            if (ip.depth > 0)
            {
                using StereoKernel = Saiga::Kernel::BAPointStereo<T>;

                StereoKernel::ResidualType res;
                res = StereoKernel::evaluateResidual(scam, extr, wp, ip.point, ip.depth, 1);


                auto sqrtrw = sqrt(w);

                res *= sqrtrw;

                auto c = res.squaredNorm();
                newChi2 += c;
            }
            else
            {
                using MonoKernel = Saiga::Kernel::BAPointMono<T>;
                MonoKernel::ResidualType res;
                res = MonoKernel::evaluateResidual(camera, extr, wp, ip.point, 1);


                auto sqrtrw = sqrt(w);
                res *= sqrtrw;

                auto c = res.squaredNorm();
                newChi2 += c;
            }
        }
    }

    //    std::cout << "chi2 " << newChi2 << std::endl;


    return newChi2;
}

void BAPoseOnly::addDelta()
{
    for (int i = 0; i < n; ++i)
    {
        x_v[i] += delta_x[i];
    }
}


void BAPoseOnly::revertDelta()
{
    x_v = oldx_v;
}

void BAPoseOnly::finalize()
{
    Scene& scene = *_scene;

    for (int i = 0; i < n; ++i)
    {
        scene.worldPoints[i].p = x_v[i];
    }
}


void BAPoseOnly::addLambda(double lambda)
{
    for (int i = 0; i < n; ++i)
    {
        applyLMDiagonalInner(diagBlocks[i], lambda);
    }
}



void BAPoseOnly::solveLinearSystem()
{
    for (int i = 0; i < n; ++i)
    {
        delta_x[i] = diagBlocks[i].ldlt().solve(resBlocks[i]);
    }
}


void BAPoseOnly::init()
{
    Scene& scene = *_scene;
    n            = scene.worldPoints.size();

    diagBlocks.resize(n);
    resBlocks.resize(n);
    x_v.resize(n);
    delta_x.resize(n);

    for (int i = 0; i < n; ++i)
    {
        x_v[i] = scene.worldPoints[i].p;
    }
}

}  // namespace Saiga
