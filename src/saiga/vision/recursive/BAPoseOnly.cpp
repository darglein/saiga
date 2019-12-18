#include "BAPoseOnly.h"

#include "saiga/core/math/random.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/Thread/omp.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/util/LM.h"

#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"

#include <fstream>

namespace Saiga
{
void BAPoseOnly::init()
{
    Scene& scene = *_scene;
    n            = scene.images.size();

    diagBlocks.resize(n);
    resBlocks.resize(n);
    x_v.resize(n);
    delta_x.resize(n);
    oldx_v.resize(n);

    for (int i = 0; i < n; ++i)
    {
        x_v[i] = scene.extrinsics[i].se3;
    }
}

double BAPoseOnly::computeQuadraticForm()
{
    Scene& scene = *_scene;

    double chi2 = 0;


    for (int i = 0; i < (int)scene.images.size(); ++i)
    {
        auto& img   = scene.images[i];
        auto camera = scene.intrinsics[img.intr];
        StereoCamera4 scam(camera, scene.bf);


        auto& targetJ   = diagBlocks[i];
        auto& targetRes = resBlocks[i];
        targetJ.setZero();
        targetRes.setZero();

        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;
            auto wp   = scene.worldPoints[ip.wp].p;
            auto extr = x_v[i];
            auto w    = ip.weight * img.imageWeight * scene.scale();

            if (ip.depth > 0)
            {
                using StereoKernel = Saiga::Kernel::BAPoseStereo<T>;
                StereoKernel::JacobiType JrowPoint;
                StereoKernel::ResidualType res;
                StereoKernel::evaluateResidualAndJacobian(scam, extr, wp, ip.point, ip.depth, res, JrowPoint, 1.0);


                auto sqrtrw = sqrt(w);
                JrowPoint *= sqrtrw;
                res *= sqrtrw;

                auto c = res.squaredNorm();
                chi2 += c;

                targetJ += JrowPoint.transpose() * JrowPoint;
                targetRes += JrowPoint.transpose() * res;
            }
            else
            {
                using MonoKernel = Saiga::Kernel::BAPoseMono<T>;
                MonoKernel::JacobiType JrowPoint;
                MonoKernel::ResidualType res;
                MonoKernel::evaluateResidualAndJacobian(camera, extr, wp, ip.point, res, JrowPoint, 1.0);


                auto sqrtrw = sqrt(w);
                JrowPoint *= sqrtrw;
                res *= sqrtrw;

                auto c = res.squaredNorm();
                chi2 += c;

                targetJ += JrowPoint.transpose() * JrowPoint;
                targetRes += JrowPoint.transpose() * res;
            }
        }
    }
    return chi2;
}

double BAPoseOnly::computeCost()
{
    Scene& scene = *_scene;

    double chi2 = 0;


    for (int i = 0; i < (int)scene.images.size(); ++i)
    {
        auto& img   = scene.images[i];
        auto camera = scene.intrinsics[img.intr];
        StereoCamera4 scam(camera, scene.bf);

        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;
            auto wp   = scene.worldPoints[ip.wp].p;
            auto extr = x_v[i];
            auto w    = ip.weight * img.imageWeight * scene.scale();
            if (ip.depth > 0)
            {
                using StereoKernel = Saiga::Kernel::BAPoseStereo<T>;
                StereoKernel::ResidualType res;
                res = StereoKernel::evaluateResidual(scam, extr, wp, ip.point, ip.depth, 1.0);

                auto sqrtrw = sqrt(w);
                res *= sqrtrw;

                auto c = res.squaredNorm();
                chi2 += c;
            }
            else
            {
                using MonoKernel = Saiga::Kernel::BAPoseMono<T>;
                MonoKernel::ResidualType res;
                res = MonoKernel::evaluateResidual(camera, extr, wp, ip.point, 1.0);

                auto sqrtrw = sqrt(w);
                res *= sqrtrw;

                auto c = res.squaredNorm();
                chi2 += c;
            }
        }
    }
    return chi2;
}

bool BAPoseOnly::addDelta()
{
    for (int i = 0; i < n; ++i)
    {
        oldx_v[i] = x_v[i];
        //        x_v[i] += SE3::exp(delta_x[i]);
        x_v[i] = SE3::exp(delta_x[i]) * x_v[i];
    }
    return true;
}


void BAPoseOnly::revertDelta()
{
    //    x_v = oldx_v;

    for (int i = 0; i < n; ++i)
    {
        x_v[i] = oldx_v[i];
    }
}

void BAPoseOnly::finalize()
{
    Scene& scene = *_scene;


    for (int i = 0; i < n; ++i)
    {
        scene.extrinsics[i].se3 = x_v[i];
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



}  // namespace Saiga
