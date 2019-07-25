#include "BAPoseOnly.h"

#include "saiga/core/math/random.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/Thread/omp.h"
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

    // ===== Threading Tmps ======
    localChi2.resize(threads);
    diagTemp.resize(threads - 1);
    resTemp.resize(threads - 1);
    for (auto& a : diagTemp) a.resize(n);
    for (auto& a : resTemp) a.resize(n);
}

double BAPoseOnly::computeQuadraticForm()
{
    Scene& scene = *_scene;

    //    for (int i = 0; i < n; ++i)
    //    {
    //        diagBlocks[i].setZero();
    //        resBlocks[i].setZero();
    //    }

    //    double newChi2 = 0;

    int tid = OMP::getThreadNum();

    double& newChi2 = localChi2[tid];
    newChi2         = 0;
    DiagType* diagArray;
    ResType* resArray;

    if (tid == 0)
    {
        // thread 0 directly writes into the recursive matrix
        diagArray = diagBlocks.data();
        resArray  = resBlocks.data();
    }
    else
    {
        diagArray = diagTemp[tid - 1].data();
        resArray  = resTemp[tid - 1].data();
    }

    // every thread has to zero its own local copy
    for (int i = 0; i < n; ++i)
    {
        diagArray[i].setZero();
        resArray[i].setZero();
    }

#pragma omp for
    for (int i = 0; i < (int)scene.images.size(); ++i)
    {
        auto& img   = scene.images[i];
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

            auto& targetJ   = diagArray[i];
            auto& targetRes = resArray[i];
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

#pragma omp for
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < threads - 1; ++j)
        {
            diagBlocks[i] += diagTemp[j][i];
            resBlocks[i] += resTemp[j][i];
        }
    }



    double chi2 = 0;
    for (int i = 0; i < threads; ++i)
    {
        chi2 += localChi2[i];
    }

    return chi2;
}

double BAPoseOnly::computeCost()
{
    Scene& scene = *_scene;

    int tid = OMP::getThreadNum();

    double& newChi2 = localChi2[tid];
    newChi2         = 0;

#pragma omp for
    for (int i = 0; i < (int)scene.images.size(); ++i)
    {
        auto& img   = scene.images[i];
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

    double chi2 = 0;
    for (int i = 0; i < threads; ++i)
    {
        chi2 += localChi2[i];
    }

    return chi2;
}

void BAPoseOnly::addDelta()
{
#pragma omp for
    for (int i = 0; i < n; ++i)
    {
        x_v[i] += delta_x[i];
    }
}


void BAPoseOnly::revertDelta()
{
//    x_v = oldx_v;
#pragma omp for
    for (int i = 0; i < n; ++i)
    {
        x_v[i] = oldx_v[i];
    }
}

void BAPoseOnly::finalize()
{
    Scene& scene = *_scene;

#pragma omp for
    for (int i = 0; i < n; ++i)
    {
        scene.worldPoints[i].p = x_v[i];
    }
}


void BAPoseOnly::addLambda(double lambda)
{
#pragma omp for
    for (int i = 0; i < n; ++i)
    {
        applyLMDiagonalInner(diagBlocks[i], lambda);
    }
}



void BAPoseOnly::solveLinearSystem()
{
#pragma omp for
    for (int i = 0; i < n; ++i)
    {
        delta_x[i] = diagBlocks[i].ldlt().solve(resBlocks[i]);
    }
}



}  // namespace Saiga
