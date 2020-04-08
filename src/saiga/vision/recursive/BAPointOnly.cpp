#include "BAPointOnly.h"

#include "saiga/core/math/random.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/Thread/omp.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/kernels/BAPoint.h"
#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/kernels/BAPosePoint.h"
#include "saiga/vision/util/LM.h"

#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"

#include <fstream>

namespace Saiga
{
void BAPointOnly::init()
{
    Scene& scene = *_scene;
    n            = scene.worldPoints.size();

    diagBlocks.resize(n);
    resBlocks.resize(n);
    x_v.resize(n);
    delta_x.resize(n);
    oldx_v.resize(n);

    for (int i = 0; i < n; ++i)
    {
        x_v[i] = scene.worldPoints[i].p;
    }

    // ===== Threading Tmps ======
    diagTemp.resize(threads - 1);
    resTemp.resize(threads - 1);
    for (auto& a : diagTemp) a.resize(n);
    for (auto& a : resTemp) a.resize(n);

    chi2_per_point.resize(n);
    chi2_per_point_new.resize(n);
}

void BAPointOnly::solve()
{
    //    std::cout << "BAPointOnly " << optimizationOptions.maxIterations << std::endl;
    for (auto k = 0; k < optimizationOptions.maxIterations; ++k)
    {
        auto chi2_before = computeQuadraticForm();
        addLambda(1e-4);
        solveLinearSystem();
        addDelta();
        auto chi2 = computeCost();

        for (int i = 0; i < n; ++i)
        {
            // revert if chi2 got worse
            if (chi2_per_point_new[i] > chi2_per_point[i])
            {
                x_v[i] = oldx_v[i];
            }
        }

        if (optimizationOptions.debugOutput)

        {
            std::cout << "it " << k << " " << chi2_before << " -> " << chi2 << std::endl;
        }
    }
    finalize();
}

double BAPointOnly::computeQuadraticForm()
{
    Scene& scene = *_scene;

    //    for (int i = 0; i < n; ++i)
    //    {
    //        diagBlocks[i].setZero();
    //        resBlocks[i].setZero();
    //    }

    //    double newChi2 = 0;


    DiagType* diagArray;
    ResType* resArray;


    // thread 0 directly writes into the recursive matrix
    diagArray = diagBlocks.data();
    resArray  = resBlocks.data();


    double chi2_sum = 0;
    // every thread has to zero its own local copy
    for (int i = 0; i < n; ++i)
    {
        diagArray[i].setZero();
        resArray[i].setZero();
        chi2_per_point[i] = 0;
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
                chi2_per_point[i] += c;
                chi2_sum += c;

                targetJ += JrowPoint.transpose() * JrowPoint;
                targetRes += JrowPoint.transpose() * res;
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
                chi2_per_point[i] += c;
                chi2_sum += c;

                targetJ += JrowPoint.transpose() * JrowPoint;
                targetRes += JrowPoint.transpose() * res;
            }
        }
    }
    return chi2_sum;
}

double BAPointOnly::computeCost()
{
    Scene& scene    = *_scene;
    double chi2_sum = 0;
    for (int i = 0; i < n; ++i)
    {
        chi2_per_point_new[i] = 0;
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
                chi2_per_point_new[i] += c;
                chi2_sum += c;
            }
            else
            {
                using MonoKernel = Saiga::Kernel::BAPointMono<T>;
                MonoKernel::ResidualType res;
                res = MonoKernel::evaluateResidual(camera, extr, wp, ip.point, 1);


                auto sqrtrw = sqrt(w);
                res *= sqrtrw;

                auto c = res.squaredNorm();
                chi2_per_point_new[i] += c;
                chi2_sum += c;
            }
        }
    }


    return chi2_sum;
}

bool BAPointOnly::addDelta()
{
#pragma omp for
    for (int i = 0; i < n; ++i)
    {
        oldx_v[i] = x_v[i];
        x_v[i] += delta_x[i];
    }
    return true;
}


void BAPointOnly::finalize()
{
    Scene& scene = *_scene;

#pragma omp for
    for (int i = 0; i < n; ++i)
    {
        scene.worldPoints[i].p = x_v[i];
    }
}


void BAPointOnly::addLambda(double lambda)
{
#pragma omp for
    for (int i = 0; i < n; ++i)
    {
        applyLMDiagonalInner(diagBlocks[i], lambda);
    }
}



void BAPointOnly::solveLinearSystem()
{
#pragma omp for
    for (int i = 0; i < n; ++i)
    {
        delta_x[i] = diagBlocks[i].ldlt().solve(resBlocks[i]);
    }
}



}  // namespace Saiga
