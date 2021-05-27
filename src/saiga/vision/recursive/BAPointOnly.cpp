#include "BAPointOnly.h"

#include "saiga/core/math/random.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/Thread/omp.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/kernels/BA.h"
#include "saiga/vision/util/LM.h"

#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"

#include <fstream>

namespace Saiga
{
OptimizationResults BAPointOnly::initAndSolve()
{
    OptimizationResults result;
    float time = 0;
    {
        ScopedTimer tim(time);
        init();
        result = solve();
    }
    result.total_time = time;
    return result;
}

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

OptimizationResults BAPointOnly::solve()
{
    OptimizationResults result;
    //    std::cout << "BAPointOnly " << optimizationOptions.maxIterations << std::endl;
    double chi2 = std::numeric_limits<double>::infinity();
    for (auto k = 0; k < optimizationOptions.maxIterations; ++k)
    {
        auto chi2_before = computeQuadraticForm();

        if (k == 0)
        {
            result.cost_initial = chi2_before;
        }

        addLambda(1e-4);
        solveLinearSystem();
        addDelta();
        chi2 = computeCost();

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
    result.cost_final = chi2;
    finalize();
    return result;
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
        auto extr   = img.se3;
        auto camera = scene.intrinsics[img.intr];
        StereoCamera4 scam(camera, scene.bf);

        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;
            auto i = ip.wp;

            //            auto wp = scene.worldPoints[ip.wp].p;
            auto wp = x_v[i];
            auto w  = ip.weight * scene.scale();

            auto& targetJ   = diagArray[i];
            auto& targetRes = resArray[i];

            if (ip.IsStereoOrDepth())
            {
                auto stereo_point = ip.GetStereoPoint(scene.bf);

                Matrix<double, 3, 3> JrowPoint;
                auto [res, depth] = BundleAdjustmentStereo(scam, ip.point, stereo_point, extr, wp, w,
                                                           w * scene.stereo_weight, nullptr, &JrowPoint);
                auto c            = res.squaredNorm();
                chi2_per_point[i] += c;
                chi2_sum += c;

                targetJ += JrowPoint.transpose() * JrowPoint;
                targetRes -= JrowPoint.transpose() * res;
            }
            else
            {
                Matrix<double, 2, 3> JrowPoint;
                auto [res, depth] = BundleAdjustment<double>(camera, ip.point, extr, wp, w, nullptr, &JrowPoint);


                auto c = res.squaredNorm();
                chi2_per_point[i] += c;
                chi2_sum += c;

                targetJ += JrowPoint.transpose() * JrowPoint;
                targetRes -= JrowPoint.transpose() * res;
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
        auto extr   = img.se3;
        auto camera = scene.intrinsics[img.intr];
        StereoCamera4 scam(camera, scene.bf);

        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;
            //            auto wp = scene.worldPoints[ip.wp].p;
            auto i  = ip.wp;
            auto wp = x_v[i];
            auto w  = ip.weight * scene.scale();



            if (ip.IsStereoOrDepth())
            {
                auto stereo_point = ip.GetStereoPoint(scene.bf);
                auto [res, depth] =
                    BundleAdjustmentStereo(scam, ip.point, stereo_point, extr, wp, w, w * scene.stereo_weight);



                auto c = res.squaredNorm();
                chi2_per_point_new[i] += c;
                chi2_sum += c;
            }
            else
            {
                auto [res, depth] = BundleAdjustment(scam, ip.point, extr, wp, w);

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
