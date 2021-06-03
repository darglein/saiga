#include "BAPoseOnly.h"

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
        x_v[i] = scene.images[i].se3;
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
            auto w    = ip.weight * scene.scale();

            if (ip.IsStereoOrDepth())
            {
                auto stereo_point = ip.GetStereoPoint(scene.bf);

                Matrix<double, 3, 6> JrowPose;
                auto [res, depth] = BundleAdjustmentStereo(scam, ip.point, stereo_point, extr, wp, w,
                                                           w * scene.stereo_weight, &JrowPose, nullptr);


                auto c = res.squaredNorm();
                chi2 += c;

                targetJ += JrowPose.transpose() * JrowPose;
                targetRes += JrowPose.transpose() * res;
            }
            else
            {
                Matrix<double, 2, 6> JrowPose;
                auto [res, depth] = BundleAdjustment<double>(camera, ip.point, extr, wp, w, &JrowPose, nullptr);



                auto c = res.squaredNorm();
                chi2 += c;

                targetJ += JrowPose.transpose() * JrowPose;
                targetRes += JrowPose.transpose() * res;
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
            auto w    = ip.weight * scene.scale();
            if (ip.IsStereoOrDepth())
            {
                auto stereo_point = ip.GetStereoPoint(scene.bf);
                auto [res, depth] =
                    BundleAdjustmentStereo(scam, ip.point, stereo_point, extr, wp, w, w * scene.stereo_weight);

                auto c = res.squaredNorm();
                chi2 += c;
            }
            else
            {
                auto [res, depth] = BundleAdjustment(scam, ip.point, extr, wp, w);

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
        scene.images[i].se3 = x_v[i];
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
