/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Scene.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/kernels/PGO.h"
#include "saiga/vision/kernels/Robust.h"
#include "saiga/vision/util/Random.h"

#include <fstream>
namespace Saiga
{
Eigen::Vector3d Scene::residual3(const SceneImage& img, const StereoImagePoint& ip)
{
    WorldPoint& wp = worldPoints[ip.wp];

    SAIGA_ASSERT(ip);
    SAIGA_ASSERT(wp);
    SAIGA_ASSERT(ip.depth > 0);

    // project to screen
    Vec3 p = img.se3 * wp.p;
    auto z = p(2);

    Vec2 p_norm = p.head<2>() / z;

    if (!distortion.empty())
    {
        p_norm = distortNormalizedPoint(p_norm, distortion[img.intr]);
    }

    auto p_img = intrinsics[img.intr].normalizedToImage(p_norm);

    auto w = ip.weight * scale();

    Eigen::Vector3d res;
    res.head<2>() = (ip.point - p_img);
    //    res(2)        = (1.0 / ip.depth - 1.0 / z) * bf;

    auto disparity      = p_img(0) - bf / z;
    auto stereoPointObs = ip.point(0) - bf / ip.depth;
    res(2)              = stereoPointObs - disparity;

    res *= w;

    // if (z <= 0) res *= 10000000;

    return res;
}

Eigen::Vector2d Scene::residual2(const SceneImage& img, const StereoImagePoint& ip)
{
    WorldPoint& wp = worldPoints[ip.wp];


    auto p = img.se3 * wp.p;
    auto z = p(2);

    Vec2 p_norm = p.head<2>() / z;

    if (!distortion.empty())
    {
        p_norm = distortNormalizedPoint(p_norm, distortion[img.intr]);
    }

    auto p_img = intrinsics[img.intr].normalizedToImage(p_norm);

    auto w = ip.weight * scale();
    Eigen::Vector2d res;
    res.head<2>() = (ip.point - p_img);
    res *= w;

    // if (z <= 0) res *= 10000000;
    return res;
}

void Scene::clear()
{
    intrinsics.clear();
    worldPoints.clear();
    images.clear();
    rel_pose_constraints.clear();
}

void Scene::reserve(int _images, int points, int observations)
{
    intrinsics.reserve(1);
    worldPoints.reserve(points);
    images.reserve(_images);
}

double Scene::residualNorm2(const SceneImage& img, const StereoImagePoint& ip)
{
    if (ip.depth > 0)
        return residual3(img, ip).squaredNorm();
    else
        return residual2(img, ip).squaredNorm();
}

double Scene::depth(const SceneImage& img, const StereoImagePoint& ip)
{
    WorldPoint& wp = worldPoints[ip.wp];

    SAIGA_ASSERT(ip);
    SAIGA_ASSERT(wp);

    // project to screen
    auto p = img.se3 * wp.p;
    auto z = p(2);

    return z;
}

void Scene::transformScene(const Saiga::SE3& transform)
{
    for (WorldPoint& wp : worldPoints)
    {
        wp.p = transform * wp.p;
    }


    for (SceneImage& e : images)
    {
        e.se3 = e.se3 * transform.inverse();
    }
}

void Scene::rescale(double s)
{
    for (WorldPoint& wp : worldPoints)
    {
        wp.p = s * wp.p;
    }
    for (SceneImage& e : images)
    {
        e.se3.translation() = s * e.se3.translation();
    }
}

void Scene::normalize()
{
    auto d        = depthStatistics();
    double target = sqrt(2) / d.median;
    rescale(target);

    auto m = medianWorldPoint();
    SE3 trans;
    trans.translation() = -m;
    transformScene(trans);
}

void Scene::fixWorldPointReferences()
{
    for (WorldPoint& wp : worldPoints)
    {
        wp.stereoreferences.clear();
        wp.valid = false;
    }


    int iid = 0;
    for (SceneImage& i : images)
    {
        int ipid      = 0;
        i.validPoints = 0;

        for (auto& ip : i.stereoPoints)
        {
            if (ip.wp >= 0)
            {
                WorldPoint& wp = worldPoints[ip.wp];
                wp.stereoreferences.emplace_back(iid, ipid);
                wp.valid = true;
                i.validPoints++;
            }
            ipid++;
        }
        iid++;
    }
}

bool Scene::valid() const
{
    int imgid = 0;
    for (const SceneImage& i : images)
    {
        if (i.intr < 0) return false;
        if (i.intr >= (int)intrinsics.size()) return false;

        for (auto& ip : i.stereoPoints)
        {
            if (!ip) continue;
            if (ip.wp >= (int)worldPoints.size()) return false;
            auto& wp = worldPoints[ip.wp];
            if (!wp.isReferencedByStereoFrame(imgid)) return false;
        }
        imgid++;
    }

    for (auto& wp : worldPoints)
    {
        if (!wp.uniqueReferences()) return false;
    }
    return true;
}


Saiga::Statistics<double> Scene::statistics()
{
    std::vector<double> stats;
    for (SceneImage& im : images)
    {
        for (auto& o : im.stereoPoints)
        {
            if (!o.wp) continue;
            stats.push_back(std::sqrt(residualNorm2(im, o)));
        }
    }

    Saiga::Statistics<double> sr(stats);
    return sr;
}

Saiga::Statistics<double> Scene::depthStatistics()
{
    std::vector<double> stats;
    for (SceneImage& im : images)
    {
        for (auto& o : im.stereoPoints)
        {
            if (!o.wp) continue;
            stats.push_back((depth(im, o)));
        }
    }
    Saiga::Statistics<double> sr(stats);
    return sr;
}

void Scene::removeOutliersFactor(float factor)
{
    SAIGA_ASSERT(valid());
    auto sr        = statistics();
    auto threshold = std::max(sr.median * factor, 1.0);
    removeOutliers(threshold);
}

void Scene::removeOutliers(float threshold)
{
    int pointsRemoved = 0;
    for (SceneImage& im : images)
    {
        for (auto& o : im.stereoPoints)
        {
            if (!o) continue;
            double r = std::sqrt(residualNorm2(im, o));
            if (r > threshold)
            {
                o.wp = -1;
                pointsRemoved++;
            }
        }
    }
    //    std::cout << "Removed " << pointsRemoved << " outlier observations above the threshold " << threshold <<
    //    std::endl;
    fixWorldPointReferences();
}

void Scene::removeWorldPoint(int id)
{
    SAIGA_ASSERT(id >= 0 && id < (int)worldPoints.size());

    WorldPoint& wp = worldPoints[id];
    if (!wp) return;


    // Remove all references
    for (auto& ref : wp.stereoreferences)
    {
        auto& ip = images[ref.first].stereoPoints[ref.second];
        ip.wp    = -1;
    }

    wp.valid = false;
    wp.stereoreferences.clear();
    SAIGA_ASSERT(!wp);
}

void Scene::removeCamera(int id)
{
    SAIGA_ASSERT(id >= 0 && id < (int)images.size());

    auto& im = images[id];

    int iip = 0;
    for (auto& ip : im.stereoPoints)
    {
        if (!ip) continue;
        auto& wp = worldPoints[ip.wp];
        wp.removeStereoReference(id, iip);
        ip.wp = -1;
        ++iip;
    }

    im.validPoints = 0;
    SAIGA_ASSERT(!im);
    SAIGA_ASSERT(valid());
}

void Scene::compress()
{
    fixWorldPointReferences();


    AlignedVector<WorldPoint> newWorldPoints;

    for (auto& wp : worldPoints)
    {
        if (wp.isValid())
        {
            int newid = newWorldPoints.size();
            newWorldPoints.push_back(wp);

            // update new world point id for every reference
            for (auto& p : wp.stereoreferences)
            {
                auto& ip = images[p.first].stereoPoints[p.second];
                ip.wp    = newid;
            }
        }
        else
        {
            // std::cout << "removed wp" << std::endl;
        }
    }
    worldPoints = newWorldPoints;
    SAIGA_ASSERT(valid());

    // count ips for each image

    int i = 0;
    for (auto& img : images)
    {
        img.validPoints = 0;
        for (auto& ip : img.stereoPoints)
        {
            if (ip) img.validPoints++;
        }
        if (img.validPoints == 0) std::cout << "invalid camera " << i << std::endl;
        i++;
    }
}

std::vector<int> Scene::validImages()
{
    std::vector<int> res;
    for (int i = 0; i < (int)images.size(); ++i)
    {
        if (images[i]) res.push_back(i);
    }
    return res;
}

std::vector<int> Scene::validPoints()
{
    std::vector<int> res;
    for (int i = 0; i < (int)worldPoints.size(); ++i)
    {
        if (worldPoints[i]) res.push_back(i);
    }
    return res;
}

double Scene::chi2(double huber)
{
    double error = 0;

    //    int stereoEdges = 0;
    //    int monoEdges   = 0;

    for (int i = 0; i < (int)images.size(); ++i)
    {
        SceneImage& im     = images[i];
        double image_error = 0;

        for (auto& o : im.stereoPoints)
        {
            if (!o) continue;
            double res_2 = residualNorm2(im, o);

            if (huber > 0)
            {
                auto rw = Kernel::HuberLoss<double>(huber, res_2);
                res_2   = rw(0);
            }
            image_error += res_2;
        }
        error += image_error;
        //        std::cout << "Rep " << i << " " << image_error << std::endl;
    }

    //    std::cout << "Scene stereo/mono/dense " << stereoEdges << "/" << monoEdges << "/" << 0 << " chi2: " << error
    //              << std::endl;
    //    std::cout << "BA error: " << error << std::endl;

    //    std::cout << "RPC SIze: " << rel_pose_constraints.size() << std::endl;
    for (auto& rpc : rel_pose_constraints)
    {
        SAIGA_ASSERT(rpc.img1 >= 0);
        SAIGA_ASSERT(rpc.img2 >= 0);
        auto& p1 = images[rpc.img1].se3;
        auto& p2 = images[rpc.img2].se3;
        auto e   = rpc.Residual(p1, p2).squaredNorm();
        error += e;
        //        std::cout << "RPC " << rpc.img1 << " - " << rpc.img2 << " Error: " << e << std::endl;
    }

    //    for (auto& i : images)
    //    {
    //        std::cout << i.se3 << std::endl;
    //    }

    return error;
}


double Scene::rms()
{
    double error = 0;

    int stereoEdges = 0;
    int monoEdges   = 0;

    for (SceneImage& im : images)
    {
        for (auto& o : im.stereoPoints)
        {
            if (!o) continue;
            double res_2 = residualNorm2(im, o);

            if (o.depth > 0)
                stereoEdges++;
            else
                monoEdges++;
            error += res_2;
        }
    }

    auto error2 = error / (monoEdges + stereoEdges);
    error2      = sqrt(error2);
    //    std::cout << "Scene stereo/mono/dense " << stereoEdges << "/" << monoEdges << "/" << 0 << " Error: " << error2
    //              << " chi2: " << error << std::endl;
    return error2;
}


void Scene::rmsPrint()
{
    for (int i = 0; i < (int)images.size(); ++i)
    {
        SceneImage& im     = images[i];
        double image_error = 0;
        for (int j = 0; j < (int)im.stereoPoints.size(); ++j)
        {
            auto& o = im.stereoPoints[j];
            if (!o) continue;
            double res_2 = residualNorm2(im, o);

            image_error += res_2;
            std::cout << "(" << i << ", " << j << ") " << res_2 << ", " << sqrt(res_2) << std::endl;
        }
        std::cout << "Image " << i << " Chi2: " << image_error << std::endl;
    }
}

double Scene::getSchurDensity()
{
    std::vector<std::vector<int>> schurStructure;

    auto imgs = validImages();
    long n    = images.size();

    schurStructure.clear();
    schurStructure.resize(n, std::vector<int>(n, -1));
    for (auto& wp : worldPoints)
    {
        for (auto& ref : wp.stereoreferences)
        {
            for (auto& ref2 : wp.stereoreferences)
            {
                int i1 = imgs[ref.first];
                int i2 = imgs[ref2.first];

                schurStructure[i1][ref2.first] = ref2.first;
                schurStructure[i2][ref.first]  = ref.first;
            }
        }
    }

    // compact it
    long schurEdges = 0;
    for (auto& v : schurStructure)
    {
        v.erase(std::remove(v.begin(), v.end(), -1), v.end());
        schurEdges += v.size();
    }

    double density = double(schurEdges) / double(n * n);
    return density;
}

void Scene::addWorldPointNoise(double stddev)
{
    for (auto& wp : worldPoints)
    {
        wp.p += Random::MatrixGauss<Vec3>(0, stddev);
    }
}

void Scene::addImagePointNoise(double stddev)
{
    for (auto& img : images)
    {
        for (auto& mp : img.stereoPoints) mp.point += Random::MatrixGauss<Vec2>(0, stddev);
    }
}

void Scene::addExtrinsicNoise(double stddev)
{
    for (SceneImage& e : images)
    {
        e.se3.translation() += Random::MatrixGauss<Vec3>(0, stddev);
    }
}
void Scene::addIntrinsicNoise(double stddev)
{
    for (auto& intr : intrinsics)
    {
        Vec5 rnd = Random::MatrixGauss<Vec5>(0, stddev);
        rnd(4) *= 0.1;
        intr = Vec5(intr.coeffs() + rnd);
    }
}

void Scene::applyErrorToImagePoints()
{
    for (SceneImage& img : images)
    {
        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;

            WorldPoint& wp = worldPoints[ip.wp];

            SAIGA_ASSERT(ip);
            SAIGA_ASSERT(wp);

            // project to screen
            auto p  = img.se3 * wp.p;
            auto p2 = intrinsics[img.intr].project(p);

            ip.point = p2;
        }
    }
}

void Scene::sortByWorldPointId()
{
    for (auto& img : images)
    {
        std::sort(img.stereoPoints.begin(), img.stereoPoints.end(),
                  [](const StereoImagePoint& i1, const StereoImagePoint& i2) { return i1.wp < i2.wp; });
    }
    fixWorldPointReferences();
    SAIGA_ASSERT(valid());
}

Vec3 Scene::medianWorldPoint()
{
    std::vector<double> mx, my, mz;
    mx.reserve(worldPoints.size());
    my.reserve(worldPoints.size());
    mz.reserve(worldPoints.size());
    for (auto& wp : worldPoints)
    {
        if (wp.isValid())
        {
            mx.push_back(wp.p(0));
            my.push_back(wp.p(1));
            mz.push_back(wp.p(2));
        }
    }
    std::sort(mx.begin(), mx.end());
    std::sort(my.begin(), my.end());
    std::sort(mz.begin(), mz.end());
    return {mx[mx.size() / 2], my[my.size() / 2], mz[mz.size() / 2]};
}

void Scene::removeNegativeProjections()
{
    int removedObs = 0;
    for (SceneImage& im : images)
    {
        for (auto& o : im.stereoPoints)
        {
            if (o.wp < 0) continue;
            auto p = im.se3 * worldPoints[o.wp].p;
            if (p(2) <= 0)
            {
                o.wp = -1;
                removedObs++;
            }
        }
    }
    fixWorldPointReferences();
    std::cout << "removed " << removedObs << " negative projections" << std::endl;
}

Vec6 RelPoseConstraint::Residual(const SE3& p1, const SE3& p2)
{
    Vec6 res = relPoseErrorView(rel_pose.inverse(), p1, p2, weight_rotation, weight_translation);
    //    SE3 rel  = p2 * p1.inverse();
    //    Vec6 res = Sophus::se3_logd(rel_pose.inverse() * rel);
    //    res.head<3>() *= weight_translation;
    //    res.tail<3>() *= weight_rotation;
    return res;
}



}  // namespace Saiga
