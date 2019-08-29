/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Scene.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"
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
    auto p = extrinsics[img.extr].se3 * wp.p;
    auto z = p(2);


    auto p2 = intrinsics[img.intr].project(p);

    auto w = ip.weight * img.imageWeight * scale();

    Eigen::Vector3d res;
    res.head<2>() = (ip.point - p2);
    //    res(2)        = (1.0 / ip.depth - 1.0 / z) * bf;

    auto disparity      = p2(0) - bf / z;
    auto stereoPointObs = ip.point(0) - bf / ip.depth;
    res(2)              = stereoPointObs - disparity;

    res *= w;

    if (z <= 0) res *= 10000000;

    return res;
}

Eigen::Vector2d Scene::residual2(const SceneImage& img, const StereoImagePoint& ip)
{
    WorldPoint& wp = worldPoints[ip.wp];

    SAIGA_ASSERT(ip);
    SAIGA_ASSERT(wp);

    // project to screen
    auto p  = extrinsics[img.extr].se3 * wp.p;
    auto z  = p(2);
    auto p2 = intrinsics[img.intr].project(p);
    auto w  = ip.weight * img.imageWeight * scale();
    Eigen::Vector2d res;
    res.head<2>() = (ip.point - p2);
    res *= w;

    if (z <= 0) res *= 10000000;
    return res;
}

void Scene::clear()
{
    intrinsics.clear();
    extrinsics.clear();
    worldPoints.clear();
    images.clear();
}

void Scene::reserve(int _images, int points, int observations)
{
    intrinsics.reserve(1);
    extrinsics.reserve(_images);
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
    auto p = extrinsics[img.extr].se3 * wp.p;
    auto z = p(2);

    return z;
}

void Scene::transformScene(const Saiga::SE3& transform)
{
    for (WorldPoint& wp : worldPoints)
    {
        wp.p = transform * wp.p;
    }


    for (Extrinsics& e : extrinsics)
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
    for (Extrinsics& e : extrinsics)
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
        if (i.extr < 0 || i.intr < 0) return false;

        if (i.extr >= (int)extrinsics.size() || i.intr >= (int)intrinsics.size()) return false;

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
    auto sr = statistics();
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
            if (!o.wp) continue;
            double r = std::sqrt(residualNorm2(im, o));
            if (r > threshold)
            {
                o.wp = -1;
                pointsRemoved++;
            }
        }
    }
    std::cout << "Removed " << pointsRemoved << " outlier observations above the threshold " << threshold << std::endl;
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

double Scene::chi2()
{
    double error = 0;

    int stereoEdges = 0;
    int monoEdges   = 0;

    for (SceneImage& im : images)
    {
        double sqerror;

        for (auto& o : im.stereoPoints)
        {
            if (!o) continue;
            sqerror = residualNorm2(im, o);

            if (o.depth > 0)
                stereoEdges++;
            else
                monoEdges++;
            error += sqerror;
        }
    }

    return error;
}


double Scene::rms()
{
    double error = 0;

    int stereoEdges = 0;
    int monoEdges   = 0;

    for (SceneImage& im : images)
    {
        double sqerror;

        for (auto& o : im.stereoPoints)
        {
            if (!o) continue;
            sqerror = residualNorm2(im, o);

            if (o.depth > 0)
                stereoEdges++;
            else
                monoEdges++;
            error += sqerror;
        }
    }

    auto error2 = error / (monoEdges + stereoEdges);
    error2      = sqrt(error2);
    //    std::cout << "Scene stereo/mono/dense " << stereoEdges << "/" << monoEdges << "/" << 0 << " Error: " << error2
    //         << " chi2: " << error << std::endl;
    return error2;
}


double Scene::rmsDense()
{
    using T = double;

    double rms            = 0;
    int totalObservations = 0;

    for (SceneImage& im : images)
    {
        auto& e2      = extrinsics[im.extr];
        auto& se3_ref = e2.se3;
        auto& intr    = intrinsics[im.intr];

        for (DenseConstraint& dc : im.densePoints)
        {
            SceneImage& target = images[dc.targetImageId];
            auto se3_tar       = extrinsics[target.extr].se3;


            // Transform from the reference to world space
            Vec3 ip_ref(T(dc.referencePoint(0)), T(dc.referencePoint(1)), T(1));
            ip_ref(0) = (ip_ref(0) - T(intr.cx)) / T(intr.fx);
            ip_ref(1) = (ip_ref(1) - T(intr.cy)) / T(intr.fy);
            ip_ref *= T(dc.referenceDepth);

            Vec3 wp = se3_ref.inverse() * ip_ref;


            Vec3 pc = se3_tar * wp;

            auto x = pc(0);
            auto y = pc(1);
            auto z = pc(2);


            Vec2 ip(T(intr.fx) * x / z + T(intr.cx), T(intr.fy) * y / z + T(intr.cy));

            auto I = target.intensity.getImageView().inter(ip(1), ip(0));
            auto D = target.depth.getImageView().inter(ip(1), ip(0));

            Vec2 error(T(dc.referenceIntensity) - I, T(bf) / z - T(bf) * D);



            SAIGA_ASSERT(error.allFinite());
            //            std::cout << error.transpose() << std::endl;
            //            std::cout << cI << " " << obs(0) << std::endl;
            //            std::cout << cD << " " << z << std::endl;
            //            std::cout << error.squaredNorm() << std::endl;

            rms += error.squaredNorm();
            totalObservations++;
        }
    }

    //    exit(0);

    rms /= totalObservations;
    rms = sqrt(rms);
    return rms;
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
        wp.p += Random::gaussRandMatrix<Vec3>(0, stddev);
    }
}

void Scene::addImagePointNoise(double stddev)
{
    for (auto& img : images)
    {
        for (auto& mp : img.stereoPoints) mp.point += Random::gaussRandMatrix<Vec2>(0, stddev);
    }
}

void Scene::addExtrinsicNoise(double stddev)
{
    for (auto& e : extrinsics)
    {
        e.se3.translation() += Random::gaussRandMatrix<Vec3>(0, stddev);
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
            auto p  = extrinsics[img.extr].se3 * wp.p;
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
            auto p = extrinsics[im.extr].se3 * worldPoints[o.wp].p;
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



}  // namespace Saiga
