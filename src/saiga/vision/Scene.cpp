/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Scene.h"

#include "saiga/imgui/imgui.h"
#include "saiga/util/assert.h"
#include "saiga/vision/Random.h"

#include <fstream>
namespace Saiga
{
Eigen::Vector3d Scene::residual(const SceneImage& img, const StereoImagePoint& ip)
{
    WorldPoint& wp = worldPoints[ip.wp];

    // project to screen
    auto p = extrinsics[img.extr].se3 * wp.p;
    auto z = p(2);


    auto p2 = intrinsics[img.intr].project(p);

    Eigen::Vector3d res;
    res.head<2>() = (ip.point - p2) * img.imageWeight;
    res(2)        = (1.0 / ip.depth - 1.0 / z) * bf * img.imageWeight;
    return res;
}

Eigen::Vector2d Scene::residual(const SceneImage& img, const MonoImagePoint& ip)
{
    WorldPoint& wp = worldPoints[ip.wp];

    // project to screen
    auto p  = extrinsics[img.extr].se3 * wp.p;
    auto p2 = intrinsics[img.intr].project(p);

    Eigen::Vector2d res;
    res.head<2>() = (ip.point - p2) * img.imageWeight;
    return res;
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

void Scene::fixWorldPointReferences()
{
    for (WorldPoint& wp : worldPoints)
    {
        wp.monoreferences.clear();
        wp.stereoreferences.clear();
    }


    int iid = 0;
    for (SceneImage& i : images)
    {
        int ipid      = 0;
        i.validPoints = 0;
        for (auto& ip : i.monoPoints)
        {
            if (ip.wp >= 0)
            {
                WorldPoint& wp = worldPoints[ip.wp];
                wp.monoreferences.emplace_back(iid, ipid);
                i.validPoints++;
            }
            ipid++;
        }
        for (auto& ip : i.stereoPoints)
        {
            if (ip.wp >= 0)
            {
                WorldPoint& wp = worldPoints[ip.wp];
                wp.stereoreferences.emplace_back(iid, ipid);
                i.validPoints++;
            }
            ipid++;
        }
        iid++;
    }
}

bool Scene::valid()
{
    for (SceneImage& i : images)
    {
        if (i.extr < 0 || i.intr < 0) return false;

        if (i.extr >= (int)extrinsics.size() || i.intr >= (int)intrinsics.size()) return false;

        for (auto& ip : i.monoPoints)
        {
            if (ip.wp < 0) continue;
            if (ip.wp >= (int)worldPoints.size()) return false;
        }
        for (auto& ip : i.stereoPoints)
        {
            if (ip.wp < 0) continue;
            if (ip.wp >= (int)worldPoints.size()) return false;
        }
    }
    return true;
}

Saiga::Statistics<double> Scene::statistics()
{
    std::vector<double> stats;
    for (SceneImage& im : images)
    {
        for (auto& o : im.monoPoints)
        {
            if (o.wp < 0) continue;
            stats.push_back(residual(im, o).norm());
        }
        for (auto& o : im.stereoPoints)
        {
            if (o.wp < 0) continue;
            stats.push_back(residual(im, o).norm());
        }
    }

    Saiga::Statistics<double> sr(stats);
    return sr;
}


void Scene::removeOutliers(float factor)
{
    SAIGA_ASSERT(0);
    auto sr = statistics();

    auto threshold = std::max(sr.median * factor, 1.0);

    int pointsRemoved = 0;
    for (SceneImage& im : images)
    {
        //        auto e = extrinsics[im.extr];
        //        auto i = intrinsics[im.intr];
        for (auto& o : im.stereoPoints)
        {
            if (o.wp < 0) continue;

            Eigen::Vector3d res = residual(im, o);
            double r            = res.norm();

            if (r > threshold)
            {
                o.wp = -1;
                pointsRemoved++;
            }
        }
    }
    //    cout << "removed " << pointsRemoved << " points." << endl;
    fixWorldPointReferences();
}

void Scene::compress()
{
    fixWorldPointReferences();


    std::vector<WorldPoint> newWorldPoints;

    for (auto& wp : worldPoints)
    {
        if (wp.isValid())
        {
            int newid = newWorldPoints.size();
            newWorldPoints.push_back(wp);

            // update new world point id for every reference
            for (auto& p : wp.monoreferences)
            {
                MonoImagePoint& ip = images[p.first].monoPoints[p.second];
                ip.wp              = newid;
            }
        }
        else
        {
            cout << "removed wp" << endl;
        }
    }
    worldPoints = newWorldPoints;
    SAIGA_ASSERT(valid());

    // count ips for each image

    int i = 0;
    for (auto& img : images)
    {
        img.validPoints = 0;
        for (auto& ip : img.monoPoints)
        {
            if (ip) img.validPoints++;
        }
        if (img.validPoints == 0) cout << "invalid camera " << i << endl;
        i++;
    }
}

std::vector<int> Scene::validImages()
{
    std::vector<int> res;
    for (int i = 0; i < images.size(); ++i)
    {
        if (images[i].valid()) res.push_back(i);
    }
    return res;
}



double Scene::rms()
{
    double error = 0;

    int stereoEdges = 0;
    int monoEdges   = 0;

    for (SceneImage& im : images)
    {
        for (auto& o : im.monoPoints)
        {
            if (o.wp < 0) continue;
            double sqerror = residual(im, o).squaredNorm();
            error += sqerror;
            monoEdges++;
        }
        for (auto& o : im.stereoPoints)
        {
            if (o.wp < 0) continue;
            double sqerror = residual(im, o).squaredNorm();
            error += sqerror;
            stereoEdges++;
        }
    }

    auto error2 = error / (monoEdges + stereoEdges);
    error2      = sqrt(error2);
    //    cout << "Scene stereo/mono/dense " << stereoEdges << "/" << monoEdges << "/" << 0 << " Error: " << error2
    //         << " chi2: " << error << endl;
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
            //            cout << error.transpose() << endl;
            //            cout << cI << " " << obs(0) << endl;
            //            cout << cD << " " << z << endl;
            //            cout << error.squaredNorm() << endl;

            rms += error.squaredNorm();
            totalObservations++;
        }
    }

    //    exit(0);

    rms /= totalObservations;
    rms = sqrt(rms);
    return rms;
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
        for (auto& mp : img.monoPoints) mp.point += Random::gaussRandMatrix<Vec2>(0, stddev);
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

void Scene::sortByWorldPointId()
{
    for (auto& img : images)
    {
        std::sort(img.monoPoints.begin(), img.monoPoints.end(),
                  [](const MonoImagePoint& i1, const MonoImagePoint& i2) { return i1.wp < i2.wp; });
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
        for (auto& o : im.monoPoints)
        {
            if (o.wp < 0) continue;
            auto p = extrinsics[im.extr].se3 * worldPoints[o.wp].p;
            if (p(2) <= 0)
            {
                o.wp = -1;
                removedObs++;
            }
        }
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
    cout << "removed " << removedObs << " negative projections" << endl;
}



bool Scene::imgui()
{
    ImGui::PushID(3495672353);
    bool changed = false;

    if (ImGui::Button("RMS"))
    {
        rms();
    }

    if (ImGui::Button("Normalize"))
    {
        auto m = medianWorldPoint();
        cout << "median world point " << m.transpose() << endl;
        Saiga::SE3 T(Saiga::Quat::Identity(), -m);
        transformScene(T);
        changed = true;
    }

    static float sigma = 0.01;
    ImGui::InputFloat("sigma", &sigma);


    if (ImGui::Button("WP Noise"))
    {
        addWorldPointNoise(sigma);
        changed = true;
    }

    if (ImGui::Button("IP Noise"))
    {
        addImagePointNoise(sigma);
        changed = true;
    }

    if (ImGui::Button("Extr Noise"))
    {
        addExtrinsicNoise(0.1);
        changed = true;
    }

    ImGui::PopID();
    return changed;
}

}  // namespace Saiga
