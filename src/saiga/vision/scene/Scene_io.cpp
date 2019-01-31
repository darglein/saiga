/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/Random.h"

#include "Scene.h"

#include <fstream>
namespace Saiga
{
bool Scene::imgui()
{
    ImGui::PushID(473441235);
    bool changed = false;

    if (ImGui::Button("RMS"))
    {
        cout << "rms/chi2: " << rms() << " / " << chi2() << endl;
    }

    if (ImGui::Button("Normalize"))
    {
        auto m = medianWorldPoint();
        cout << "median world point " << m.transpose() << endl;
        Saiga::SE3 T(Saiga::Quat::Identity(), -m);
        transformScene(T);
        changed = true;
    }


    static float scalefactor = 1;
    ImGui::InputFloat("scale factor", &scalefactor);
    if (ImGui::Button("rescale"))
    {
        rescale(scalefactor);
        changed = true;
    }

    if (ImGui::Button("Normalize Scale"))
    {
        auto d = depthStatistics();

        double target = sqrt(2);
        rescale(target / d.median);
        changed = true;
    }

    if (ImGui::Button("Normalize Position"))
    {
        auto m = medianWorldPoint();
        SE3 trans;
        trans.translation() = -m;
        transformScene(trans);
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

void Scene::save(const std::string& file)
{
    SAIGA_ASSERT(valid());

    cout << "Saving scene to " << file << "." << endl;
    std::ofstream strm(file);
    SAIGA_ASSERT(strm.is_open());
    strm.precision(20);
    strm << std::scientific;


    strm << "# Saiga Scene file." << endl;
    strm << "#" << endl;
    strm << "# <num_intrinsics> <num_extrinsics> <num_images> <num_worldPoints>" << endl;
    strm << "# Intrinsics" << endl;
    strm << "# <fx> <fy> <cx> <cy>" << endl;
    strm << "# Extrinsics" << endl;
    strm << "# constant tx ty tz rx ry rz rw" << endl;
    strm << "# Images" << endl;
    strm << "# intr extr weight num_points" << endl;
    strm << "# wp depth px py weight" << endl;
    strm << "# WorldPoints" << endl;
    strm << "# x y z" << endl;
    strm << intrinsics.size() << " " << extrinsics.size() << " " << images.size() << " " << worldPoints.size() << " "
         << bf << " " << globalScale << endl;
    for (auto& i : intrinsics)
    {
        strm << i.coeffs().transpose() << endl;
    }
    for (auto& e : extrinsics)
    {
        strm << e.constant << " " << e.se3.params().transpose() << endl;
    }
    for (auto& img : images)
    {
        strm << img.intr << " " << img.extr << " " << img.imageWeight << " " << img.stereoPoints.size() << endl;
        for (auto& ip : img.stereoPoints)
        {
            strm << ip.wp << " " << ip.depth << " " << ip.point.transpose() << " " << ip.weight << endl;
        }
    }

    for (auto& wp : worldPoints)
    {
        strm << wp.p.transpose() << endl;
    }
}

void Scene::load(const std::string& file)
{
    cout << "Loading scene from " << file << "." << endl;


    std::ifstream strm(file);
    SAIGA_ASSERT(strm.is_open());


    auto consumeComment = [&]() {
        while (true)
        {
            auto c = strm.peek();
            if (c == '#')
            {
                std::string s;
                std::getline(strm, s);
            }
            else
            {
                break;
            }
        }
    };


    consumeComment();
    int num_intrinsics, num_extrinsics, num_images, num_worldPoints;
    strm >> num_intrinsics >> num_extrinsics >> num_images >> num_worldPoints >> bf >> globalScale;
    intrinsics.resize(num_intrinsics);
    extrinsics.resize(num_extrinsics);
    images.resize(num_images);
    worldPoints.resize(num_worldPoints);
    for (auto& i : intrinsics)
    {
        Vec4 test;
        strm >> test;
        i = test;
    }
    for (auto& e : extrinsics)
    {
        Eigen::Map<Sophus::Vector<double, SE3::num_parameters>> v2(e.se3.data());
        Sophus::Vector<double, SE3::num_parameters> v;
        strm >> e.constant >> v;
        v2 = v;
    }

    for (auto& img : images)
    {
        int numpoints;
        strm >> img.intr >> img.extr >> img.imageWeight >> numpoints;
        img.stereoPoints.resize(numpoints);
        for (auto& ip : img.stereoPoints)
        {
            strm >> ip.wp >> ip.depth >> ip.point >> ip.weight;
        }
    }

    for (auto& wp : worldPoints)
    {
        strm >> wp.p;
    }

    fixWorldPointReferences();
    SAIGA_ASSERT(valid());
}


std::ostream& operator<<(std::ostream& strm, Scene& scene)
{
    strm << "[Scene]" << endl;

    int n = scene.validImages().size();
    int m = scene.validPoints().size();

    strm << " Images: " << n << "/" << scene.images.size() << endl;
    strm << " Points: " << m << "/" << scene.worldPoints.size() << endl;

    int stereoEdges = 0;
    int monoEdges   = 0;
    for (SceneImage& im : scene.images)
    {
        for (auto& o : im.stereoPoints)
        {
            if (!o) continue;

            if (o.depth > 0)
                stereoEdges++;
            else
                monoEdges++;
        }
    }
    strm << " MonoEdges: " << monoEdges << endl;
    strm << " StereoEdges: " << stereoEdges << endl;
    strm << " TotalEdges: " << monoEdges + stereoEdges << endl;
    strm << " Rms: " << scene.rms() << endl;
    strm << " Chi2: " << scene.chi2() << endl;

    double density = double(monoEdges + stereoEdges) / double(n * m);
    strm << " W Density: " << density * 100 << "%" << endl;

    return strm;
}

}  // namespace Saiga
