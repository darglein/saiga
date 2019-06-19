/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/util/fileChecker.h"
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
        std::cout << "rms/chi2: " << rms() << " / " << chi2() << std::endl;
    }

    if (ImGui::Button("Normalize"))
    {
        auto m = medianWorldPoint();
        std::cout << "median world point " << m.transpose() << std::endl;
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

    if (ImGui::Button("Normalize"))
    {
        normalize();
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

    std::cout << "Saving scene to " << file << "." << std::endl;
    std::ofstream strm(file);
    SAIGA_ASSERT(strm.is_open());
    strm.precision(20);
    strm << std::scientific;


    strm << "# Saiga Scene file." << std::endl;
    strm << "#" << std::endl;
    strm << "# <num_intrinsics> <num_extrinsics> <num_images> <num_worldPoints>" << std::endl;
    strm << "# Intrinsics" << std::endl;
    strm << "# <fx> <fy> <cx> <cy>" << std::endl;
    strm << "# Extrinsics" << std::endl;
    strm << "# constant tx ty tz rx ry rz rw" << std::endl;
    strm << "# Images" << std::endl;
    strm << "# intr extr weight num_points" << std::endl;
    strm << "# wp depth px py weight" << std::endl;
    strm << "# WorldPoints" << std::endl;
    strm << "# x y z" << std::endl;
    strm << intrinsics.size() << " " << extrinsics.size() << " " << images.size() << " " << worldPoints.size() << " "
         << bf << " " << globalScale << std::endl;
    for (auto& i : intrinsics)
    {
        strm << i.coeffs().transpose() << std::endl;
    }
    for (auto& e : extrinsics)
    {
        strm << e.constant << " " << e.se3.params().transpose() << std::endl;
    }
    for (auto& img : images)
    {
        strm << img.intr << " " << img.extr << " " << img.imageWeight << " " << img.stereoPoints.size() << std::endl;
        for (auto& ip : img.stereoPoints)
        {
            strm << ip.wp << " " << ip.depth << " " << ip.point.transpose() << " " << ip.weight << std::endl;
        }
    }

    for (auto& wp : worldPoints)
    {
        strm << wp.p.transpose() << std::endl;
    }
}

void Scene::load(const std::string& file)
{
    std::cout << "Loading scene from " << file << "." << std::endl;


    std::ifstream strm(SearchPathes::data(file));
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
    strm << "[Scene]" << std::endl;

    int n = scene.validImages().size();
    int m = scene.validPoints().size();

    strm << " Images: " << n << "/" << scene.images.size() << std::endl;
    strm << " Points: " << m << "/" << scene.worldPoints.size() << std::endl;

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
    strm << " MonoEdges: " << monoEdges << std::endl;
    strm << " StereoEdges: " << stereoEdges << std::endl;
    strm << " TotalEdges: " << monoEdges + stereoEdges << std::endl;
    strm << " Rms: " << scene.rms() << std::endl;
    strm << " Chi2: " << scene.chi2() << std::endl;

    double density = double(monoEdges + stereoEdges) / double(n * m);
    strm << " W Density: " << density * 100 << "%" << std::endl;

    return strm;
}

}  // namespace Saiga
