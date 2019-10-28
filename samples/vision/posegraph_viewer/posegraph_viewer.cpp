/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "posegraph_viewer.h"

#include "saiga/core/image/imageTransformations.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/color.h"
#include "saiga/core/util/cv.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/scene/BALDataset.h"
//#include "saiga/vision/Eigen_GLM.h"
#include "saiga/vision/ceres/CeresBA.h"
#include "saiga/vision/ceres/CeresPGO.h"
#include "saiga/vision/g2o/g2oBA2.h"
#include "saiga/vision/g2o/g2oPoseGraph.h"
#include "saiga/vision/recursive/BAPoseOnly.h"
#include "saiga/vision/recursive/PGORecursive.h"

#if defined(SAIGA_VULKAN_INCLUDED)
#    error Vulkan was included somewhere.
#endif

Sample::Sample()
{
    Saiga::SearchPathes::data.getFiles(datasets, "vision", ".posegraph");
    std::sort(datasets.begin(), datasets.end());
    std::cout << "Found " << datasets.size() << " posegraph datasets" << std::endl;


    Saiga::SearchPathes::data.getFiles(baldatasets, "vision", ".txt");
    std::sort(baldatasets.begin(), baldatasets.end());
    std::cout << "Found " << baldatasets.size() << " BAL datasets" << std::endl;


    frustum.createFrustum(camera.proj, 0.01);
    frustum.setColor(vec4{0, 1, 0, 1});

    auto shader = shaderLoader.load<MVPShader>("colored_points.glsl");
    frustum.create(shader, shader, shader, shader);
    frustum.loadDefaultShaders();
}



void Sample::update(float dt)
{
    Base::update(dt);

    if (change)
    {
        chi2 = scene.chi2();
        rms  = scene.rms();
        lineSoup.lines.clear();

        for (auto& e : scene.edges)
        {
            int i = e.from;
            int j = e.to;

            vec3 p1 = inverseMatchingSE3(scene.poses[i].se3).inverse().translation().cast<float>();
            vec3 p2 = inverseMatchingSE3(scene.poses[j].se3).inverse().translation().cast<float>();

            PointVertex pc1;
            PointVertex pc2;

            pc1.position = p1;
            pc2.position = p2;

            pc1.color = vec3(0, 0, 1);
            pc2.color = vec3(0, 0, 1);

            lineSoup.lines.push_back(pc1);
            lineSoup.lines.push_back(pc2);
        }


        lineSoup.updateBuffer();
        change = false;
    }
}


void Sample::renderOverlay(Camera* cam)
{
    //    Base::renderOverlay(cam);
    lineSoup.render(cam);



    for (auto& i : scene.poses)
    {
        Saiga::SE3 se3 = inverseMatchingSE3(i.se3);
        mat4 v         = (se3.matrix()).cast<float>();
        v              = Saiga::cvViewToGLView(v);
        v              = mat4(inverse(v));

        //            std::cout << v << std::endl;
        //        vec4 color = i.constant ? vec4(0, 0, 1, 0) : vec4(1, 0, 0, 0);

        frustum.render(cam, v);
    }
}

void Sample::renderFinal(Camera* cam)
{
    Base::renderFinal(cam);
    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("Pose Graph Viewer");

    ImGui::Text("Rms: %f", rms);
    ImGui::Text("chi2: %f", chi2);

    scene.imgui();

    ImGui::Separator();

    baoptions.imgui();

    ImGui::Separator();


#if 1
    {
        std::vector<const char*> strings;
        for (auto& d : datasets) strings.push_back(d.data());
        static int currentItem = 0;
        ImGui::Combo("Dataset", &currentItem, strings.data(), strings.size());
        if (ImGui::Button("Load Dataset"))
        {
            scene.load(Saiga::SearchPathes::data(datasets[currentItem]));
            //            scene.poses[0].constant = true;
            change = true;
        }
    }
#endif


    {
        std::vector<const char*> strings;
        for (auto& d : baldatasets) strings.push_back(d.data());
        static int currentItem = 0;
        ImGui::Combo("BAL Dataset", &currentItem, strings.data(), strings.size());
        if (ImGui::Button("Load BAL Dataset"))
        {
            Saiga::BALDataset bal(baldatasets[currentItem]);
            Saiga::Scene sc         = bal.makeScene();
            scene                   = Saiga::PoseGraph(sc);
            scene.poses[0].constant = true;
            change                  = true;
            std::cout << scene.chi2() << std::endl;
        }
    }

    //    if (ImGui::Button("Reload"))
    //    {
    //        scene.load(Saiga::SearchPathes::data("vision/loop.posegraph"));
    //    }

    if (ImGui::Button("Solve G2O"))
    {
        Saiga::g2oPGO ba;
        ba.optimizationOptions = baoptions;
        ba.create(scene);
        ba.initAndSolve();
        change = true;
    }

    //    barec.imgui();
    if (ImGui::Button("Solve Recursive"))
    {
        Saiga::PGORec barec;
        barec.optimizationOptions = baoptions;
        barec.create(scene);
        barec.initAndSolve();
        change = true;
    }

    if (ImGui::Button("Solve Ceres"))
    {
        Saiga::CeresPGO barec;
        barec.optimizationOptions = baoptions;
        barec.create(scene);
        barec.initAndSolve();
        change = true;
    }


    ImGui::End();
}

int main(const int argc, const char* argv[])
{
    using namespace Saiga;

    {
        Sample example;

        example.run();
    }

    return 0;
}
