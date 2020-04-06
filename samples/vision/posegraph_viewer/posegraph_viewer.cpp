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
#include "saiga/core/math/random.h"
#include "saiga/core/util/color.h"
#include "saiga/core/util/cv.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/scene/BALDataset.h"
#include "saiga/vision/scene/SynteticPoseGraph.h"
//#include "saiga/vision/Eigen_GLM.h"
#include "saiga/vision/ceres/CeresBA.h"
#include "saiga/vision/ceres/CeresPGO.h"
#include "saiga/vision/g2o/g2oBA2.h"
#include "saiga/vision/g2o/g2oPoseGraph.h"
#include "saiga/vision/recursive/BAPointOnly.h"
#include "saiga/vision/recursive/PGORecursive.h"
#include "saiga/vision/recursive/PGOSim3Recursive.h"

#if defined(SAIGA_VULKAN_INCLUDED)
#    error Vulkan was included somewhere.
#endif

Sample::Sample()
{
    std::cout << Saiga::SearchPathes::data << std::endl;
    //    Saiga::SearchPathes::data.getFiles(datasets, "vision", ".posegraph");
    Saiga::SearchPathes::data.getFiles(datasets, "user", ".posegraph");
    std::sort(datasets.begin(), datasets.end());
    std::cout << "Found " << datasets.size() << " posegraph datasets" << std::endl;


    Saiga::SearchPathes::data.getFiles(baldatasets, "vision", ".txt");
    std::sort(baldatasets.begin(), baldatasets.end());
    std::cout << "Found " << baldatasets.size() << " BAL datasets" << std::endl;


    frustum.createFrustum(camera.proj, 0.05);
    frustum.setColor(vec4{1, 1, 1, 1});
    frustum.create();

    baoptions.solverType   = OptimizationOptions::SolverType::Direct;
    baoptions.minChi2Delta = 1e-10;
    // in most cases only a few iterations are required, but setting the max to more doesn't really hurt
    baoptions.maxIterations = 50;
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

            vec3 p1;
            vec3 p2;

            if (render_inverse)
            {
                p1 = inverseMatchingSE3(scene.vertices[i].Sim3Pose()).inverse().translation().cast<float>();
                p2 = inverseMatchingSE3(scene.vertices[j].Sim3Pose()).inverse().translation().cast<float>();
            }
            else
            {
                p1 = scene.vertices[i].Pose().translation().cast<float>();
                p2 = scene.vertices[j].Pose().translation().cast<float>();
            }
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



    for (auto& i : scene.vertices)
    {
        Saiga::SE3 se3 = i.Pose();
        if (render_inverse)
        {
        }
        else
        {
            se3 = se3.inverse();
        }
        mat4 v = (se3.matrix()).cast<float>();
        v      = Saiga::cvViewToGLView(v);
        v      = mat4(inverse(v));

        //            std::cout << v << std::endl;
        //        vec4 color = i.constant ? vec4(0, 0, 1, 0) : vec4(1, 0, 0, 0);

        frustum.SetShaderColor(i.constant ? vec4(1, 0, 0, 1) : vec4(0, 1, 0, 1));
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

    if (ImGui::Checkbox("render_inverse", &render_inverse))
    {
        change = true;
    }
    if (scene.imgui())
    {
        change = true;
    }

    ImGui::Separator();

    baoptions.imgui();

    ImGui::Separator();

    if (ImGui::Button("Syntetic Linear"))
    {
        scene  = SyntheticPoseGraph::Linear(100, 3);
        change = true;
    }
    if (ImGui::Button("Syntetic Circle"))
    {
        scene  = SyntheticPoseGraph::Circle(5, 250, 6);
        change = true;
    }

    if (ImGui::Button("Syntetic CircleWithDrift"))
    {
        Saiga::Random::setSeed(39486);
        srand(39457);
        scene  = SyntheticPoseGraph::CircleWithDrift(5, 250, 6, 0.01, 0);
        change = true;
    }


    if (ImGui::Button("Syntetic CircleWithDriftAndScale"))
    {
        scene  = SyntheticPoseGraph::CircleWithDrift(5, 250, 6, 0.01, 0.005);
        change = true;
    }


    if (ImGui::Button("Print Scene Scale"))
    {
        int i = 0;
        for (auto v : scene.vertices)
        {
            std::cout << i << " : " << v.Sim3Pose().scale() << std::endl;
            i++;
        }
    }

#if 1
    {
        std::vector<const char*> strings;
        for (auto& d : datasets) strings.push_back(d.data());
        static int currentItem = 0;
        ImGui::Combo("Dataset", &currentItem, strings.data(), strings.size());
        if (ImGui::Button("Load Dataset"))
        {
            scene.load(datasets[currentItem]);
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
            Saiga::Scene sc            = bal.makeScene();
            scene                      = Saiga::PoseGraph(sc);
            scene.vertices[0].constant = true;
            change                     = true;
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
        {
            SAIGA_BLOCK_TIMER();
            ba.initAndSolve();
        }
        change = true;
    }

    //    barec.imgui();
    if (ImGui::Button("Solve Recursive"))
    {
        Saiga::PGORec barec;
        barec.optimizationOptions = baoptions;
        barec.create(scene);
        {
            SAIGA_BLOCK_TIMER();
            barec.initAndSolve();
        }
        change = true;
    }
    if (ImGui::Button("Solve Recursive Sim3"))
    {
        Saiga::PGOSim3Rec barec;
        barec.optimizationOptions = baoptions;
        barec.create(scene);
        {
            SAIGA_BLOCK_TIMER();
            barec.initAndSolve();
        }
        change = true;
    }


    if (ImGui::Button("Solve Ceres"))
    {
        Saiga::CeresPGO barec;
        barec.optimizationOptions = baoptions;
        barec.create(scene);
        {
            SAIGA_BLOCK_TIMER();
            barec.initAndSolve();
        }
        change = true;
    }


    ImGui::End();
}

int main(const int argc, const char* argv[])
{
    Saiga::initSaigaSample();
    Sample example;
    example.run();
    return 0;
}
