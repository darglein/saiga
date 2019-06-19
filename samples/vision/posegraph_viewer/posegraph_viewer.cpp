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
#include "saiga/vision/BALDataset.h"
#include "saiga/vision/Eigen_GLM.h"
#include "saiga/vision/ceres/CeresBA.h"
#include "saiga/vision/g2o/g2oBA2.h"
#include "saiga/vision/g2o/g2oPoseGraph.h"
#include "saiga/vision/recursive/BAPoseOnly.h"
#include "saiga/vision/recursive/PGORecursive.h"
#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

VulkanExample::VulkanExample(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer)
    : VulkanSDLExampleBase(window, renderer)
{
    Saiga::SearchPathes::data.getFiles(datasets, "vision", ".posegraph");
    std::sort(datasets.begin(), datasets.end());
    std::cout << "Found " << datasets.size() << " posegraph datasets" << std::endl;


    Saiga::SearchPathes::data.getFiles(baldatasets, "vision", ".txt");
    std::sort(baldatasets.begin(), baldatasets.end());
    std::cout << "Found " << baldatasets.size() << " BAL datasets" << std::endl;

    init(renderer.base());
}

VulkanExample::~VulkanExample() {}

void VulkanExample::init(Saiga::Vulkan::VulkanBase& base)
{
    assetRenderer.init(base, renderer.renderPass);
    lineAssetRenderer.init(base, renderer.renderPass, 2);
    textureDisplay.init(base, renderer.renderPass);



    grid.createGrid(10, 10);
    grid.init(renderer.base());



    //    frustum.createFrustum(perspective(70.0f, float(640) / float(480), 0.1f, 1.0f), 0.02, vec4(1, 0, 0, 1), false);
    //    frustum.init(renderer.base());

    lineAsset.init(base, 10 * 1000 * 1000);
    lineAsset.size = 0;

    frustum.createFrustum(perspective(70.0f, float(640) / float(480), 0.1f, 1.0f), 0.05, vec4(1, 1, 1, 1), false);
    frustum.init(renderer.base());


    pointCloud.init(base, 1000 * 1000 * 10);

    change = true;
}



void VulkanExample::update(float dt)
{
    VulkanSDLExampleBase::update(dt);
}

void VulkanExample::transfer(vk::CommandBuffer cmd)
{
    assetRenderer.updateUniformBuffers(cmd, camera.view, camera.proj);
    lineAssetRenderer.updateUniformBuffers(cmd, camera.view, camera.proj);

    if (change)
    {
        chi2 = scene.chi2();
        rms  = scene.rms();

        lines.clear();
        for (auto& e : scene.edges)
        {
            int i = e.from;
            int j = e.to;

            auto p1 = scene.poses[i].se3.inverse().translation();
            auto p2 = scene.poses[j].se3.inverse().translation();

            lines.emplace_back(vec3(p1(0), p1(1), p1(2)), make_vec3(0), vec3(0, 1, 0));
            lines.emplace_back(vec3(p2(0), p2(1), p2(2)), make_vec3(0), vec3(0, 1, 0));
        }

        if (lines.size() > 0)
        {
            std::cout << "num lines: " << lines.size() << std::endl;
            lineAsset.size = lines.size();
            std::copy(lines.begin(), lines.end(), lineAsset.pointCloud.begin());
            lineAsset.updateBuffer(cmd, 0, lineAsset.size);
        }

        change = false;
    }
}


void VulkanExample::render(vk::CommandBuffer cmd)
{
    if (lineAssetRenderer.bind(cmd))
    {
        lineAssetRenderer.pushModel(cmd, identityMat4());
        grid.render(cmd);

        for (auto& i : scene.poses)
        {
            Saiga::SE3 se3 = i.se3;
            mat4 v         = Saiga::toglm(se3.matrix());
            v              = Saiga::cvViewToGLView(v);
            v              = mat4(inverse(v));

            //            std::cout << v << std::endl;
            vec4 color = i.constant ? vec4(0, 0, 1, 0) : vec4(1, 0, 0, 0);
            lineAssetRenderer.pushModel(cmd, v, color);
            frustum.render(cmd);
        }

        if (lines.size() > 0)
        {
            lineAssetRenderer.pushModel(cmd, identityMat4());
            lineAsset.render(cmd, 0, lineAsset.size);
        }
    }
}

void VulkanExample::renderGUI()
{
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



    ImGui::End();
}
