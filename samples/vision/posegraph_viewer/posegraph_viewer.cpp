/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "posegraph_viewer.h"

#include "saiga/image/imageTransformations.h"
#include "saiga/imgui/imgui.h"
#include "saiga/util/color.h"
#include "saiga/util/cv.h"
#include "saiga/util/directory.h"
#include "saiga/util/fileChecker.h"
#include "saiga/vision/BALDataset.h"
#include "saiga/vision/Eigen_GLM.h"
#include "saiga/vision/ba/BAPoseOnly.h"
#include "saiga/vision/ceres/CeresBA.h"
#include "saiga/vision/g2o/g2oBA2.h"
#include "saiga/vision/g2o/g2oPoseGraph.h"
#include "saiga/vision/pgo/PGORecursive.h"
#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

VulkanExample::VulkanExample(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer)
    : VulkanSDLExampleBase(window, renderer)
{
    scene.load(Saiga::SearchPathes::data("vision/loop.posegraph"));
    //    scene.load(Saiga::SearchPathes::data("vision/slam_30_431.posegraph"));
    scene.poses[0].constant = true;
}

VulkanExample::~VulkanExample() {}

void VulkanExample::init(Saiga::Vulkan::VulkanBase& base)
{
    assetRenderer.init(base, renderer.renderPass);
    lineAssetRenderer.init(base, renderer.renderPass, 2);
    textureDisplay.init(base, renderer.renderPass);



    grid.createGrid(10, 10);
    grid.init(renderer.base);

    lineAsset.init(base, 100000);
    lineAsset.size = 0;

    //    frustum.createFrustum(camera.proj, 2, vec4(1), true);
    frustum.createFrustum(glm::perspective(70.0f, float(640) / float(480), 0.1f, 1.0f), 0.05, vec4(1, 1, 1, 1), false);
    frustum.init(renderer.base);

    pointCloud.init(base, 1000 * 1000 * 10);

    change = true;
}



void VulkanExample::update(float dt)
{
    VulkanSDLExampleBase::update(dt);

    if (change)
    {
        lines.clear();
        for (auto& e : scene.edges)
        {
            int i = e.from;
            int j = e.to;

            auto p1 = scene.poses[i].se3.inverse().translation();
            auto p2 = scene.poses[j].se3.inverse().translation();

            lines.emplace_back(vec3(p1(0), p1(1), p1(2)), vec3(0), vec3(0, 1, 0));
            lines.emplace_back(vec3(p2(0), p2(1), p2(2)), vec3(0), vec3(0, 1, 0));
        }
    }
}

void VulkanExample::transfer(vk::CommandBuffer cmd)
{
    assetRenderer.updateUniformBuffers(cmd, camera.view, camera.proj);
    lineAssetRenderer.updateUniformBuffers(cmd, camera.view, camera.proj);

    if (lines.size() > 0)
    {
        lineAsset.size = lines.size();
        std::copy(lines.begin(), lines.end(), lineAsset.pointCloud.begin());
        lineAsset.updateBuffer(cmd, 0, lineAsset.size);
    }
}


void VulkanExample::render(vk::CommandBuffer cmd)
{
    if (lineAssetRenderer.bind(cmd))
    {
        lineAssetRenderer.pushModel(cmd, mat4(1));
        grid.render(cmd);

        for (auto& i : scene.poses)
        {
            Saiga::SE3 se3 = i.se3;
            mat4 v         = Saiga::toglm(se3.matrix());
            v              = Saiga::cvViewToGLView(v);
            v              = mat4(inverse(v));

            vec4 color = i.constant ? vec4(0, 0, 1, 0) : vec4(1, 0, 0, 0);
            lineAssetRenderer.pushModel(cmd, v, color);
            frustum.render(cmd);
        }

        if (lines.size() > 0)
        {
            lineAssetRenderer.pushModel(cmd, mat4(1));
            lineAsset.render(cmd, 0, lineAsset.size);
        }
    }
}

void VulkanExample::renderGUI()
{
    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Vision BA Sample");


    scene.imgui();

    ImGui::Separator();

    baoptions.imgui();

    ImGui::Separator();

    if (ImGui::Button("reload"))
    {
        scene.load(Saiga::SearchPathes::data("vision/loop.posegraph"));
    }

    if (ImGui::Button("Bundle Adjust G2O"))
    {
        Saiga::g2oPGO ba;
        ba.solve(scene, baoptions);
        change = true;
    }

    //    barec.imgui();
    if (ImGui::Button("sba recursive"))
    {
        Saiga::PGORec barec;
        barec.solve(scene, baoptions);
        change = true;
    }



    ImGui::End();
}
