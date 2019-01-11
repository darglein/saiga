/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "ba.h"

#include "saiga/image/imageTransformations.h"
#include "saiga/imgui/imgui.h"
#include "saiga/util/color.h"
#include "saiga/util/cv.h"
#include "saiga/util/directory.h"
#include "saiga/vision/BALDataset.h"
#include "saiga/vision/Eigen_GLM.h"
#include "saiga/vision/ceres/CeresBA.h"
#include "saiga/vision/g2o/g2oBA2.h"

#include "BAPoseOnly.h"
#include "BARecursive.h"
#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

VulkanExample::VulkanExample(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer)
    : VulkanSDLExampleBase(window, renderer)
{
    //    Saiga::BALDataset bald("problem-49-7776-pre.txt");
    //    Saiga::BALDataset bald("problem-1723-156502-pre.txt");

    sscene.numCameras     = 1;
    sscene.numWorldPoints = 3;
    scene                 = sscene.circleSphere();
    scene.addWorldPointNoise(0.01);

    findBALDatasets();
}

VulkanExample::~VulkanExample() {}

void VulkanExample::init(Saiga::Vulkan::VulkanBase& base)
{
    assetRenderer.init(base, renderer.renderPass);
    lineAssetRenderer.init(base, renderer.renderPass, 2);
    pointCloudRenderer.init(base, renderer.renderPass, 2);
    textureDisplay.init(base, renderer.renderPass);



    grid.createGrid(10, 10);
    grid.init(renderer.base);

    //    frustum.createFrustum(camera.proj, 2, vec4(1), true);
    frustum.createFrustum(glm::perspective(70.0f, float(640) / float(480), 0.1f, 1.0f), 0.05, vec4(1, 0, 0, 1), false);
    frustum.init(renderer.base);

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
    pointCloudRenderer.updateUniformBuffers(cmd, camera.view, camera.proj);

    if (change)
    {
        int i = 0;
        for (auto& wp : scene.worldPoints)
        {
            Saiga::VertexNC v;
            v.position                 = vec4(Saiga::toglm(wp.p), 1);
            v.color                    = vec4(glm::linearRand(vec3(1), vec3(1)), 1);
            pointCloud.pointCloud[i++] = v;
        }
        pointCloud.size = i;
        change          = false;
        pointCloud.updateBuffer(cmd);
        rms = scene.rms();
    }
}


void VulkanExample::render(vk::CommandBuffer cmd)
{
    if (lineAssetRenderer.bind(cmd))
    {
        lineAssetRenderer.pushModel(cmd, mat4(1));
        grid.render(cmd);

        for (auto& i : scene.images)
        {
            auto& extr     = scene.extrinsics[i.extr];
            Saiga::SE3 se3 = extr.se3;
            mat4 v         = Saiga::toglm(se3.matrix());
            v              = Saiga::cvViewToGLView(v);
            v              = mat4(inverse(v));
            lineAssetRenderer.pushModel(cmd, v);
            frustum.render(cmd);
        }
    }



    if (pointCloudRenderer.bind(cmd))
    {
        pointCloudRenderer.pushModel(cmd, mat4(1));
        pointCloud.render(cmd);
    }
}

void VulkanExample::renderGUI()
{
    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Scene Loading");


    sscene.imgui();
    if (ImGui::Button("Syntetic - circleSphere"))
    {
        scene  = sscene.circleSphere();
        change = true;
    }

    ImGui::Separator();


    std::vector<const char*> strings;
    for (auto& d : datasets) strings.push_back(d.data());
    static int currentItem = 0;
    ImGui::Combo("dataset", &currentItem, strings.data(), strings.size());
    if (ImGui::Button("load BAL"))
    {
        Saiga::BALDataset bald(datasets[currentItem]);
        scene = bald.makeScene();
        scene.compress();
        scene.sortByWorldPointId();
        change = true;
    }

    ImGui::End();



    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Vision BA Sample");



    static int its = 1;
    ImGui::SliderInt("its", &its, 0, 10);

    if (ImGui::Button("Bundle Adjust G2O"))
    {
        Saiga::g2oBA2 ba;
        ba.optimize(scene, its);
        change = true;
    }

    if (ImGui::Button("Bundle Adjust ceres"))
    {
        Saiga::CeresBA ba;
        ba.optimize(scene, its);
        change = true;
    }


    if (ImGui::Button("poseOnlySparse"))
    {
        Saiga::BAPoseOnly ba;
        ba.poseOnlySparse(scene, its);
        change = true;
    }

    if (ImGui::Button("posePointDense"))
    {
        Saiga::BAPoseOnly ba;
        ba.posePointDense(scene, its);
        change = true;
    }

    if (ImGui::Button("posePointDenseBlock"))
    {
        Saiga::BAPoseOnly ba;
        ba.posePointDenseBlock(scene, its);
        change = true;
    }

    if (ImGui::Button("sba recursive"))
    {
        Saiga::BARec ba;
        ba.solve(scene, its);
        change = true;
    }

    if (ImGui::Button("posePointSparse"))
    {
        Saiga::BAPoseOnly ba;
        ba.posePointSparse(scene, its);
        change = true;
    }


    ImGui::End();

    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Scene");
    ImGui::Text("RMS: %f", rms);
    change |= scene.imgui();

    ImGui::End();
    Saiga::VulkanSDLExampleBase::renderGUI();
}

void VulkanExample::findBALDatasets()
{
    Saiga::Directory dir(".");
    dir.getFilesPrefix(datasets, "problem-");
    std::sort(datasets.begin(), datasets.end());
    cout << "Found " << datasets.size() << " BAL datasets" << endl;
}
