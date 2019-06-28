/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "ba.h"

#include "saiga/core/image/imageTransformations.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/core/util/color.h"
#include "saiga/core/util/cv.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/BALDataset.h"
#include "saiga/vision/Eigen_GLM.h"
#include "saiga/vision/ceres/CeresBA.h"
#include "saiga/vision/g2o/g2oBA2.h"
#include "saiga/vision/recursive/BAPoseOnly.h"
#include "saiga/vision/scene/PoseGraph.h"

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
    sscene.numImagePoints = 2;
    scene                 = sscene.circleSphere();
    scene.addWorldPointNoise(0.01);

    Saiga::SearchPathes::data.getFiles(datasets, "vision/bal", ".txt");
    std::sort(datasets.begin(), datasets.end());
    std::cout << "Found " << datasets.size() << " BAL datasets" << std::endl;

    init(renderer.base());
}

VulkanExample::~VulkanExample() {}

void VulkanExample::init(Saiga::Vulkan::VulkanBase& base)
{
    assetRenderer.init(base, renderer.renderPass);
    lineAssetRenderer.init(base, renderer.renderPass, 2);
    pointCloudRenderer.init(base, renderer.renderPass, 1);
    textureDisplay.init(base, renderer.renderPass);



    grid.createGrid(10, 10);
    grid.init(renderer.base());

    frustum.createFrustum(perspective(70.0f, float(640) / float(480), 0.1f, 1.0f), 0.02, vec4(1, 0, 0, 1), false);
    frustum.init(renderer.base());

    pointCloud.init(base, 1000 * 1000 * 10);
    graphLines.init(base, 1000000);
    change = true;
}



void VulkanExample::update(float dt)
{
    VulkanSDLExampleBase::update(dt);

    float speed = 360.0f / 10.0 * dt;
    //        float speed = 2 * pi<float>();
    camera.mouseRotateAroundPoint(speed * 0.5, 0, vec3(0, 5, 0), vec3(0, 1, 0));
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
            v.position                 = make_vec4(Saiga::toglm(wp.p), 1);
            v.color                    = make_vec4(linearRand(make_vec3(1), make_vec3(1)), 1);
            pointCloud.pointCloud[i++] = v;
        }
        pointCloud.size = i;
        pointCloud.updateBuffer(cmd);
        rms    = scene.rms();
        change = false;

        PoseGraph pg(scene, minMatchEdge);
        i = 0;
        for (auto e : pg.edges)
        {
            if (!scene.images[e.from].valid() || !scene.images[e.to].valid()) continue;
            auto p1 = pg.poses[e.from].se3.inverse().translation();
            auto p2 = pg.poses[e.to].se3.inverse().translation();

            if ((p1 - p2).norm() > maxEdgeDistance) continue;
            //            if (p1.norm() > 10 || p2.norm() > 10) continue;

            auto& pc1 = graphLines.pointCloud[i];
            auto& pc2 = graphLines.pointCloud[i + 1];

            pc1.position = make_vec4(p1.cast<float>(), 1);
            pc2.position = make_vec4(p2.cast<float>(), 1);

            pc1.color = vec4(0, 1, 0, 1);
            pc2.color = pc1.color;

            pc1.normal = vec4(1, 1, 1, 1);
            pc2.normal = vec4(1, 1, 1, 1);
            //            lines.emplace_back(make_vec4(p1.cast<float>(),1),make_vec4(0),make_vec4(e.color.cast<float>(),1));
            //            lines.emplace_back(make_vec4(p2.cast<float>(),1),make_vec4(0),make_vec4(e.color.cast<float>(),1));
            i += 2;
        }
        graphLines.size = i;
        graphLines.updateBuffer(cmd, 0, graphLines.size);
        std::cout << "pg size " << i << std::endl;
    }
}


void VulkanExample::render(vk::CommandBuffer cmd)
{
    if (lineAssetRenderer.bind(cmd))
    {
        lineAssetRenderer.pushModel(cmd, identityMat4());
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


        lineAssetRenderer.pushModel(cmd, mat4::Identity());
        if (graphLines.size > 0) graphLines.render(cmd, 0, graphLines.size);
    }



    if (pointCloudRenderer.bind(cmd))
    {
        pointCloudRenderer.pushModel(cmd, identityMat4());
        pointCloud.render(cmd);
    }
}

void VulkanExample::renderGUI()
{
    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("Scene Loading");

    ImGui::InputInt("minMatchEdge", &minMatchEdge);
    ImGui::InputFloat("maxEdgeDistance", &maxEdgeDistance);

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



    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("Vision BA Sample");



    static int its = 1;
    ImGui::SliderInt("its", &its, 0, 10);

    if (ImGui::Button("Bundle Adjust G2O"))
    {
        SAIGA_BLOCK_TIMER();
        Saiga::g2oBA2 ba;
        ba.baOptions = baoptions;
        ba.create(scene);
        ba.initAndSolve();
        change = true;
    }

    if (ImGui::Button("Bundle Adjust ceres"))
    {
        SAIGA_BLOCK_TIMER();
        Saiga::CeresBA ba;
        ba.baOptions = baoptions;
        ba.create(scene);
        ba.initAndSolve();
        change = true;
    }


    //    if (ImGui::Button("poseOnlySparse"))
    //    {
    //        SAIGA_BLOCK_TIMER();
    //        Saiga::BAPoseOnly ba;
    //        ba.poseOnlySparse(scene, its);
    //        change = true;
    //    }



    baoptions.imgui();
    if (ImGui::Button("sba recursive"))
    {
        SAIGA_BLOCK_TIMER();
        //        Saiga::BARec barec;
        barec.baOptions = baoptions;
        barec.create(scene);
        barec.initAndSolve();
        change = true;
    }

    if (ImGui::Button("posePointSparse"))
    {
        SAIGA_BLOCK_TIMER();
        Saiga::BAPoseOnly ba;
        ba.baOptions = baoptions;
        ba.create(scene);
        ba.initAndSolve();
        change = true;
    }


    ImGui::End();

    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("Scene");
    ImGui::Text("RMS: %f", rms);
    change |= scene.imgui();

    ImGui::End();
    Saiga::VulkanSDLExampleBase::renderGUI();
}
