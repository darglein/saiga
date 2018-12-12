/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "sample.h"

#include "saiga/image/imageTransformations.h"
#include "saiga/imgui/imgui.h"
#include "saiga/util/color.h"
#include "saiga/util/cv.h"
#include "saiga/vision/BALDataset.h"
#include "saiga/vision/Eigen_GLM.h"
#include "saiga/vision/g2o/g2oBA2.h"
#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

VulkanExample::VulkanExample(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer)
    : Updating(window), Saiga::Vulkan::VulkanForwardRenderingInterface(renderer), renderer(renderer)
{
    float aspect = window.getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 50.0f, true);
    camera.setView(vec3(0, 1, 3), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.rotationPoint = vec3(0);

    window.setCamera(&camera);

    //    Saiga::BALDataset bald("problem-49-7776-pre.txt");
    //    Saiga::BALDataset bald("problem-1723-156502-pre.txt");
    Saiga::BALDataset bald("problem-257-65132-pre.txt");

    scene = bald.makeScene();
    scene.removeNegativeProjections();
    cout << scene.rms() << endl;
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
    frustum.createFrustum(glm::perspective(70.0f, float(640) / float(480), 0.1f, 1.0f), 0.04, vec4(0, 1, 1, 1), false);
    frustum.init(renderer.base);

    pointCloud.init(base, 1000 * 1000);

    change = true;
}



void VulkanExample::update(float dt)
{
    camera.update(dt);
    camera.interpolate(dt, 0);
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
    ImGui::Begin("Vision BA Sample");



    if (ImGui::Button("Bundle Adjust"))
    {
        Saiga::g2oBA2 ba;
        ba.optimize(scene, 10);
        change = true;
    }

    if (ImGui::Button("RMS"))
    {
        scene.rms();
    }

    if (ImGui::Button("Normalize"))
    {
        auto m = scene.medianWorldPoint();
        cout << "median world point " << m.transpose() << endl;
        Saiga::SE3 T(Saiga::Quat::Identity(), -m);
        scene.transformScene(T);
        change = true;
    }



    ImGui::End();

    parentWindow.renderImGui();
}


void VulkanExample::keyPressed(SDL_Keysym key)
{
    switch (key.scancode)
    {
        case SDL_SCANCODE_ESCAPE:
            parentWindow.close();
            break;
        default:
            break;
    }
}

void VulkanExample::keyReleased(SDL_Keysym key) {}
