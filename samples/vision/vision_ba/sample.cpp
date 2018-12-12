/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "sample.h"

#include "saiga/image/imageTransformations.h"
#include "saiga/util/color.h"

#include <saiga/imgui/imgui.h>

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

VulkanExample::VulkanExample(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer)
    : Updating(window), Saiga::Vulkan::VulkanForwardRenderingInterface(renderer), renderer(renderer)
{
    float aspect = window.getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 50.0f, true);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.rotationPoint = vec3(0);

    window.setCamera(&camera);
}

VulkanExample::~VulkanExample() {}

void VulkanExample::init(Saiga::Vulkan::VulkanBase& base)
{
    assetRenderer.init(base, renderer.renderPass);
    lineAssetRenderer.init(base, renderer.renderPass, 2);
    pointCloudRenderer.init(base, renderer.renderPass, 5);
    textureDisplay.init(base, renderer.renderPass);



    grid.createGrid(10, 10);
    grid.init(renderer.base);

    frustum.createFrustum(camera.proj, 2, vec4(1), true);
    frustum.init(renderer.base);

    pointCloud.init(base, 1000);
    for (int i = 0; i < 1000; ++i)
    {
        Saiga::VertexNC v;
        v.position               = vec4(glm::linearRand(vec3(-3), vec3(3)), 1);
        v.color                  = vec4(glm::linearRand(vec3(1), vec3(1)), 1);
        pointCloud.pointCloud[i] = v;
    }
    pointCloud.size = 1000;

    auto cmd = base.createAndBeginTransferCommand();
    pointCloud.updateBuffer(cmd);
    base.endTransferWait(cmd);
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
}


void VulkanExample::render(vk::CommandBuffer cmd)
{
    if (lineAssetRenderer.bind(cmd))
    {
        lineAssetRenderer.pushModel(cmd, mat4(1));
        grid.render(cmd);

        lineAssetRenderer.pushModel(cmd, mat4(1));
        frustum.render(cmd);
    }



    if (pointCloudRenderer.bind(cmd))
    {
        pointCloudRenderer.pushModel(cmd, mat4(1));
        pointCloud.render(cmd, 0, pointCloud.capacity);
    }
}

void VulkanExample::renderGUI()
{
    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Example settings");
    ImGui::Checkbox("Render models", &displayModels);



    if (ImGui::Button("change point cloud"))
    {
        change = true;
    }


    if (ImGui::Button("reload shader"))
    {
        assetRenderer.reload();
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
