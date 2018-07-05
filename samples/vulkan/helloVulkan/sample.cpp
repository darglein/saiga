/*
* Vulkan Example - imGui (https://github.com/ocornut/imgui)
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "sample.h"
#include <saiga/imgui/imgui.h>
#include "saiga/util/color.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#error OpenGL was included somewhere.
#endif

VulkanExample::VulkanExample(Saiga::Vulkan::VulkanWindow &window, Saiga::Vulkan::VulkanForwardRenderer &renderer)
    :  Updating(window), Saiga::Vulkan::VulkanForwardRenderingInterface(renderer), renderer(renderer)
{
    float aspect = window.getAspectRatio();
    camera.setProj(60.0f,aspect,0.1f,50.0f,true);
    camera.setView(vec3(0,5,10),vec3(0,0,0),vec3(0,1,0));
    camera.rotationPoint = vec3(0);

    window.setCamera(&camera);
}

VulkanExample::~VulkanExample()
{
    teapot.destroy();
    plane.destroy();
    assetRenderer.destroy();
}

void VulkanExample::init(Saiga::Vulkan::Queue& queue, vk::CommandBuffer cmd)
{
    assetRenderer.init(renderer.vulkanDevice,renderer.pipelineCache,renderer.renderPass);


    teapot.loadObj("objs/teapot.obj");
    teapot.updateBuffer(renderer.vulkanDevice, renderer.graphicsQueue);
    teapotTrans.translateGlobal(vec3(0,1,0));
    teapotTrans.calculateModel();

    plane.createCheckerBoard(vec2(20,20),1.0f,Saiga::Colors::firebrick,Saiga::Colors::gray);
    plane.updateBuffer(renderer.vulkanDevice, renderer.graphicsQueue);
}



void VulkanExample::update(float dt)
{

    camera.update(dt);
    camera.interpolate(dt,0);
}

void VulkanExample::transfer(VkCommandBuffer cmd)
{
    assetRenderer.updateUniformBuffers(cmd,camera.view,camera.proj);
}


void VulkanExample::render(VkCommandBuffer cmd)
{
    assetRenderer.bind(cmd);
    if(displayModels)
    {
        assetRenderer.pushModel(cmd,teapotTrans.model);
        teapot.render(cmd);
        assetRenderer.pushModel(cmd,mat4(1));
        plane.render(cmd);
    }
}

void VulkanExample::renderGUI()
{

    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Example settings");
    ImGui::Checkbox("Render models", &displayModels);
    ImGui::End();
//    return;

    parentWindow.renderImGui();
//    ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
//    ImGui::ShowTestWindow();
}


void VulkanExample::keyPressed(SDL_Keysym key)
{
    switch(key.scancode){
    case SDL_SCANCODE_ESCAPE:
        parentWindow.close();
        break;
    default:
        break;
    }
}

void VulkanExample::keyReleased(SDL_Keysym key)
{
}


