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
#include "saiga/image/imageTransformations.h"
#include <glm/gtc/matrix_transform.hpp>
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
}

void VulkanExample::init(Saiga::Vulkan::VulkanBase &base)
{
//    m_location1 = base.memory.vertexIndexAllocator.allocate(1024);
//    m_location2 = base.memory.vertexIndexAllocator.allocate(1024);
//    m_location3 = base.memory.vertexIndexAllocator.allocate(1024);
}



void VulkanExample::update(float dt)
{
    camera.update(dt);
    camera.interpolate(dt,0);

//    renderer.base.memory.vertexIndexAllocator.deallocate(m_location3);
//    m_location3 = renderer.base.memory.vertexIndexAllocator.allocate(1025);
}

void VulkanExample::transfer(vk::CommandBuffer cmd)
{

}


void VulkanExample::render(vk::CommandBuffer cmd)
{

}

void VulkanExample::renderGUI()
{

    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Example settings");
    ImGui::End();

    parentWindow.renderImGui();
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


