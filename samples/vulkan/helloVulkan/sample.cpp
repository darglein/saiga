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
    box.destroy();
    teapot.destroy();
    plane.destroy();
    assetRenderer.destroy();
    lineAssetRenderer.destroy();
    pointCloudRenderer.destroy();
    texturedAssetRenderer.destroy();
    grid.destroy();
    frustum.destroy();
    pointCloud.destroy();
}

void VulkanExample::init(Saiga::Vulkan::VulkanBase &base)
{
    assetRenderer.init(base,renderer.renderPass);
    lineAssetRenderer.init(base,renderer.renderPass,2);
    pointCloudRenderer.init(base,renderer.renderPass,5);
    texturedAssetRenderer.init(base,renderer.renderPass);


    box.loadObj("objs/box.obj");
    box.updateBuffer(renderer.base);
    box.descriptor = texturedAssetRenderer.createAndUpdateDescriptorSet(box.textures[0]);

    teapot.loadObj("objs/teapot.obj");
    teapot.updateBuffer(renderer.base);
    teapotTrans.translateGlobal(vec3(0,1,0));
    teapotTrans.calculateModel();

    plane.createCheckerBoard(vec2(20,20),1.0f,Saiga::Colors::firebrick,Saiga::Colors::gray);
    plane.updateBuffer(renderer.base);

    grid.createGrid(10,10);
    grid.updateBuffer(renderer.base);

    frustum.createFrustum(camera.proj,2,vec4(1),true);
    frustum.updateBuffer(renderer.base);




    for(int i = 0; i < 1000; ++i)
    {
        Saiga::VertexNC v;
        v.position = vec4(glm::linearRand(vec3(-3),vec3(3)),1);
        v.color = vec4(glm::linearRand(vec3(0),vec3(1)),1);
        pointCloud.mesh.points.push_back(v);
    }
    pointCloud.updateBuffer(renderer.base);
//    pointCloud.updateBuffer();
}



void VulkanExample::update(float dt)
{

    camera.update(dt);
    camera.interpolate(dt,0);
}

void VulkanExample::transfer(VkCommandBuffer cmd)
{
    assetRenderer.updateUniformBuffers(cmd,camera.view,camera.proj);
    lineAssetRenderer.updateUniformBuffers(cmd,camera.view,camera.proj);
    pointCloudRenderer.updateUniformBuffers(cmd,camera.view,camera.proj);
    texturedAssetRenderer.updateUniformBuffers(cmd,camera.view,camera.proj);

    if(change)
    {
        renderer.waitIdle();
        for(int i = 0; i < 1000; ++i)
        {
            Saiga::VertexNC v;
            v.position = vec4(glm::linearRand(vec3(-3),vec3(3)),1);
            v.color = vec4(glm::linearRand(vec3(0),vec3(1)),1);
            pointCloud.mesh.points.push_back(v);
        }
        pointCloud.updateBuffer(renderer.base);

        change = false;
    }
}


void VulkanExample::render(VkCommandBuffer cmd)
{
    if(displayModels)
    {
        assetRenderer.bind(cmd);
        assetRenderer.pushModel(cmd,teapotTrans.model);
        teapot.render(cmd);
        //        assetRenderer.pushModel(cmd,mat4(1));
        //        plane.render(cmd);

        lineAssetRenderer.bind(cmd);

        lineAssetRenderer.pushModel(cmd,mat4(1));
        grid.render(cmd);

        lineAssetRenderer.pushModel(cmd,mat4(1));
        frustum.render(cmd);



        pointCloudRenderer.bind(cmd);

        pointCloudRenderer.pushModel(cmd,mat4(1));
        pointCloud.render(cmd);

        texturedAssetRenderer.bind(cmd);
        texturedAssetRenderer.pushModel(cmd,mat4(1));
        texturedAssetRenderer.bindTexture(cmd,box.descriptor);
        box.render(cmd);
    }
}

void VulkanExample::renderGUI()
{

    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Example settings");
    ImGui::Checkbox("Render models", &displayModels);



    if(ImGui::Button("change point cloud"))
    {
        change = true;
    }



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


