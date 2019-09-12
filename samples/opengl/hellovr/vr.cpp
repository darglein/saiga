/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "vr.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/shader/shaderLoader.h"

VRSample::VRSample() : StandaloneWindow("config.ini")
{
    // create a perspective camera
    float aspect = window->getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 50.0f);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.enableInput();

    // Set the camera from which view the scene is rendered
    window->setCamera(&camera);


    ObjAssetLoader assetLoader;


    auto cubeAsset = assetLoader.loadTexturedAsset("box.obj");

    cube1.asset = cubeAsset;
    cube2.asset = cubeAsset;
    cube1.translateGlobal(vec3(11, 1, -2));
    cube1.calculateModel();

    cube2.translateGlobal(vec3(-11, 1, 2));
    cube2.calculateModel();

    auto sphereAsset = assetLoader.loadBasicAsset("teapot.obj");
    sphere.asset     = sphereAsset;
    sphere.translateGlobal(vec3(0, 1, 8));
    sphere.rotateLocal(vec3(0, 1, 0), 180);
    sphere.calculateModel();

    groundPlane.asset = assetLoader.loadDebugPlaneAsset(vec2(20, 20), 1.0f, Colors::lightgray, Colors::gray);


    std::cout << "Program Initialized!" << std::endl;
}

VRSample::~VRSample() {}

void VRSample::update(float dt)
{
    // Update the camera position
    camera.update(dt);
}

void VRSample::interpolate(float dt, float interpolation)
{
    // Update the camera rotation. This could also be done in 'update' but
    // doing it in the interpolate step will reduce latency
    camera.interpolate(dt, interpolation);
}


void VRSample::renderOverlay(Camera* cam)
{
    // The skybox is rendered after lighting and before post processing
    groundPlane.renderForward(cam);
    skybox.render(cam);
}

void VRSample::renderFinal(Camera* cam)
{
    // The final render path (after post processing).
    // Usually the GUI is rendered here.



    {
        ImGui::SetNextWindowPos(ImVec2(50, 400), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin("An Imgui Window :D");

        ImGui::SliderFloat("Rotation Speed", &rotationSpeed, 0, 10);

        if (ImGui::Button("Reload Shaders"))
        {
            shaderLoader.reload();
        }

        ImGui::End();
    }

    // parentWindow.renderImGui();
}


void VRSample::keyPressed(SDL_Keysym key)
{
    switch (key.scancode)
    {
        case SDL_SCANCODE_ESCAPE:
            window->close();
            break;
        default:
            break;
    }
}

void VRSample::keyReleased(SDL_Keysym key) {}

int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    VRSample window;
    window.run();

    return 0;
}
