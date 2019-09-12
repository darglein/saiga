/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "SampleWindowDeferred.h"

namespace Saiga
{
SampleWindowDeferred::SampleWindowDeferred() : StandaloneWindow("config.ini")
{
    // create a perspective camera
    float aspect = window->getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 100.0f);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.rotationPoint = make_vec3(0);

    // Set the camera from which view the scene is rendered
    window->setCamera(&camera);


    // This simple AssetLoader can create assets from meshes and generate some generic debug assets
    ObjAssetLoader assetLoader;
    groundPlane.asset = assetLoader.loadDebugPlaneAsset2(make_ivec2(20, 20), 1.0f, Colors::firebrick, Colors::gray);

    // create one directional light
    sun = renderer->lighting.createDirectionalLight();
    sun->setDirection(vec3(-1, -3, -2));
    sun->setColorDiffuse(LightColorPresets::DirectSunlight);
    sun->setIntensity(1.0);
    sun->setAmbientIntensity(0.3f);
    sun->createShadowMap(2048, 2048);
    sun->enableShadows();

    std::cout << "Program Initialized!" << std::endl;
}

SampleWindowDeferred::~SampleWindowDeferred() {}

void SampleWindowDeferred::update(float dt)
{
    // Update the camera position
    if (!ImGui::captureKeyboard()) camera.update(dt);
    sun->fitShadowToCamera(&camera);
}

void SampleWindowDeferred::interpolate(float dt, float interpolation)
{
    // Update the camera rotation. This could also be done in 'update' but
    // doing it in the interpolate step will reduce latency
    if (!ImGui::captureMouse()) camera.interpolate(dt, interpolation);
}

void SampleWindowDeferred::render(Camera* cam)
{
    // Render all objects from the viewpoint of 'cam'
    if (showGrid) groundPlane.render(cam);
}

void SampleWindowDeferred::renderDepth(Camera* cam)
{
    // Render the depth of all objects from the viewpoint of 'cam'
    // This will be called automatically for shadow casting light sources to create shadow maps
    if (showGrid) groundPlane.renderDepth(cam);
}

void SampleWindowDeferred::renderOverlay(Camera* cam)
{
    // The skybox is rendered after lighting and before post processing
    if (showSkybox)
    {
        skybox.sunDir = vec3(sun->getDirection());
        skybox.render(cam);
    }
}

void SampleWindowDeferred::renderFinal(Camera* cam)
{
    // The final render path (after post processing).
    // Usually the GUI is rendered here.

    window->renderImGui();

    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin("Saiga Sample Base");
        ImGui::Checkbox("showSkybox", &showSkybox);
        ImGui::Checkbox("showGrid", &showGrid);
        camera.imgui();
        ImGui::End();
    }
}


void SampleWindowDeferred::keyPressed(SDL_Keysym key)
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

void SampleWindowDeferred::keyReleased(SDL_Keysym key) {}

}  // namespace Saiga
