/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_SDL

#    include "SampleWindowDeferred.h"

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
    sun = std::make_shared<DirectionalLight>();
    renderer->lighting.AddLight(sun);
    sun->createShadowMap(2048, 2048);
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


void SampleWindowDeferred::render(Camera* cam, RenderPass render_pass)
{
    if (render_pass == RenderPass::Deferred || render_pass == RenderPass::Shadow)
    {
        groundPlane.render(cam, render_pass);
    }
    else if (render_pass == RenderPass::Forward)
    {
        skybox.sunDir = vec3(sun->getDirection());
        skybox.render(cam);
    }
    else if (render_pass == RenderPass::GUI)
    {
        window->renderImGui();


        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin("Saiga Sample Base");
        ImGui::Checkbox("showSkybox", &showSkybox);
        ImGui::Checkbox("showGrid", &showGrid);
        camera.imgui();
        skybox.imgui();
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
#endif
