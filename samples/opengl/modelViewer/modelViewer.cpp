/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "modelViewer.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/opengl/shader/shaderLoader.h"

Sample::Sample(OpenGLWindow& window, Renderer& renderer) : Updating(window), DeferredRenderingInterface(renderer)
{
    // create a perspective camera
    float aspect = window.getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 500.0f);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.enableInput();
    // How fast the camera moves
    camera.movementSpeed     = 10;
    camera.movementSpeedFast = 20;
    camera.mouseTurnLocal    = true;

    // Set the camera from which view the scene is rendered
    window.setCamera(&camera);


    ObjAssetLoader assetLoader;


    auto asset   = assetLoader.loadTexturedAsset("box.obj");
    object.asset = asset;



    //    groundPlane.asset = assetLoader.loadDebugPlaneAsset(make_vec2(20, 20), 1.0f, Colors::lightgray, Colors::gray);
    groundPlane.asset = assetLoader.loadDebugGrid(20, 20);

    // create one directional light
    Deferred_Renderer& r = static_cast<Deferred_Renderer&>(parentRenderer);
    sun                  = r.lighting.createDirectionalLight();
    sun->setDirection(vec3(-1, -3, -2));
    sun->setColorDiffuse(LightColorPresets::DirectSunlight);
    sun->setIntensity(0);
    sun->setAmbientIntensity(1);
    sun->createShadowMap(2048, 2048);
    sun->enableShadows();


    std::cout << "Program Initialized!" << std::endl;
}

Sample::~Sample() {}

void Sample::update(float dt)
{
    if (!ImGui::captureKeyboard()) camera.update(dt);
}

void Sample::interpolate(float dt, float interpolation)
{
    if (!ImGui::captureMouse()) camera.interpolate(dt, interpolation);
}

void Sample::render(Camera* cam)
{
    object.render(cam);
}

void Sample::renderDepth(Camera* cam)
{
    object.renderDepth(cam);
}

void Sample::renderOverlay(Camera* cam)
{
    if (showSkybox) skybox.render(cam);
    if (showGrid) groundPlane.renderForward(cam);
}

void Sample::renderFinal(Camera* cam)
{
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400, 400), ImGuiCond_FirstUseEver);
    ImGui::Begin("Model Viewer");

    ImGui::Checkbox("showSkybox", &showSkybox);
    ImGui::Checkbox("showGrid", &showGrid);

    ImGui::InputText("File", file.data(), file.size());


    if (ImGui::Button("Load OBJ with Texture"))
    {
        ObjAssetLoader assetLoader;
        auto asset = assetLoader.loadTexturedAsset(std::string(file.data()));
        if (asset) object.asset = asset;
    }

    if (ImGui::Button("Load OBJ with Vertex Color"))
    {
        ObjAssetLoader assetLoader;
        auto asset = assetLoader.loadBasicAsset(std::string(file.data()));
        if (asset) object.asset = asset;
    }


    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(0, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("Camera");

    camera.imgui();
    ImGui::End();
}


void Sample::keyPressed(SDL_Keysym key)
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

void Sample::keyReleased(SDL_Keysym key) {}
