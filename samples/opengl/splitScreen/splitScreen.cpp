/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "splitScreen.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/shader/shaderLoader.h"


SplitScreen::SplitScreen() : StandaloneWindow("config.ini")
{
    renderer->params.userViewPort = true;

    rw = window->getWidth();
    rh = window->getHeight();
    setupCameras();


    // This simple AssetLoader can create assets from meshes and generate some generic debug assets
    ObjAssetLoader assetLoader;

    teapot.asset = assetLoader.loadBasicAsset("models/teapot.obj");
    //    teapot.asset = assetLoader.loadTexturedAsset("cat.obj");
    teapot.translateGlobal(vec3(0, 1, 0));
    teapot.calculateModel();

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

void SplitScreen::setupCameras()
{
    cameras.clear();


    int h = rh;
    int w = rw;


    float aspect = window->getAspectRatio();

    SDLCamera<PerspectiveCamera> defaultCamera;
    defaultCamera.setProj(60.0f, aspect, 0.1f, 50.0f);
    defaultCamera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    defaultCamera.rotationPoint = make_vec3(0);


    std::vector<std::pair<Camera*, ViewPort>> camerasVps;

    if (cameraCount == 1)
    {
        cameras.push_back(defaultCamera);
        camerasVps.emplace_back(&cameras[0], ViewPort({0, 0}, {w, h}));
    }
    else if (cameraCount == 2)
    {
        // the 2 camera setup has a different aspect ratio
        defaultCamera.setProj(60.0f, float(w) / (h / 2), 0.1f, 50.0f);

        cameras.push_back(defaultCamera);
        cameras.push_back(defaultCamera);

        camerasVps.emplace_back(&cameras[0], ViewPort({0, 0}, {w, h / 2}));
        camerasVps.emplace_back(&cameras[1], ViewPort({0, h / 2}, {w, h / 2}));
    }
    else if (cameraCount == 4)
    {
        int h2 = h / 2;
        int w2 = w / 2;

        for (int i = 0; i < 4; ++i) cameras.push_back(defaultCamera);

        camerasVps.emplace_back(&cameras[0], ViewPort({0, 0}, {w2, h2}));
        camerasVps.emplace_back(&cameras[1], ViewPort({w2, 0}, {w2, h2}));
        camerasVps.emplace_back(&cameras[2], ViewPort({0, h2}, {w2, h2}));
        camerasVps.emplace_back(&cameras[3], ViewPort({w2, h2}, {w2, h2}));
    }
    else if (cameraCount == 16)
    {
        int h4 = h / 4;
        int w4 = w / 4;

        for (int i = 0; i < 16; ++i) cameras.push_back(defaultCamera);

        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                camerasVps.emplace_back(&cameras[i * 4 + j], ViewPort({j * w4, i * h4}, {w4, h4}));
            }
        }
    }



    window->setMultiCamera(camerasVps);
    activeCamera = 0;
}

void SplitScreen::update(float dt)
{
    // Update the camera position
    if (!ImGui::captureKeyboard()) cameras[activeCamera].update(dt);
    sun->fitShadowToCamera(&cameras[activeCamera]);
}

void SplitScreen::interpolate(float dt, float interpolation)
{
    // Update the camera rotation. This could also be done in 'update' but
    // doing it in the interpolate step will reduce latency
    if (!ImGui::captureMouse()) cameras[activeCamera].interpolate(dt, interpolation);
}

void SplitScreen::render(Camera* cam)
{
    // Render all objects from the viewpoint of 'cam'
    groundPlane.render(cam);
    teapot.render(cam);
}

void SplitScreen::renderDepth(Camera* cam)
{
    // Render the depth of all objects from the viewpoint of 'cam'
    // This will be called automatically for shadow casting light sources to create shadow maps
    groundPlane.renderDepth(cam);
    teapot.renderDepth(cam);
}

void SplitScreen::renderOverlay(Camera* cam)
{
    // The skybox is rendered after lighting and before post processing
    skybox.sunDir = vec3(sun->getDirection());
    //    sun->setDirection(skybox.sunDir);
    skybox.render(cam);
}

void SplitScreen::renderFinal(Camera* cam)
{
    // The final render path (after post processing).
    // Usually the GUI is rendered here.

    window->renderImGui();

    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin("Split Screen");


        static int item             = 2;
        static const char* items[4] = {"1", "2", "4", "16"};
        static int itemsi[4]        = {1, 2, 4, 16};
        if (ImGui::Combo("Num Cameras", &item, items, 4))
        {
            cameraCount = itemsi[item];
            setupCameras();
        }
        ImGui::SliderInt("Active CAmera", &activeCamera, 0, cameraCount - 1);

        ImGui::End();
    }
}


void SplitScreen::keyPressed(SDL_Keysym key)
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

void SplitScreen::keyReleased(SDL_Keysym key) {}



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    SplitScreen window;
    window.run();

    return 0;
}
