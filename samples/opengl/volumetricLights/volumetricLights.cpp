/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "volumetricLights.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/shader/shaderLoader.h"

Sample::Sample(OpenGLWindow& window, Renderer& renderer) : Updating(window), DeferredRenderingInterface(renderer)
{
    // create a perspective camera
    float aspect = window.getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 50.0f);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.enableInput();

    // Set the camera from which view the scene is rendered
    window.setCamera(&camera);


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


    ShadowQuality sq = ShadowQuality::HIGH;

    Deferred_Renderer& r = static_cast<Deferred_Renderer&>(parentRenderer);
    sun                  = r.lighting.createDirectionalLight();
    sun->setDirection(vec3(-1, -3, -2));
    sun->setColorDiffuse(LightColorPresets::DirectSunlight);
    sun->setIntensity(0.5);
    sun->setAmbientIntensity(0.1f);
    sun->createShadowMap(2048, 2048, 1, sq);
    sun->enableShadows();

    pointLight = r.lighting.createPointLight();
    //        pointLight->setAttenuation(AttenuationPresets::Quadratic);
    pointLight->setAttenuation(vec3(0, 0, 5));
    pointLight->setIntensity(2);
    pointLight->setRadius(10);
    pointLight->setPosition(vec3(9, 3, 0));
    pointLight->setColorDiffuse(make_vec3(1));
    pointLight->calculateModel();
    //        pointLight->createShadowMap(256,256,sq);
    pointLight->createShadowMap(512, 512, sq);
    pointLight->enableShadows();
    pointLight->setVolumetric(true);

    spotLight = r.lighting.createSpotLight();
    spotLight->setAttenuation(vec3(0, 0, 5));
    spotLight->setIntensity(2);
    spotLight->setRadius(8);
    spotLight->setPosition(vec3(-10, 5, 0));
    spotLight->setColorDiffuse(make_vec3(1));
    spotLight->calculateModel();
    spotLight->createShadowMap(512, 512, sq);
    spotLight->enableShadows();
    spotLight->setVolumetric(true);

    boxLight = r.lighting.createBoxLight();
    boxLight->setIntensity(1.0);

    //        boxLight->setPosition(vec3(0,2,10));
    //        boxLight->rotateLocal(vec3(1,0,0),30);
    boxLight->setView(vec3(0, 2, 10), vec3(0, 0, 13), vec3(0, 1, 0));
    boxLight->setColorDiffuse(make_vec3(1));
    boxLight->setScale(vec3(5, 5, 8));
    boxLight->calculateModel();
    boxLight->createShadowMap(512, 512, sq);
    boxLight->enableShadows();
    boxLight->setVolumetric(true);


    r.lighting.renderVolumetric = true;


    std::cout << "Program Initialized!" << std::endl;
}

Sample::~Sample() {}

void Sample::update(float dt)
{
    // Update the camera position
    camera.update(dt);


    sun->fitShadowToCamera(&camera);
}

void Sample::interpolate(float dt, float interpolation)
{
    // Update the camera rotation. This could also be done in 'update' but
    // doing it in the interpolate step will reduce latency
    camera.interpolate(dt, interpolation);
}

void Sample::render(Camera* cam)
{
    // Render all objects from the viewpoint of 'cam'
    groundPlane.render(cam);
    cube1.render(cam);
    cube2.render(cam);
    sphere.render(cam);
}

void Sample::renderDepth(Camera* cam)
{
    // Render the depth of all objects from the viewpoint of 'cam'
    // This will be called automatically for shadow casting light sources to create shadow maps
    groundPlane.renderDepth(cam);
    cube1.renderDepth(cam);
    cube2.renderDepth(cam);
    sphere.render(cam);
}

void Sample::renderOverlay(Camera* cam)
{
    // The skybox is rendered after lighting and before post processing
    skybox.render(cam);
}

void Sample::renderFinal(Camera* cam)
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
            ShaderLoader::instance()->reload();
        }

        ImGui::End();
    }

    // parentWindow.renderImGui();
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
