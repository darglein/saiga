/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "videoRecording.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/String.h"
#include "saiga/opengl/shader/shaderLoader.h"

Sample::Sample(OpenGLWindow& window, Renderer& renderer)
    : Updating(window), DeferredRenderingInterface(renderer), enc(&window)
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
    cube1.translateGlobal(vec3(3, 1, 0));
    cube1.calculateModel();

    cube2.translateGlobal(vec3(3, 1, 5));
    cube2.calculateModel();

    auto sphereAsset = assetLoader.loadBasicAsset("teapot.obj");
    sphere.asset     = sphereAsset;
    sphere.translateGlobal(vec3(-2, 1, 0));
    sphere.rotateLocal(vec3(0, 1, 0), 180);
    sphere.calculateModel();

    groundPlane.asset = assetLoader.loadDebugPlaneAsset(vec2(20, 20), 1.0f, Colors::lightgray, Colors::gray);

    // create one directional light
    Deferred_Renderer& r = static_cast<Deferred_Renderer&>(parentRenderer);
    sun                  = r.lighting.createDirectionalLight();
    sun->setDirection(vec3(-1, -3, -2));
    sun->setColorDiffuse(LightColorPresets::DirectSunlight);
    sun->setIntensity(0.5);
    sun->setAmbientIntensity(0.1f);
    sun->createShadowMap(2048, 2048);
    sun->enableShadows();


    testBspline();

    std::cout << "Program Initialized!" << std::endl;
}

Sample::~Sample()
{
    // We don't need to delete anything here, because objects obtained from saiga are wrapped in smart pointers.
}

void Sample::testBspline()
{
    std::cout << "Testing Bspline..." << std::endl;


    {
        quat q1(151, 621, -16, 16);
        q1 = normalize(q1);

        quat q2(-25, 1617, 15, -781);
        q2 = normalize(q2);

        std::cout << "mix   " << q1 << " " << q2 << " " << normalize(mix(q1, q2, 0.3f)) << std::endl;
        std::cout << "slerp " << q1 << " " << q2 << " " << normalize(slerp(q1, q2, 0.3f)) << std::endl;
    }

    {
        std::cout << "Linear bspline" << std::endl;
        Bspline<vec2> spline(1, {{0, 0}, {1, 0}, {1, 1}, {2, 2}});
        spline.normalize();
        std::cout << spline << std::endl;

        int steps = 50;
        for (int i = 0; i < steps; ++i)
        {
            float alpha = float(i) / (steps - 1);
            std::cout << i << " " << alpha << " " << spline.getPointOnCurve(alpha) << std::endl;
        }
    }

    {
        std::cout << "Cubic bspline" << std::endl;
        Bspline<vec2> spline(3, {{0, 0}, {1, 0}, {1, 1}, {2, 2}});
        spline.normalize();
        std::cout << spline << std::endl;

        int steps = 50;
        for (int i = 0; i < steps; ++i)
        {
            float alpha = float(i) / (steps - 1);
            std::cout << i << " " << alpha << " " << spline.getPointOnCurve(alpha) << std::endl;
        }
    }

    //    cameraInterpolation.positionSpline.addPoint({0,3,0});
    //    cameraInterpolation.positionSpline.addPoint({1,3,0});
    //    cameraInterpolation.positionSpline.addPoint({1,3,1});
    //    cameraInterpolation.positionSpline.addPoint({0,3,1});
    //    cameraInterpolation.positionSpline.normalize(true);

    //    std::cout << cameraInterpolation.positionSpline << std::endl;
    //    cameraInterpolation.createAsset();


    //    exit(0);
}

void Sample::update(float dt)
{
    // Update the camera position
    camera.update(dt);
    sun->fitShadowToCamera(&camera);

    enc.update();

    remainingFrames--;

    if (rotateCamera)
    {
        float speed = 360.0f / 10.0 * dt;
        //        float speed = 2 * pi<float>();
        camera.mouseRotateAroundPoint(speed, 0, vec3(0, 5, 0), vec3(0, 1, 0));
    }


    if (cameraInterpolation.isRunning())
    {
        cameraInterpolation.update(camera);
    }
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


    cameraInterpolation.render();
}



void Sample::renderFinal(Camera* cam)
{
    {
        ImGui::SetNextWindowPos(ImVec2(50, 400), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin("Video Encoding");
        enc.renderGUI();

        ImGui::End();
    }



    {
        ImGui::SetNextWindowPos(ImVec2(50, 400), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin("Camera");


        if (ImGui::Button("load path"))
        {
            cameraInterpolation.keyframes.push_back({{0.973249, -0.229753, 0, 0}, {0, 5, 10}});
            //            cameraInterpolation.keyframes.push_back({ quat(0.973249,-0.229753,0,0), vec3(0,5,10)});
            cameraInterpolation.keyframes.push_back(
                {quat(0.950643, -0.160491, 0.261851, 0.0442066), vec3(8.99602, 5.61079, 9.23351)});
            cameraInterpolation.keyframes.push_back(
                {quat(0.6868, -0.0622925, 0.721211, 0.0654136), vec3(13.4404, 5.61079, -0.559972)});
            cameraInterpolation.updateCurve();
        }

        cameraInterpolation.renderGui(camera);

        ImGui::End();
    }
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
