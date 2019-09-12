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

Sample::Sample() : enc(window.get())
{
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



    testBspline();

    std::cout << "Program Initialized!" << std::endl;
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
    Base::update(dt);


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



void Sample::render(Camera* cam)
{
    Base::render(cam);

    cube1.render(cam);
    cube2.render(cam);
    sphere.render(cam);
}

void Sample::renderDepth(Camera* cam)
{
    Base::renderDepth(cam);
    cube1.renderDepth(cam);
    cube2.renderDepth(cam);
    sphere.render(cam);
}

void Sample::renderOverlay(Camera* cam)
{
    Base::renderOverlay(cam);
    // The skybox is rendered after lighting and before post processing
    cameraInterpolation.render();
}



void Sample::renderFinal(Camera* cam)
{
    Base::renderFinal(cam);
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



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample window;
    window.run();

    return 0;
}
