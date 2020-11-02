/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/String.h"
#include "saiga/opengl/animation/cameraAnimation.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/ffmpeg/ffmpegEncoder.h"
#include "saiga/opengl/ffmpeg/videoEncoder.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"

using namespace Saiga;


class SampleVideoRecording : public SampleWindowDeferred
{
   public:
    using Base = SampleWindowDeferred;


    SampleVideoRecording() : enc(window.get())
    {
        ObjAssetLoader assetLoader;


        auto cubeAsset = assetLoader.loadTexturedAsset("box.obj");

        cube1.asset = cubeAsset;
        cube2.asset = cubeAsset;
        cube1.translateGlobal(vec3(3, 1, 0));
        cube1.calculateModel();

        cube2.translateGlobal(vec3(3, 1, 5));
        cube2.calculateModel();

        auto sphereAsset = assetLoader.loadColoredAsset("teapot.obj");
        sphere.asset     = sphereAsset;
        sphere.translateGlobal(vec3(-2, 1, 0));
        sphere.rotateLocal(vec3(0, 1, 0), 180);
        sphere.calculateModel();



        std::cout << "Program Initialized!" << std::endl;
    }

    void update(float dt) override
    {
        Base::update(dt);
        enc.update();
        remainingFrames--;
        if (rotateCamera)
        {
            float speed = 360.0f / 10.0 * dt;
            camera.mouseRotateAroundPoint(speed, 0, vec3(0, 5, 0), vec3(0, 1, 0));
        }

        if (cameraInterpolation.isRunning())
        {
            cameraInterpolation.update(camera);
        }
    }


    void render(Camera* cam, RenderPass render_pass) override
    {
        Base::render(cam, render_pass);
        if (render_pass == RenderPass::Deferred || render_pass == RenderPass::Shadow)
        {
            cube1.render(cam);
            cube2.render(cam);
            sphere.render(cam);
        }
        else if (render_pass == RenderPass::GUI)
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
                cameraInterpolation.renderGui(camera);
                ImGui::End();
            }
        }
    }



   private:
    SimpleAssetObject cube1, cube2;
    SimpleAssetObject sphere;
    int remainingFrames;
    bool rotateCamera = false;
    std::shared_ptr<FFMPEGEncoder> encoder;
    VideoEncoder enc;
    Interpolation cameraInterpolation;
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();
    SampleVideoRecording window;
    window.run();
    return 0;
}
