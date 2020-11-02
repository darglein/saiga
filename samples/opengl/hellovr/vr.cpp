/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/sdl/sdl_camera.h"
#include "saiga/core/sdl/sdl_eventhandler.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/rendering/VRRendering/VRRenderer.h"
#include "saiga/opengl/rendering/overlay/deferredDebugOverlay.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/SampleWindowForward.h"
#include "saiga/opengl/window/sdl_window.h"
#include "saiga/opengl/world/proceduralSkybox.h"
using namespace Saiga;

class VRSample : public StandaloneWindow<WindowManagement::SDL, VRRenderer>, public SDL_KeyListener
{
   public:
    SDLCamera<PerspectiveCamera> camera;

    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
    SimpleAssetObject sphere;

    ProceduralSkybox skybox;


    float rotationSpeed    = 0.1;
    bool showimguidemo     = false;
    bool lightDebug        = false;
    bool pointLightShadows = false;


    VRSample() : StandaloneWindow("config.ini")
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

        auto sphereAsset = assetLoader.loadColoredAsset("teapot.obj");
        sphere.asset     = sphereAsset;
        sphere.translateGlobal(vec3(0, 1, 8));
        sphere.rotateLocal(vec3(0, 1, 0), 180);
        sphere.calculateModel();

        groundPlane.asset = assetLoader.loadDebugPlaneAsset(vec2(20, 20), 1.0f, Colors::lightgray, Colors::gray);


        std::cout << "Program Initialized!" << std::endl;
    }
    ~VRSample() {}

    void update(float dt) override
    {
        // Update the camera position
        camera.update(dt);
    }
    void interpolate(float dt, float interpolation) override
    {
        // Update the camera rotation. This could also be done in 'update' but
        // doing it in the interpolate step will reduce latency
        camera.interpolate(dt, interpolation);
    }

    void render(Camera* camera, RenderPass render_pass) override
    {
        if (render_pass == RenderPass::Forward)
        {
            // The skybox is rendered after lighting and before post processing
            groundPlane.renderForward(camera);
            skybox.render(camera);
        }
        else if (render_pass == RenderPass::GUI)
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
    }


    void keyPressed(SDL_Keysym key) override
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
    void keyReleased(SDL_Keysym key) override {}
};



int main(int argc, char* args[])
{
    initSaigaSample();
    VRSample window;
    window.run();
    return 0;
}
