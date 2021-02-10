/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/sdl/all.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/forwardRendering/forwardRendering.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/WindowTemplate.h"
#include "saiga/opengl/window/sdl_window.h"
#include "saiga/opengl/world/LineSoup.h"
#include "saiga/opengl/world/pointCloud.h"
#include "saiga/opengl/world/proceduralSkybox.h"
using namespace Saiga;

class SampleSplitScreen : public StandaloneWindow<WindowManagement::SDL, DeferredRenderer>,
                          public SDL_KeyListener,
                          public SDL_ResizeListener
{
   public:
    SampleSplitScreen() : StandaloneWindow("config.ini")
    {
        renderer->params.userViewPort = true;

        rw = window->getWidth();
        rh = window->getHeight();
        setupCameras();


        // This simple AssetLoader can create assets from meshes and generate some generic debug assets
        ObjAssetLoader assetLoader;

        teapot.asset = assetLoader.loadColoredAsset("models/teapot.obj");
        //    teapot.asset = assetLoader.loadTexturedAsset("cat.obj");
        teapot.translateGlobal(vec3(0, 1, 0));
        teapot.calculateModel();

        groundPlane.asset = assetLoader.loadDebugPlaneAsset2(make_ivec2(20, 20), 1.0f, Colors::firebrick, Colors::gray);

        // create one directional light
        sun = std::make_shared<DirectionalLight>();
        renderer->lighting.AddLight(sun);
        sun->setDirection(vec3(-1, -3, -2));
        sun->setColorDiffuse(LightColorPresets::DirectSunlight);
        sun->setIntensity(1.0);
        sun->setAmbientIntensity(0.3f);
        sun->createShadowMap(2048, 2048);
        sun->castShadows = true;

        std::cout << "Program Initialized!" << std::endl;
    }

    void setupCameras()
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

    void update(float dt) override
    {
        // Update the camera position
        if (!ImGui::captureKeyboard()) cameras[activeCamera].update(dt);
        sun->fitShadowToCamera(&cameras[activeCamera]);
    }
    void interpolate(float dt, float interpolation) override
    {
        if (!ImGui::captureMouse()) cameras[activeCamera].interpolate(dt, interpolation);
    }

    void render(Camera* cam, RenderPass render_pass) override
    {
        if (render_pass == RenderPass::Deferred || render_pass == RenderPass::Shadow)
        {
            groundPlane.render(cam, render_pass);
            teapot.render(cam, render_pass);
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

    // because we set custom viewports, we also have to update them when the window size changes!
    bool resizeWindow(Uint32 windowId, int width, int height) override
    {
        rw = width;
        rh = height;
        setupCameras();
        return false;
    }

   private:
    // render width and height
    int rw, rh;

    int cameraCount  = 4;
    int activeCamera = 0;
    std::vector<SDLCamera<PerspectiveCamera>> cameras;

    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
    SimpleAssetObject teapot;

    ProceduralSkybox skybox;

    std::shared_ptr<DirectionalLight> sun;
    std::shared_ptr<Texture> t;
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();
    SampleSplitScreen window;
    window.run();
    return 0;
}
