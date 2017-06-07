#pragma once

#include "saiga/rendering/renderer.h"
#include "saiga/world/proceduralSkybox.h"

#include "saiga/assets/all.h"
#include "saiga/assets/objAssetLoader.h"

#include "saiga/sdl/sdl_eventhandler.h"
#include "saiga/sdl/sdl_camera.h"
#include "saiga/sdl/sdl_window.h"

#include "saiga/rendering/lighting/directional_light.h"
#include "saiga/imgui/imgui_impl_sdl_gl3.h"

class SimpleWindow : public Program, public SDL_KeyListener
{
public:
    SDLCamera<PerspectiveCamera> camera;

//    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
//    SimpleAssetObject sphere;
    std::vector<SimpleAssetObject> cubes;

    ImGui_SDL_Renderer imgui;
    aabb sceneBB;
    ProceduralSkybox skybox;


    bool fitShadowToCamera = true;
    bool fitNearPlaneToScene = true;

    std::shared_ptr<DirectionalLight> sun;

    SimpleWindow(OpenGLWindow* window);
    ~SimpleWindow();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void render(Camera *cam) override;
    void renderDepth(Camera *cam) override;
    void renderOverlay(Camera *cam) override;
    void renderFinal(Camera *cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};


