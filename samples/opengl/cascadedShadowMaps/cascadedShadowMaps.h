/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/sdl/sdl_camera.h"
#include "saiga/sdl/sdl_eventhandler.h"
#include "saiga/opengl/window/sdl_window.h"
#include "saiga/opengl/world/proceduralSkybox.h"

using namespace Saiga;

class Sample : public Updating, public DeferredRenderingInterface, public SDL_KeyListener
{
   public:
    SDLCamera<PerspectiveCamera> camera;

    //    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
    //    SimpleAssetObject sphere;
    std::vector<SimpleAssetObject> cubes;

    AABB sceneBB;
    ProceduralSkybox skybox;


    bool debugLightShader    = false;
    bool fitShadowToCamera   = true;
    bool fitNearPlaneToScene = true;

    std::shared_ptr<DirectionalLight> sun;

    Sample(OpenGLWindow& window, Renderer& renderer);
    ~Sample();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void render(Camera* cam) override;
    void renderDepth(Camera* cam) override;
    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};
