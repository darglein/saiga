/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/sdl/sdl_camera.h"
#include "saiga/core/sdl/sdl_eventhandler.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/rendering/VRRendering/VRRenderer.h"
#include "saiga/opengl/rendering/overlay/deferredDebugOverlay.h"
#include "saiga/opengl/rendering/renderer.h"
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


    VRSample();
    ~VRSample();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};
