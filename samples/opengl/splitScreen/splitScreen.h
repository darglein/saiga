/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/sdl/all.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/forwardRendering/forwardRendering.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/window/WindowTemplate.h"
#include "saiga/opengl/window/sdl_window.h"
#include "saiga/opengl/world/LineSoup.h"
#include "saiga/opengl/world/pointCloud.h"
#include "saiga/opengl/world/proceduralSkybox.h"
using namespace Saiga;

class SplitScreen : public StandaloneWindow<WindowManagement::SDL, DeferredRenderer>,
                    public SDL_KeyListener,
                    public SDL_ResizeListener
{
   public:
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

    SplitScreen();

    void setupCameras();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void render(Camera* cam) override;
    void renderDepth(Camera* cam) override;
    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;

    // because we set custom viewports, we also have to update them when the window size changes!
    bool resizeWindow(Uint32 windowId, int width, int height) override
    {
        rw = width;
        rh = height;
        setupCameras();
        return false;
    }
};
