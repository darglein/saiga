﻿/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/animation/cameraAnimation.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/ffmpeg/ffmpegEncoder.h"
#include "saiga/opengl/ffmpeg/videoEncoder.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/overlay/deferredDebugOverlay.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/window/sdl_window.h"
#include "saiga/opengl/world/proceduralSkybox.h"
#include "saiga/core/sdl/sdl_camera.h"
#include "saiga/core/sdl/sdl_eventhandler.h"

using namespace Saiga;


class Sample : public Updating, public DeferredRenderingInterface, public SDL_KeyListener
{
   public:
    SDLCamera<PerspectiveCamera> camera;

    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
    SimpleAssetObject sphere;

    ProceduralSkybox skybox;



    std::shared_ptr<DirectionalLight> sun;

    int remainingFrames;
    bool rotateCamera = false;
    int frame         = 0;
    int frameSkip     = 0;
    std::shared_ptr<FFMPEGEncoder> encoder;

    VideoEncoder enc;



    Interpolation cameraInterpolation;

    Sample(OpenGLWindow& window, Renderer& renderer);
    ~Sample();

    void testBspline();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void render(Camera* cam) override;
    void renderDepth(Camera* cam) override;
    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};